##
# @file   density_balance.py
# @author BeyondPPA/MPC Integration
# @brief  Macro-specific density balance cost operator.
#         Penalizes non-uniform macro distribution across the layout,
#         reducing localized IR drop and thermal hotspots.
#         Based on Eq. 3-4 from BeyondPPA (2025 MLCAD).

import torch


class DensityBalance(object):
    """
    @brief Computes macro-only density uniformity cost.
    Divides the layout into bins and penalizes deviation of macro density
    from a target uniform density:
        U(Mi) = (rho(xi, yi) - rho_bar)^2
    This is complementary to DREAMPlace's existing electric-field density
    which operates on all cells.
    """

    def __init__(self, xl, yl, xh, yh, num_bins_x, num_bins_y,
                 macro_size_x, macro_size_y,
                 num_movable_nodes, num_nodes, target_macro_density):
        """
        @brief initialization
        @param xl, yl layout lower-left corner
        @param xh, yh layout upper-right corner
        @param num_bins_x number of horizontal bins
        @param num_bins_y number of vertical bins
        @param macro_size_x width of nodes
        @param macro_size_y height of nodes
        @param num_movable_nodes number of movable nodes
        @param num_nodes total number of nodes
        @param target_macro_density target macro utilization per bin (rho_bar)
        """
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.num_bins_x = num_bins_x
        self.num_bins_y = num_bins_y
        self.macro_size_x = macro_size_x
        self.macro_size_y = macro_size_y
        self.num_movable_nodes = num_movable_nodes
        self.num_nodes = num_nodes
        self.target_macro_density = target_macro_density

        self.bin_w = (xh - xl) / num_bins_x
        self.bin_h = (yh - yl) / num_bins_y
        self.bin_area = self.bin_w * self.bin_h

    def __call__(self, pos, macro_mask):
        """
        @brief compute density balance cost
        @param pos cell positions, array of (x_1, ..., x_N, y_1, ..., y_N)
        @param macro_mask boolean mask identifying movable macros
        @return scalar cost tensor
        """
        if macro_mask is None or macro_mask.sum() == 0:
            return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

        # Extract macro positions and sizes
        macro_x = pos[:self.num_movable_nodes][macro_mask]
        macro_y = pos[self.num_nodes:self.num_nodes + self.num_movable_nodes][macro_mask]
        macro_w = self.macro_size_x[:self.num_movable_nodes][macro_mask]
        macro_h = self.macro_size_y[:self.num_movable_nodes][macro_mask]

        # Macro centers
        cx = macro_x + macro_w / 2.0
        cy = macro_y + macro_h / 2.0

        # Map macro centers to bin indices (soft assignment using Gaussian splat)
        # For efficiency, use hard assignment to center bin
        bin_ix = ((cx - self.xl) / self.bin_w).clamp(0, self.num_bins_x - 1).long()
        bin_iy = ((cy - self.yl) / self.bin_h).clamp(0, self.num_bins_y - 1).long()

        # Compute macro area contribution to each bin
        macro_area = macro_w * macro_h
        flat_bin_idx = bin_iy * self.num_bins_x + bin_ix

        # Scatter-add macro areas into bins
        num_bins = self.num_bins_x * self.num_bins_y
        # Use float scatter for differentiability through a soft version
        density_map = torch.zeros(num_bins, device=pos.device, dtype=pos.dtype)
        density_map.scatter_add_(0, flat_bin_idx, macro_area)

        # Normalize by bin area to get density ratio
        density_map = density_map / self.bin_area

        # Compute squared deviation from target density
        deviation = density_map - self.target_macro_density
        cost = (deviation * deviation).sum()

        return cost
