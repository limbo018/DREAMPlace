##
# @file   macro_alignment.py
# @author BeyondPPA/MPC Integration
# @brief  Macro grid alignment cost operator.
#         Penalizes macros that deviate from virtual grid anchor points,
#         promoting structural regularity and improving clock tree synthesis.
#         Based on Eq. 7-8 from BeyondPPA (2025 MLCAD).

import torch


class MacroAlignment(object):
    """
    @brief Computes macro grid alignment deviation cost.
    For each macro, computes the distance to the nearest virtual grid anchor.
    Uses modular arithmetic for efficiency: deviation = pos mod grid_step.
    The cost is the sum of squared deviations from the nearest grid lines.
    """

    def __init__(self, xl, yl, xh, yh, grid_step_x, grid_step_y,
                 num_movable_nodes, num_nodes):
        """
        @brief initialization
        @param xl, yl layout lower-left corner
        @param xh, yh layout upper-right corner
        @param grid_step_x horizontal grid spacing (0 = auto from layout)
        @param grid_step_y vertical grid spacing (0 = auto from layout)
        @param num_movable_nodes number of movable nodes
        @param num_nodes total number of nodes
        """
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
        self.num_movable_nodes = num_movable_nodes
        self.num_nodes = num_nodes

        # Auto-compute grid step if not specified
        # Use ~10 grid divisions across the layout
        layout_w = xh - xl
        layout_h = yh - yl
        self.grid_step_x = grid_step_x if grid_step_x > 0 else layout_w / 10.0
        self.grid_step_y = grid_step_y if grid_step_y > 0 else layout_h / 10.0

    def __call__(self, pos, macro_mask):
        """
        @brief compute grid alignment cost
        @param pos cell positions, array of (x_1, ..., x_N, y_1, ..., y_N)
        @param macro_mask boolean mask identifying movable macros
        @return scalar cost tensor
        """
        if macro_mask is None or macro_mask.sum() == 0:
            return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

        # Extract macro positions (lower-left corner)
        macro_x = pos[:self.num_movable_nodes][macro_mask]
        macro_y = pos[self.num_nodes:self.num_nodes + self.num_movable_nodes][macro_mask]

        # Compute deviation from nearest grid line using modular arithmetic
        # offset_x is in [0, grid_step_x), deviation is min(offset, grid_step - offset)
        offset_x = (macro_x - self.xl) % self.grid_step_x
        dev_x = torch.min(offset_x, self.grid_step_x - offset_x)

        offset_y = (macro_y - self.yl) % self.grid_step_y
        dev_y = torch.min(offset_y, self.grid_step_y - offset_y)

        # Sum of squared deviations (smooth, differentiable)
        cost = (dev_x * dev_x + dev_y * dev_y).sum()

        return cost
