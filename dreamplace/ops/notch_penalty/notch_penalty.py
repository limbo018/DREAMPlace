##
# @file   notch_penalty.py
# @author BeyondPPA/MPC Integration
# @brief  Notch proximity penalty operator.
#         Penalizes narrow concave gaps between adjacent macros that cause
#         routing hotspots, EM stress, and power grid discontinuities.
#         Based on Eq. 9-10 from BeyondPPA (2025 MLCAD).

import torch


class NotchPenalty(object):
    """
    @brief Computes notch proximity penalty cost.
    For each pair of macros, penalizes edge-to-edge distances below a threshold.
    Uses a smooth sigmoid approximation:
        S(Mi) = sum_{j!=i} sigmoid(-(dij - delta_notch) / temperature)
    where dij is the edge-to-edge distance between macros Mi and Mj.
    """

    def __init__(self, macro_size_x, macro_size_y,
                 num_movable_nodes, num_nodes,
                 notch_threshold, temperature=1.0):
        """
        @brief initialization
        @param macro_size_x width of nodes
        @param macro_size_y height of nodes
        @param num_movable_nodes number of movable nodes
        @param num_nodes total number of nodes
        @param notch_threshold minimum acceptable gap between macros (delta_notch)
        @param temperature smoothing temperature for sigmoid
        """
        self.macro_size_x = macro_size_x
        self.macro_size_y = macro_size_y
        self.num_movable_nodes = num_movable_nodes
        self.num_nodes = num_nodes
        self.notch_threshold = notch_threshold
        self.temperature = temperature

    def __call__(self, pos, macro_mask):
        """
        @brief compute notch penalty cost
        @param pos cell positions, array of (x_1, ..., x_N, y_1, ..., y_N)
        @param macro_mask boolean mask identifying movable macros
        @return scalar cost tensor
        """
        if macro_mask is None or macro_mask.sum() < 2:
            return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

        # Extract macro positions and sizes
        macro_x = pos[:self.num_movable_nodes][macro_mask]
        macro_y = pos[self.num_nodes:self.num_nodes + self.num_movable_nodes][macro_mask]
        macro_w = self.macro_size_x[:self.num_movable_nodes][macro_mask]
        macro_h = self.macro_size_y[:self.num_movable_nodes][macro_mask]

        num_macros = macro_x.size(0)

        # Compute pairwise edge-to-edge distances
        # For rectangles: gap = max(0, |cx_i - cx_j| - (w_i + w_j)/2) in x
        #                 gap = max(0, |cy_i - cy_j| - (h_i + h_j)/2) in y
        # Edge-to-edge distance = max(gap_x, gap_y) when non-overlapping
        #                       = sqrt(gap_x^2 + gap_y^2) for corner distances

        # Macro centers
        cx = macro_x + macro_w / 2.0
        cy = macro_y + macro_h / 2.0

        # Pairwise center differences [num_macros, num_macros]
        dx = cx.unsqueeze(1) - cx.unsqueeze(0)
        dy = cy.unsqueeze(1) - cy.unsqueeze(0)

        # Half-width/height sums for each pair
        hw_sum = (macro_w.unsqueeze(1) + macro_w.unsqueeze(0)) / 2.0
        hh_sum = (macro_h.unsqueeze(1) + macro_h.unsqueeze(0)) / 2.0

        # Edge-to-edge gap in each dimension
        gap_x = (dx.abs() - hw_sum).clamp(min=0)
        gap_y = (dy.abs() - hh_sum).clamp(min=0)

        # Euclidean edge-to-edge distance
        edge_dist = torch.sqrt(gap_x * gap_x + gap_y * gap_y + 1e-8)

        # Mask out self-pairs (diagonal)
        mask = ~torch.eye(num_macros, device=pos.device, dtype=torch.bool)
        edge_dist = edge_dist[mask].reshape(num_macros, num_macros - 1)

        # Smooth penalty: sigmoid(-(dist - threshold) / temperature)
        # Penalizes pairs closer than notch_threshold
        penalty = torch.sigmoid(-(edge_dist - self.notch_threshold) / self.temperature)

        # Sum all penalties (each pair counted once from each direction, divide by 2)
        cost = penalty.sum() / 2.0

        return cost
