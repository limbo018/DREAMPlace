##
# @file   io_keepout.py
# @author BeyondPPA/MPC Integration
# @brief  IO keepout zone violation cost operator.
#         Penalizes macros placed too close to IO pins (terminals),
#         preventing buffer insertion blockage and noise coupling.
#         Based on Eq. 5-6 from BeyondPPA (2025 MLCAD).

import torch


class IOKeepout(object):
    """
    @brief Computes IO keepout zone violation cost.
    For each macro, checks proximity to IO terminal pins.
    Uses a smooth sigmoid approximation for differentiability:
        C(Mi, K) ~ sum_j sigmoid(-(dist(Mi, Kj) - margin) / temperature)
    """

    def __init__(self, terminal_pos_x, terminal_pos_y, macro_size_x, macro_size_y,
                 num_movable_nodes, num_terminals, num_nodes, keepout_margin,
                 temperature=1.0):
        """
        @brief initialization
        @param terminal_pos_x x coordinates of terminal/IO pins (fixed)
        @param terminal_pos_y y coordinates of terminal/IO pins (fixed)
        @param macro_size_x width of macro nodes
        @param macro_size_y height of macro nodes
        @param num_movable_nodes number of movable nodes
        @param num_terminals number of terminal nodes
        @param num_nodes total number of nodes (for pos indexing)
        @param keepout_margin minimum clearance distance from IO pins
        @param temperature smoothing temperature for sigmoid
        """
        self.terminal_pos_x = terminal_pos_x
        self.terminal_pos_y = terminal_pos_y
        self.macro_size_x = macro_size_x
        self.macro_size_y = macro_size_y
        self.num_movable_nodes = num_movable_nodes
        self.num_terminals = num_terminals
        self.num_nodes = num_nodes
        self.keepout_margin = keepout_margin
        self.temperature = temperature

    def __call__(self, pos, macro_mask):
        """
        @brief compute IO keepout violation cost
        @param pos cell positions, array of (x_1, ..., x_N, y_1, ..., y_N)
        @param macro_mask boolean mask identifying movable macros among movable nodes
        @return scalar cost tensor
        """
        if macro_mask is None or macro_mask.sum() == 0:
            return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

        # Extract macro center positions
        macro_x = pos[:self.num_movable_nodes][macro_mask]
        macro_y = pos[self.num_nodes:self.num_nodes + self.num_movable_nodes][macro_mask]
        macro_w = self.macro_size_x[:self.num_movable_nodes][macro_mask]
        macro_h = self.macro_size_y[:self.num_movable_nodes][macro_mask]

        # Macro centers (pos stores lower-left corner)
        macro_cx = macro_x + macro_w / 2.0
        macro_cy = macro_y + macro_h / 2.0

        num_macros = macro_cx.size(0)
        num_terminals = self.terminal_pos_x.size(0)

        if num_terminals == 0:
            return torch.tensor(0.0, device=pos.device, dtype=pos.dtype)

        # Compute pairwise distances between macro centers and terminal positions
        # Shape: [num_macros, num_terminals]
        dx = macro_cx.unsqueeze(1) - self.terminal_pos_x.unsqueeze(0)
        dy = macro_cy.unsqueeze(1) - self.terminal_pos_y.unsqueeze(0)

        # Use L-infinity (Chebyshev) distance for rectangular keepout zones
        # accounting for macro half-widths
        half_w = (macro_w / 2.0).unsqueeze(1)
        half_h = (macro_h / 2.0).unsqueeze(1)
        dist_x = dx.abs() - half_w
        dist_y = dy.abs() - half_h
        # Clamp to zero: if macro overlaps the terminal, distance is 0
        dist_x = dist_x.clamp(min=0)
        dist_y = dist_y.clamp(min=0)
        dist = torch.sqrt(dist_x * dist_x + dist_y * dist_y + 1e-8)

        # Smooth violation: sigmoid(-(dist - margin) / temperature)
        # When dist < margin, violation is high; when dist > margin, violation is low
        violation = torch.sigmoid(-(dist - self.keepout_margin) / self.temperature)

        # Sum over all macro-terminal pairs
        cost = violation.sum()

        return cost
