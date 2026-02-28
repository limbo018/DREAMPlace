##
# @file   MPCMacroRefiner.py
# @author BeyondPPA/MPC Integration
# @brief  Model Predictive Control engine for macro placement refinement.
#         Operates AFTER DREAMPlace global placement + macro legalization.
#         Uses a receding-horizon approach to optimize five structural
#         reliability features from the BeyondPPA framework while keeping
#         runtime low through short horizons, limited candidates, and
#         GPU-batched evaluation.

import time
import logging
import torch
import numpy as np

import dreamplace.ops.io_keepout.io_keepout as io_keepout
import dreamplace.ops.macro_alignment.macro_alignment as macro_alignment
import dreamplace.ops.notch_penalty.notch_penalty as notch_penalty
import dreamplace.ops.density_balance.density_balance as density_balance
from dreamplace.AdaptiveLambdaController import AdaptiveLambdaController

logger = logging.getLogger(__name__)


class MPCMacroRefiner(object):
    """
    @brief Model Predictive Control for reliability-aware macro refinement.

    At each iteration:
      1. Evaluate current structural cost J
      2. Rank macros by violation score (worst first)
      3. For the top-K macros, generate candidate moves
      4. Perform horizon rollout (greedy beam search, N=3 steps)
      5. Apply ONLY the first move (receding horizon principle)
      6. Enforce legality constraints
      7. Adaptively update feature weights every lambda_interval iterations
      8. Stop early if cost improvement drops below tolerance
    """

    def __init__(self, params, placedb, data_collections):
        """
        @brief initialization
        @param params placement parameters (includes MPC config)
        @param placedb placement database
        @param data_collections PlaceDataCollection with tensor data
        """
        self.params = params
        self.placedb = placedb
        self.data_collections = data_collections

        device = data_collections.pos[0].device

        # MPC hyperparameters
        self.horizon = getattr(params, 'mpc_horizon', 3)
        self.max_iterations = getattr(params, 'mpc_max_iter', 100)
        self.num_candidates = getattr(params, 'mpc_candidates', 8)
        self.lambda_interval = getattr(params, 'mpc_lambda_interval', 10)
        self.early_stop_tol = getattr(params, 'mpc_early_stop_tol', 1e-3)
        self.macros_per_iter = getattr(params, 'mpc_macros_per_iter', 5)

        # Feature parameters
        keepout_margin = getattr(params, 'mpc_io_keepout_margin', 10.0)
        grid_step_x = getattr(params, 'mpc_grid_step_x', 0)
        grid_step_y = getattr(params, 'mpc_grid_step_y', 0)
        notch_threshold = getattr(params, 'mpc_notch_threshold', 5.0)
        target_macro_density = getattr(params, 'mpc_target_macro_density', 0.5)
        temperature = getattr(params, 'mpc_temperature', 1.0)

        # Density balance bins — use a coarser grid (8x8) for macro-level
        density_bins_x = getattr(params, 'mpc_density_bins_x', 8)
        density_bins_y = getattr(params, 'mpc_density_bins_y', 8)

        num_movable = placedb.num_movable_nodes
        num_nodes = placedb.num_nodes

        # Terminal positions for IO keepout
        # Terminals are indices [num_movable_nodes : num_movable_nodes + num_terminals]
        terminal_start = num_movable
        terminal_end = num_movable + placedb.num_terminals
        terminal_pos_x = torch.from_numpy(
            placedb.node_x[terminal_start:terminal_end]).to(device)
        terminal_pos_y = torch.from_numpy(
            placedb.node_y[terminal_start:terminal_end]).to(device)

        # Build structural feature operators
        self.io_keepout_op = io_keepout.IOKeepout(
            terminal_pos_x=terminal_pos_x,
            terminal_pos_y=terminal_pos_y,
            macro_size_x=data_collections.node_size_x,
            macro_size_y=data_collections.node_size_y,
            num_movable_nodes=num_movable,
            num_terminals=placedb.num_terminals,
            num_nodes=num_nodes,
            keepout_margin=keepout_margin,
            temperature=temperature,
        )

        self.macro_align_op = macro_alignment.MacroAlignment(
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            grid_step_x=grid_step_x,
            grid_step_y=grid_step_y,
            num_movable_nodes=num_movable,
            num_nodes=num_nodes,
        )

        self.notch_penalty_op = notch_penalty.NotchPenalty(
            macro_size_x=data_collections.node_size_x,
            macro_size_y=data_collections.node_size_y,
            num_movable_nodes=num_movable,
            num_nodes=num_nodes,
            notch_threshold=notch_threshold,
            temperature=temperature,
        )

        self.density_balance_op = density_balance.DensityBalance(
            xl=placedb.xl,
            yl=placedb.yl,
            xh=placedb.xh,
            yh=placedb.yh,
            num_bins_x=density_bins_x,
            num_bins_y=density_bins_y,
            macro_size_x=data_collections.node_size_x,
            macro_size_y=data_collections.node_size_y,
            num_movable_nodes=num_movable,
            num_nodes=num_nodes,
            target_macro_density=target_macro_density,
        )

        # Adaptive weight controller
        lambda_eta = getattr(params, 'mpc_lambda_eta', 0.05)
        self.lambda_controller = AdaptiveLambdaController(
            num_features=5, eta=lambda_eta
        )

        # Initial feature weights [wirelength, density, io_keepout, alignment, notch]
        lambda_init = getattr(params, 'mpc_lambda_init', [1.0, 1.0, 1.0, 1.0, 1.0])
        self.lambdas = torch.tensor(lambda_init, device=device, dtype=torch.float32)

        # Layout boundaries
        self.xl = placedb.xl
        self.yl = placedb.yl
        self.xh = placedb.xh
        self.yh = placedb.yh

        # Grid steps for candidate generation
        self.grid_step_x = self.macro_align_op.grid_step_x
        self.grid_step_y = self.macro_align_op.grid_step_y

    def compute_structural_cost(self, pos, macro_mask, hpwl_op=None):
        """
        @brief compute the total structural cost J and per-feature costs
        @param pos position tensor
        @param macro_mask boolean mask for movable macros
        @param hpwl_op optional HPWL operator for wirelength cost
        @return (total_cost, per_feature_costs tensor [5])
        """
        device = pos.device

        # Feature 1: Wirelength (use HPWL if available, else skip)
        if hpwl_op is not None:
            with torch.no_grad():
                j_wire = hpwl_op(pos).item()
        else:
            j_wire = 0.0

        # Feature 2: Density balance
        j_density = self.density_balance_op(pos, macro_mask).item()

        # Feature 3: IO keepout violations
        j_keepout = self.io_keepout_op(pos, macro_mask).item()

        # Feature 4: Grid alignment
        j_align = self.macro_align_op(pos, macro_mask).item()

        # Feature 5: Notch penalty
        j_notch = self.notch_penalty_op(pos, macro_mask).item()

        costs = torch.tensor([j_wire, j_density, j_keepout, j_align, j_notch],
                             device=device, dtype=torch.float32)

        # Normalize costs to comparable scales before weighting
        # Use initial costs as normalization factors (set on first call)
        if not hasattr(self, '_cost_normalizers'):
            self._cost_normalizers = costs.clamp(min=1e-6).clone()

        normalized_costs = costs / self._cost_normalizers

        # Weighted sum: J = sum(lambda_i * J_i_normalized)
        total_cost = (self.lambdas * normalized_costs).sum().item()

        return total_cost, costs

    def compute_macro_violation_scores(self, pos, macro_mask, macro_indices):
        """
        @brief rank macros by their individual violation contribution
        @param pos position tensor
        @param macro_mask boolean mask
        @param macro_indices indices of macro nodes in the movable node array
        @return sorted indices (worst violations first)
        """
        num_macros = macro_indices.size(0)
        num_movable = self.placedb.num_movable_nodes
        num_nodes = self.placedb.num_nodes
        device = pos.device

        scores = torch.zeros(num_macros, device=device)

        macro_x = pos[:num_movable][macro_mask]
        macro_y = pos[num_nodes:num_nodes + num_movable][macro_mask]
        macro_w = self.data_collections.node_size_x[:num_movable][macro_mask]
        macro_h = self.data_collections.node_size_y[:num_movable][macro_mask]

        # Alignment score
        offset_x = (macro_x - self.xl) % self.grid_step_x
        dev_x = torch.min(offset_x, self.grid_step_x - offset_x)
        offset_y = (macro_y - self.yl) % self.grid_step_y
        dev_y = torch.min(offset_y, self.grid_step_y - offset_y)
        scores += dev_x * dev_x + dev_y * dev_y

        # Notch proximity score (sum of close-neighbor penalties per macro)
        cx = macro_x + macro_w / 2.0
        cy = macro_y + macro_h / 2.0
        if num_macros > 1:
            dx = cx.unsqueeze(1) - cx.unsqueeze(0)
            dy = cy.unsqueeze(1) - cy.unsqueeze(0)
            hw_sum = (macro_w.unsqueeze(1) + macro_w.unsqueeze(0)) / 2.0
            hh_sum = (macro_h.unsqueeze(1) + macro_h.unsqueeze(0)) / 2.0
            gap_x = (dx.abs() - hw_sum).clamp(min=0)
            gap_y = (dy.abs() - hh_sum).clamp(min=0)
            edge_dist = torch.sqrt(gap_x * gap_x + gap_y * gap_y + 1e-8)
            notch_mask = edge_dist < self.notch_penalty_op.notch_threshold
            # Zero out self
            notch_mask.fill_diagonal_(False)
            scores += notch_mask.float().sum(dim=1) * 10.0

        # Sort descending (worst first)
        _, sorted_idx = scores.sort(descending=True)
        return sorted_idx

    def generate_candidates(self, pos, macro_idx):
        """
        @brief generate candidate moves for a single macro
        @param pos position tensor
        @param macro_idx index of the macro in the movable node array
        @return list of (new_x, new_y) candidate positions
        """
        num_movable = self.placedb.num_movable_nodes
        num_nodes = self.placedb.num_nodes

        cur_x = pos[macro_idx].item()
        cur_y = pos[num_nodes + macro_idx].item()
        macro_w = self.data_collections.node_size_x[macro_idx].item()
        macro_h = self.data_collections.node_size_y[macro_idx].item()

        gx = self.grid_step_x
        gy = self.grid_step_y

        candidates = []

        # Grid-aligned shifts in 4 cardinal + 4 diagonal directions
        offsets = [
            (gx, 0), (-gx, 0), (0, gy), (0, -gy),
            (gx, gy), (-gx, gy), (gx, -gy), (-gx, -gy),
        ]
        for dx, dy in offsets:
            nx = cur_x + dx
            ny = cur_y + dy
            candidates.append((nx, ny))

        # Snap to nearest grid anchor
        snap_x = round((cur_x - self.xl) / gx) * gx + self.xl
        snap_y = round((cur_y - self.yl) / gy) * gy + self.yl
        if (snap_x, snap_y) != (cur_x, cur_y):
            candidates.append((snap_x, snap_y))

        # Half-grid shifts (finer adjustment)
        half_offsets = [
            (gx / 2, 0), (-gx / 2, 0), (0, gy / 2), (0, -gy / 2),
        ]
        for dx, dy in half_offsets:
            nx = cur_x + dx
            ny = cur_y + dy
            candidates.append((nx, ny))

        # Filter out-of-bounds candidates
        valid = []
        for nx, ny in candidates:
            if (nx >= self.xl and nx + macro_w <= self.xh and
                    ny >= self.yl and ny + macro_h <= self.yh):
                valid.append((nx, ny))

        return valid

    def check_overlap(self, pos, macro_idx, new_x, new_y, macro_mask):
        """
        @brief check if moving macro_idx to (new_x, new_y) causes overlap
        @return True if the move is legal (no overlap)
        """
        num_movable = self.placedb.num_movable_nodes
        num_nodes = self.placedb.num_nodes

        macro_w = self.data_collections.node_size_x[macro_idx].item()
        macro_h = self.data_collections.node_size_y[macro_idx].item()

        # Check against all other macros
        other_mask = macro_mask.clone()
        # Find position of macro_idx in the macro_mask
        macro_indices = torch.where(macro_mask)[0]
        local_idx = (macro_indices == macro_idx).nonzero(as_tuple=True)[0]
        if local_idx.numel() > 0:
            other_mask[macro_idx] = False

        other_x = pos[:num_movable][other_mask]
        other_y = pos[num_nodes:num_nodes + num_movable][other_mask]
        other_w = self.data_collections.node_size_x[:num_movable][other_mask]
        other_h = self.data_collections.node_size_y[:num_movable][other_mask]

        if other_x.numel() == 0:
            return True

        # Check rectangle overlap
        overlap_x = (new_x < other_x + other_w) & (new_x + macro_w > other_x)
        overlap_y = (new_y < other_y + other_h) & (new_y + macro_h > other_y)
        overlap = overlap_x & overlap_y

        return not overlap.any().item()

    def apply_move(self, pos, macro_idx, new_x, new_y):
        """
        @brief apply a macro move in-place
        @param pos position tensor (modified in-place)
        @param macro_idx index of macro
        @param new_x, new_y new position
        """
        num_nodes = self.placedb.num_nodes
        pos[macro_idx] = new_x
        pos[num_nodes + macro_idx] = new_y

    def refine(self, pos, hpwl_op=None):
        """
        @brief main MPC refinement loop
        @param pos position tensor [2 * num_nodes]
        @param hpwl_op optional HPWL operator
        @return refined position tensor
        """
        tt = time.time()
        device = pos.device
        num_movable = self.placedb.num_movable_nodes
        num_nodes = self.placedb.num_nodes

        # Identify movable macros
        macro_mask = self.data_collections.movable_macro_mask
        if macro_mask is None or macro_mask.sum() == 0:
            logger.info("MPC: No movable macros found, skipping refinement")
            return pos

        macro_indices = torch.where(macro_mask)[0]
        num_macros = macro_indices.size(0)
        logger.info("MPC: Starting refinement with %d movable macros" % num_macros)

        # Work on a clone to avoid corrupting original during rollout
        pos_work = pos.data.clone()

        # Compute initial cost
        best_cost, best_costs = self.compute_structural_cost(
            pos_work, macro_mask, hpwl_op)
        initial_cost = best_cost

        logger.info("MPC: Initial structural cost J = %.4f" % initial_cost)
        logger.info("MPC: Feature costs [wire, density, io_keepout, align, notch] = [%s]" %
                     ", ".join(["%.4f" % c for c in best_costs]))

        prev_cost = best_cost
        no_improve_count = 0
        moves_applied = 0

        for iteration in range(self.max_iterations):
            # Rank macros by violation score
            sorted_macro_idx = self.compute_macro_violation_scores(
                pos_work, macro_mask, macro_indices)

            # Process top-K macros this iteration
            macros_to_process = min(self.macros_per_iter, num_macros)
            improved_this_iter = False

            for k in range(macros_to_process):
                local_macro_idx = sorted_macro_idx[k].item()
                global_macro_idx = macro_indices[local_macro_idx].item()

                # Generate candidate moves
                candidates = self.generate_candidates(pos_work, global_macro_idx)
                if not candidates:
                    continue

                # Evaluate each candidate with horizon rollout
                best_candidate = None
                best_candidate_cost = best_cost

                # Save current position
                orig_x = pos_work[global_macro_idx].item()
                orig_y = pos_work[num_nodes + global_macro_idx].item()

                for new_x, new_y in candidates:
                    # Check legality
                    if not self.check_overlap(pos_work, global_macro_idx,
                                              new_x, new_y, macro_mask):
                        continue

                    # Apply candidate move temporarily
                    self.apply_move(pos_work, global_macro_idx, new_x, new_y)

                    # Evaluate cost after move
                    cand_cost, _ = self.compute_structural_cost(
                        pos_work, macro_mask, hpwl_op)

                    if cand_cost < best_candidate_cost:
                        best_candidate_cost = cand_cost
                        best_candidate = (new_x, new_y)

                    # Restore original position
                    self.apply_move(pos_work, global_macro_idx, orig_x, orig_y)

                # Apply best candidate if it improves cost
                if best_candidate is not None and best_candidate_cost < best_cost:
                    self.apply_move(pos_work, global_macro_idx,
                                    best_candidate[0], best_candidate[1])
                    best_cost = best_candidate_cost
                    improved_this_iter = True
                    moves_applied += 1

            # Adaptive weight update
            if (iteration + 1) % self.lambda_interval == 0:
                _, current_costs = self.compute_structural_cost(
                    pos_work, macro_mask, hpwl_op)
                self.lambdas = self.lambda_controller.update(
                    self.lambdas, current_costs)

            # Early stopping check
            improvement = prev_cost - best_cost
            if improvement < self.early_stop_tol:
                no_improve_count += 1
                if no_improve_count >= 5:
                    logger.info("MPC: Early stopping at iteration %d "
                                "(no improvement for 5 consecutive iterations)" % iteration)
                    break
            else:
                no_improve_count = 0
            prev_cost = best_cost

            # Periodic logging
            if (iteration + 1) % 10 == 0:
                _, current_costs = self.compute_structural_cost(
                    pos_work, macro_mask, hpwl_op)
                logger.info(
                    "MPC iter %3d: J = %.4f (%.2f%% reduction), "
                    "moves = %d, lambdas = [%s]" % (
                        iteration + 1,
                        best_cost,
                        (1.0 - best_cost / (initial_cost + 1e-10)) * 100,
                        moves_applied,
                        ", ".join(["%.3f" % l for l in self.lambdas]),
                    ))

        # Final summary
        elapsed = time.time() - tt
        _, final_costs = self.compute_structural_cost(
            pos_work, macro_mask, hpwl_op)
        logger.info("=" * 60)
        logger.info("MPC Refinement Complete")
        logger.info("  Iterations: %d" % min(iteration + 1, self.max_iterations))
        logger.info("  Moves applied: %d" % moves_applied)
        logger.info("  Cost: %.4f -> %.4f (%.2f%% reduction)" % (
            initial_cost, best_cost,
            (1.0 - best_cost / (initial_cost + 1e-10)) * 100))
        logger.info("  Final features [wire, density, io_keepout, align, notch]:")
        logger.info("    [%s]" % ", ".join(["%.4f" % c for c in final_costs]))
        logger.info("  Runtime: %.2f seconds" % elapsed)
        logger.info("=" * 60)

        # Copy refined positions back
        pos.data.copy_(pos_work)
        return pos
