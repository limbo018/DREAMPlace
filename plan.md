# Plan: MPC-Based BeyondPPA Feature Integration into DREAMPlace

## Executive Summary

Integrate the five structural reliability features from the BeyondPPA paper (wirelength, density uniformity, IO keepout, macro grid alignment, notch avoidance) into DREAMPlace using **Model Predictive Control (MPC)** instead of the paper's RL/DQN approach. MPC gives us the multi-objective optimization benefits without RL training overhead, keeping runtime low by leveraging DREAMPlace's existing GPU-accelerated gradient infrastructure.

---

## Why MPC Instead of RL

| Aspect | RL (Paper's DQN) | MPC (Our Approach) |
|--------|-------------------|---------------------|
| Runtime | 0.75–4.0 hrs extra on top of DREAMPlace | ~5–15 min post-processing (short horizon, no training) |
| Training | Requires offline training + replay buffer | Zero training — online optimization each step |
| GPU utilization | Separate CNN/GNN forward passes | Reuses DREAMPlace's existing CUDA kernels |
| Constraint handling | Learned implicitly via reward shaping | Explicit constraint enforcement per horizon step |
| Adaptivity | Fixed policy after training | Replans every step — adapts to current layout state |
| Integration | Separate codebase, loose coupling | Plugs directly into DREAMPlace's optimization loop |

**MPC key idea**: At each step, solve a short-horizon optimization problem (3–5 macro moves ahead), apply only the first move, re-evaluate the structural cost J, and replan. This "receding horizon" strategy naturally balances the five competing BeyondPPA objectives without needing a trained neural policy.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              Existing DREAMPlace Flow                │
│  Global Placement → Legalization → Detailed Place   │
└──────────────────────┬──────────────────────────────┘
                       │ macro positions (after legalization)
                       ▼
┌─────────────────────────────────────────────────────┐
│          NEW: MPC Macro Refinement Stage             │
│                                                     │
│  ┌──────────┐    ┌─────────────┐    ┌────────────┐ │
│  │ Structural│───▶│ MPC Horizon │───▶│ Apply Best │ │
│  │ Feature   │    │ Optimizer   │    │ First Move │ │
│  │ Extractor │◀───│ (N=3–5 steps│◀───│ & Replan   │ │
│  └──────────┘    └─────────────┘    └────────────┘ │
│       │                                     │       │
│       ▼                                     ▼       │
│  J = λ₁·Jwire + λ₂·Jdensity + λ₃·Jkeepout        │
│    + λ₄·Jalign + λ₅·Jnotch                         │
│                                                     │
│  Adaptive λ update every K MPC iterations           │
└─────────────────────────────────────────────────────┘
                       │
                       ▼
              Re-legalize macros → Detailed Placement
```

---

## Step-by-Step Implementation Plan

### Step 1: Structural Feature Operators (New CUDA/C++ Ops)

Create four new ops under `dreamplace/ops/` (wirelength already exists):

#### 1a. `dreamplace/ops/io_keepout/io_keepout.py`
- **What**: Computes IO keepout zone violation cost `Jkeepout` (Eq. 5-6 from paper)
- **Inputs**: macro positions `pos[2N]`, IO pin positions from `placedb.node_x/y` for `terminal_NI` nodes, keepout margin parameter
- **Computation**: For each macro, check overlap with keepout rectangles around IO pins. Sum of indicator violations, made differentiable via a smooth sigmoid approximation: `C(Mi,K) ≈ Σ_j sigmoid(-(dist(Mi,Kj) - margin) / temperature)`
- **Output**: scalar cost tensor (differentiable)
- **Implementation**: Pure PyTorch on GPU (no custom CUDA needed — it's a distance computation over ~100s of macros vs ~1000s of IOs, fast enough)
- **Runtime impact**: Negligible — O(N_macros × N_io) with N typically < 10K

#### 1b. `dreamplace/ops/macro_alignment/macro_alignment.py`
- **What**: Computes grid alignment deviation cost `Jalign` (Eq. 7-8 from paper)
- **Inputs**: macro positions, virtual grid spacing parameters (derived from `row_height` and `site_width` multiples)
- **Computation**: For each macro, compute distance to nearest grid anchor: `D(Mi,G) = min_{(gx,gy)∈G} sqrt((xi-gx)² + (yi-gy)²)`. Use soft-min via LogSumExp for differentiability.
- **Output**: scalar cost tensor
- **Implementation**: Pure PyTorch — grid anchors are precomputed once, distance is vectorized
- **Runtime impact**: Negligible — O(N_macros × N_grid_points), can use KD-tree or just modular arithmetic

#### 1c. `dreamplace/ops/notch_penalty/notch_penalty.py`
- **What**: Computes notch proximity penalty `Jnotch` (Eq. 9-10 from paper)
- **Inputs**: macro positions, macro sizes, notch threshold `δ_notch`
- **Computation**: For each macro pair, compute edge-to-edge distance. Penalize pairs closer than `δ_notch`: `S(Mi) = Σ_{j≠i} sigmoid(-(dij - δ_notch) / temp)`. This smooth approximation replaces the hard indicator `1{dij < δ_notch}`.
- **Output**: scalar cost tensor
- **Implementation**: Pure PyTorch — O(N_macros²) but N_macros is typically 10–500, so this is fast
- **Runtime impact**: Negligible for typical macro counts

#### 1d. `dreamplace/ops/density_balance/density_balance.py`
- **What**: Enhanced density uniformity cost `Jdensity` (Eq. 3-4 from paper) — different from DREAMPlace's electric potential density, this specifically targets macro-level density balance for IR drop mitigation
- **Inputs**: macro positions, macro sizes, grid bin dimensions
- **Computation**: Compute macro-only density per bin, then penalize deviation from uniform: `U(Mi) = (ρ(xi,yi) - ρ̄)²`. This is complementary to the existing density op (which handles all cells).
- **Output**: scalar cost tensor
- **Implementation**: Reuse DREAMPlace's density_map infrastructure, filter to macro nodes only
- **Runtime impact**: Negligible — reuses existing bin infrastructure

### Step 2: MPC Controller Module

#### 2a. `dreamplace/MPCMacroRefiner.py` — Core MPC Engine

```python
class MPCMacroRefiner:
    """
    Model Predictive Control for macro refinement.
    Operates AFTER DREAMPlace global placement + legalization.
    """
    def __init__(self, params, placedb, data_collections):
        # MPC parameters
        self.horizon = params.mpc_horizon          # 3-5 steps (default: 3)
        self.max_iterations = params.mpc_max_iter   # 50-200 (default: 100)
        self.num_candidates = params.mpc_candidates # candidate moves per macro (default: 8)
        self.lambda_update_interval = params.mpc_lambda_interval  # default: 10

        # Build structural feature ops
        self.io_keepout_op = IOKeepout(...)
        self.macro_align_op = MacroAlignment(...)
        self.notch_penalty_op = NotchPenalty(...)
        self.density_balance_op = DensityBalance(...)
        # Reuse existing wirelength op
        self.wirelength_op = data_collections.wirelength_op

        # Adaptive weights (λ₁₋₅)
        self.lambdas = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
```

**MPC Loop (pseudocode)**:
```
for iter in range(max_iterations):
    # 1. Evaluate current structural cost J
    J_current = compute_total_cost(macro_pos)

    # 2. For each macro, generate candidate moves
    #    (grid-snapped shifts, corner placements, spacing adjustments)
    candidates = generate_candidates(macro_pos, num_candidates)

    # 3. Horizon rollout: simulate N steps ahead
    #    Pick greedy-best sequence over horizon
    best_sequence = horizon_rollout(macro_pos, candidates, horizon)

    # 4. Apply ONLY the first move (receding horizon)
    macro_pos = apply_move(macro_pos, best_sequence[0])

    # 5. Enforce legality (no overlaps, within boundary)
    macro_pos = enforce_constraints(macro_pos)

    # 6. Adaptive weight update (every K iterations)
    if iter % lambda_update_interval == 0:
        update_lambdas(J_components)

    # 7. Early stopping if cost converged
    if J_current - J_new < tolerance:
        break
```

#### Key Design Decisions for Low Runtime:

1. **Short horizon (N=3)**: Each rollout evaluates only 3 moves ahead — enough for lookahead without combinatorial explosion
2. **Limited candidate set (8 per macro)**: Only evaluate meaningful moves — grid-aligned shifts in 4 cardinal directions + 4 diagonal, snapped to alignment grid
3. **Greedy horizon search**: Instead of full tree search (exponential), use greedy beam search with beam width 4
4. **Macro priority ordering**: Process macros by "violation score" — macros with worst structural violations get refined first, skip macros that are already well-placed
5. **GPU batch evaluation**: Evaluate all candidates for a macro in a single batched forward pass
6. **Early termination**: Stop when structural cost improvement drops below threshold

#### 2b. Candidate Move Generation Strategy

```
For each macro Mi with position (xi, yi):
  candidates = [
    (xi ± grid_step_x, yi),           # horizontal grid shifts
    (xi, yi ± grid_step_y),           # vertical grid shifts
    (xi ± grid_step_x, yi ± grid_step_y),  # diagonal shifts
    snap_to_nearest_grid(xi, yi),      # alignment correction
    move_away_from_nearest_io(xi, yi), # IO keepout correction
    move_away_from_nearest_macro(xi, yi, δ_notch),  # notch correction
  ]
  Filter out illegal candidates (overlaps, boundary violations)
```

### Step 3: Adaptive Weight Controller

#### 3a. `dreamplace/AdaptiveLambdaController.py`

Implements the paper's dynamic weight adjustment (Sec. III-B.5) within MPC:

```python
class AdaptiveLambdaController:
    """
    Sensitivity-based adaptive weight tuning.
    Every K iterations, estimate ∂J/∂λᵢ and shift weights
    toward under-optimized objectives.
    """
    def update(self, J_components, lambdas, eta=0.05):
        # Compute per-feature improvement rate
        delta = J_components_prev - J_components_current

        # Features that improved less get higher weight
        sensitivity = 1.0 / (delta + epsilon)
        sensitivity = sensitivity / sensitivity.sum()  # normalize

        # Update: λᵢ ← λᵢ + η · Δᵢ
        lambdas = lambdas + eta * sensitivity
        lambdas = lambdas / lambdas.sum() * len(lambdas)  # re-normalize

        return lambdas
```

### Step 4: Integration into DREAMPlace Pipeline

#### 4a. Modify `dreamplace/NonLinearPlace.py`

Insert MPC refinement after macro legalization (around line 828) and before detailed placement:

```python
# --- EXISTING CODE ---
# After macro legalization (line ~828)
# ...

# --- NEW: MPC Macro Refinement ---
if params.mpc_refine_flag:
    mpc_refiner = MPCMacroRefiner(params, placedb, data_collections)
    pos = mpc_refiner.refine(pos)  # returns refined macro positions
    # Re-legalize after MPC refinement
    pos = macro_legalize_op(init_pos, pos)

# --- EXISTING CODE continues ---
# Legalization, detailed placement, etc.
```

#### 4b. Modify `dreamplace/Params.py` / `dreamplace/params.json`

Add new MPC parameters:

```json
{
    "mpc_refine_flag": 1,
    "mpc_horizon": 3,
    "mpc_max_iter": 100,
    "mpc_candidates": 8,
    "mpc_lambda_interval": 10,
    "mpc_lambda_eta": 0.05,
    "mpc_io_keepout_margin": 10.0,
    "mpc_grid_step_x": 0,
    "mpc_grid_step_y": 0,
    "mpc_notch_threshold": 5.0,
    "mpc_target_macro_density": 0.5,
    "mpc_early_stop_tol": 1e-3,
    "mpc_lambda_init": [1.0, 1.0, 1.0, 1.0, 1.0]
}
```

#### 4c. Modify `dreamplace/EvalMetrics.py`

Add new structural metrics for tracking:

```python
# New fields in EvalMetrics class
self.io_keepout_cost = None      # IO keepout violation
self.macro_alignment_cost = None  # Grid alignment deviation
self.notch_penalty_cost = None    # Notch proximity penalty
self.density_balance_cost = None  # Macro density uniformity
self.structural_cost_J = None     # Total structural cost
```

### Step 5: Configuration and Testing

#### 5a. Test configurations for ISPD benchmarks

Create test configs that enable MPC refinement:
- `test/ispd2005/adaptec1_mpc.json` — enable `mpc_refine_flag`, tune keepout margin based on IO pin distribution
- Parameters auto-derived where possible: `mpc_grid_step_x/y = 0` means auto-compute from `row_height` and average macro size

#### 5b. Logging and visualization

- Log all 5 structural cost components per MPC iteration
- Extend `draw_place` op to visualize IO keepout zones, grid lines, and notch regions
- Print runtime breakdown: DREAMPlace base time vs MPC refinement time

---

## Files to Create (New)

| File | Purpose | Est. Lines |
|------|---------|------------|
| `dreamplace/ops/io_keepout/io_keepout.py` | IO keepout zone violation op | ~80 |
| `dreamplace/ops/macro_alignment/macro_alignment.py` | Grid alignment cost op | ~70 |
| `dreamplace/ops/notch_penalty/notch_penalty.py` | Notch proximity penalty op | ~75 |
| `dreamplace/ops/density_balance/density_balance.py` | Macro density balance op | ~80 |
| `dreamplace/MPCMacroRefiner.py` | Core MPC controller | ~350 |
| `dreamplace/AdaptiveLambdaController.py` | Adaptive weight tuning | ~60 |

## Files to Modify (Existing)

| File | Change | Lines Affected |
|------|--------|----------------|
| `dreamplace/NonLinearPlace.py` | Insert MPC call after macro legalization | ~15 lines added near line 828 |
| `dreamplace/Params.py` | Add MPC parameter defaults | ~20 lines |
| `dreamplace/params.json` | Add MPC default config | ~15 lines |
| `dreamplace/EvalMetrics.py` | Add structural metric fields | ~10 lines |
| `dreamplace/BasicPlace.py` | Build MPC ops in PlaceOpCollection | ~15 lines |

---

## Runtime Estimate

| Component | Estimated Time | Notes |
|-----------|---------------|-------|
| DREAMPlace base (global + legalize) | 5–30 min | Unchanged |
| Structural feature extraction | < 1 sec/iter | Pure PyTorch on GPU, small macro count |
| MPC horizon rollout (per iter) | ~0.5 sec | 3-step horizon × 8 candidates × batch eval |
| Full MPC refinement (100 iters) | ~1–3 min | With early stopping typically 30-60 iters |
| Re-legalization | < 30 sec | Existing macro_legalize |
| **Total overhead** | **~2–5 min** | **<10% of base DREAMPlace time** |

Compared to BeyondPPA's RL approach (0.75–4.0 hours extra), this is approximately **10-50x faster**.

---

## Implementation Order

1. **Phase 1** — Feature Ops (Steps 1a-1d): Build and unit-test each structural cost operator independently
2. **Phase 2** — MPC Core (Steps 2a-2b): Build the MPC controller with candidate generation, horizon rollout, and constraint enforcement
3. **Phase 3** — Adaptive Weights (Step 3a): Add the sensitivity-based lambda controller
4. **Phase 4** — Integration (Steps 4a-4c): Wire MPC into the DREAMPlace pipeline, add params, metrics
5. **Phase 5** — Test & Tune (Step 5): Run on ISPD benchmarks, tune MPC hyperparameters
