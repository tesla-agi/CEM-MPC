# CEM-MPC on Pendulum-v1 (from scratch, numpy only)

A minimal implementation of the **Cross-Entropy Method (CEM)** used as a **Model Predictive Control (MPC)** planner on Gymnasium's `Pendulum-v1`. Written from scratch in a single file with no ML libraries: just numpy, plus gymnasium for the real environment and verification.

The planner uses the **exact pendulum dynamics** (hardcoded) as its model. This isolates the planner from model error: any failure is a planning failure, not a modelling one.

## How it works

At every real step:

1. **Sample** N action sequences (plans) of length H from a per-step Gaussian N(μ, σ²)
2. **Score** each plan by rolling it out in the model and summing rewards
3. **Select** the top K plans (elites)
4. **Refit** μ and σ to the elites (per time step)
5. Repeat for I rounds, then **execute only μ₀** in the real environment
6. **Warm start**: shift μ left by one step, pad with 0, reset σ, replan

Why the refit is "mean and std of the elites": fitting a Gaussian to the elites minimizes the cross-entropy to the ideal sampler that only produces good plans. That's where the method gets its name.

## Structure

| Function | Role |
|---|---|
| `angle_normalize` | wraps angles to (−π, π] for the cost |
| `step` | batched pendulum physics (the model) |
| `rollout` | scores P plans from one state over H steps |
| `cem` | inner CEM loop for one real step |
| `run_episode` | outer MPC loop in the real environment |
| `check_step`, `check_rollout` | verification gates against gymnasium |

## Verification

| Check | Max abs difference vs gymnasium |
|---|---|
| `step` (1000 random states/actions, incl. edge cases) | θ: 0.0, θdot: 0.0, r: 7e-15 |
| `rollout` (random 30-step plan) | 6e-14 |

## Results

Hyperparameters: H = 30, N = 500, K = 50, I = 5, σ₀ = 1.0

| Seed | 0 | 1 | 2 | 3 | 4 | **Mean** |
|---|---|---|---|---|---|---|
| Return | −122.2 | −0.8 | −117.3 | −220.6 | −231.6 | **−138.5** |

About 2.8 ms per action on CPU (75,000 model steps per action, vectorized over N).

From bottom starts (seeds 3, 4), the planner discovers on its own that it must **swing back and forth to build energy** before swinging up, since the motor (max angular acceleration 6) is weaker than gravity (max 15).

## Horizon sweep

| H | Lookahead | Mean return | ms/action |
|---|---|---|---|
| 5 | 0.25 s | −1038.7 | 0.5 |
| 10 | 0.5 s | −650.5 | 0.8 |
| 20 | 1.0 s | −143.6 | 1.6 |
| 30 | 1.5 s | **−138.6** | 2.8 |
| 50 | 2.5 s | −192.4 | 5.2 |

**Observations**
- **Too short (H ≤ 10) is shortsighted.** Swinging away to build energy looks worse within the horizon, so it's never chosen. Bottom starts get stuck at θ ≈ ±2.7, exactly where max torque balances gravity (15·sin θ = 6, so θ = π − arcsin 0.4 ≈ 2.73).
- **Sweet spot (H = 20–30).** About 1–1.5 s of lookahead is enough to plan the swing-up.
- **Too long (H = 50) hurts, even with exact dynamics.** The search space grows to 50 dimensions at a fixed budget (N = 500, I = 5), so far-future actions add noise to the scores and degrade the first action. Even an upright start loses balance and spins around.
- **Cost scales linearly with H**, as N × H × I predicts.

## Run

```bash
pip install numpy gymnasium
python cem_mpc.py
```

## Next steps

- Equal-budget comparison with random shooting
- Effect of not resetting σ between steps (σ collapse)
- Replacing the exact model with a learned one, then ensembles and pessimistic planning
