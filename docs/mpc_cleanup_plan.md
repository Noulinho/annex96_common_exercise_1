# MPC Cleanup Plan

This project now follows the same broad pattern as `merlin-apen-2023`:

- `src/` contains reusable experiment code.
- `configs/` contains scenario and controller definitions.
- `scripts/` contains reproducible command-line entry points.
- `analysis/` contains notebooks for results, not core implementations.
- `results/` stores generated rollouts, summaries, and fitted models.
- `figures/` stores publication/report figures.
- `archive/` is for old scratch notebooks and run artifacts.

## Controller/Scenario Matrix

| Controller | no_occupant | occupant_present | occupant_present_peak_flattening |
|---|---:|---:|---:|
| RBC | planned | planned | optional |
| MPC | in progress | in progress | planned |
| RL | in progress | in progress | planned |
| RLMPC / Gnu-RL | in progress | in progress | planned |

## MPC Story

Use this name consistently:

**Occupant-Adaptive RC-MPC with B-spline Dynamic Comfort Preferences**

Pipeline:

1. Select building and train/test period.
2. Collect a January reference trajectory.
3. Fit a 1R1C thermal model from the reference trajectory.
4. Fit occupant dynamic preference / override behavior.
5. Run MPC under each scenario.
6. Save raw rollouts and summary metrics.
7. Use analysis notebooks for figures and comparisons.

## Migration Map From `notebooks/my_mpc.ipynb`

| Notebook concept | New home |
|---|---|
| `make_env` | `src/envs.py` |
| `run_env`, `collect_trajectories` | `src/data/collect.py` |
| 1R1C rollout and fit | `src/models/rc_1r1c.py`, `src/models/fit_rc.py` |
| 2R2C exploration | `src/models/rc_2r2c.py` |
| `Occupant` | `src/occupants/occupant.py` |
| B-spline occupancy/prefs | `src/occupants/bspline_preferences.py` |
| occupant wrappers | `src/occupants/wrappers.py` |
| `OccupantFeedbackReward` | `src/occupants/comfort.py` |
| `LearnedRCMPCPolicy` / do-mpc loop | `src/controllers/mpc.py` |
| metrics | `src/evaluation.py` |
| plots | `src/plotting.py` |

## Current MPC Status

`src/controllers/mpc.py` now contains a real `OccupantAdaptiveRCMPC.predict()`
implementation. The notebook globals have been turned into explicit inputs:

- fitted RC parameters from `RC1R1CModel`,
- observation indices from `MPCObservationIndices`,
- current timestep and price forecasts from the bound CityLearn environment,
- comfort target from either the scheduled setpoint or a provided preference model.

Next, wire `scripts/03_run_mpc.py` to build the environment, load the fitted RC
model JSON, instantiate `OccupantAdaptiveRCMPC`, collect the rollout, and save
`results/raw/mpc/<scenario>/rollout.parquet`.
