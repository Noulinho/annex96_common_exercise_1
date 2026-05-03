# RLMPC / Gnu-RL Cleanup Plan

This document defines the cleaned work that supports the notebooks in
`analysis/`. The notebooks are no longer the place where the experiment logic
lives; they are the final inspection and figure-making layer. The work done so
far is the reproducible support structure those notebooks need: shared configs,
scripts, source modules, result paths, and tests.

## Work Done So Far

We reorganized the exploratory notebook work into a small experiment pipeline:

- `analysis/02_mpc_results.ipynb` reads MPC results and produces clean plots.
- `analysis/04_rlmpc_results.ipynb` reads Gnu-RL/RLMPC rollouts, checkpoints,
  and summaries.
- `analysis/08_sac_results.ipynb` reads MERLIN-style SAC results and compares
  them with the MPC/RLMPC baselines.
- `configs/` defines the controller/scenario vocabulary used by all analysis
  notebooks.
- `scripts/` contains the command-line steps that create the files the analysis
  notebooks expect.
- `src/` contains reusable environment, controller, occupant, evaluation, and
  plotting code that was previously embedded in notebooks.
- `tests/` keeps the CityLearn/EV/series behavior checks separate from the
  analysis notebooks.
- `results/` is the intended generated output location for raw rollouts,
  summaries, fitted models, and checkpoints.

In short: the work is not only the notebooks. The notebooks under `analysis/`
are the readable output of the work; the configs, scripts, and source modules
are the reproducible experiment definition.

## Analysis Dependency Map

| Analysis notebook | What it needs |
|---|---|
| `analysis/02_mpc_results.ipynb` | `configs/scenarios.yaml`, `configs/controllers.yaml`, `configs/mpc.yaml`, `scripts/03_run_mpc.py`, `src/controllers/mpc.py`, `src/models/*`, `src/occupants/*`, `src/evaluation.py`, `src/plotting.py`, `results/raw/mpc/`, `results/summaries/mpc/` |
| `analysis/04_rlmpc_results.ipynb` | `configs/scenarios.yaml`, `configs/controllers.yaml`, `configs/rlmpc.yaml`, `scripts/04_train_rlmpc_offline.py`, `scripts/05_run_rlmpc_online.py`, `src/controllers/rlmpc.py`, `src/rlmpc_pipeline.py`, `src/occupants/tdyn.py`, `src/evaluation.py`, `src/plotting.py`, `results/models/rlmpc/`, `results/raw/rlmpc/`, `results/summaries/rlmpc/` |
| `analysis/08_sac_results.ipynb` | `configs/scenarios.yaml`, `configs/controllers.yaml`, `scripts/07_train_sac_merlin_january.py`, `scripts/08_run_sac_merlin_online.py`, `src/sac_merlin_pipeline.py`, `src/occupants/*`, `src/evaluation.py`, `src/plotting.py`, `results/models/sac_merlin/`, `results/raw/sac_merlin/`, `results/summaries/sac_merlin/` |

## Repository Roles

| Folder | Role |
|---|---|
| `configs/` | Stable experiment definitions: scenarios, controllers, building, MPC, and RLMPC settings. |
| `scripts/` | Reproducible execution entry points for fitting, training, running, and summarizing. |
| `src/` | Shared implementation code imported by scripts and notebooks. |
| `analysis/` | Clean result notebooks only; no long core implementation cells. |
| `results/` | Generated artifacts consumed by analysis notebooks. |
| `figures/` | Exported report/presentation figures. |
| `archive/` | Old scratch notebooks, backups, and exploratory artifacts. |

## Name

Use this consistently:

**Gnu-RL style RLMPC with Offline DiffMPC Imitation and Online Cost Optimization**

For occupant experiments:

**Occupant-Adaptive Gnu-RL RLMPC with persistent T_dyn comfort reference**

## Method Story

The controller is a differentiable MPC policy. Its RC-like dynamics parameters
are learned offline from January rollouts by imitation. After offline training,
the dynamics parameters are frozen. During February, online PPO-style updates
optimize only the MPC cost parameters:

- `q_track`
- `r_u`
- `sp_bias`

The occupant-specific version replaces the scheduled comfort reference with a
persistent `T_dyn` reference. `T_dyn` is initialized from the baseline schedule
or occupant preference and updated only when a new occupant thermostat override
starts.

## Notebook-To-Module Map

| Notebook concept | New home |
|---|---|
| `PaperDiffMPC` | `src/controllers/rlmpc.py` |
| `freeze_dynamics_parameters` | `src/controllers/rlmpc.py` |
| parameter snapshots | `src/controllers/rlmpc.py` |
| override-start detection | `src/occupants/tdyn.py` |
| persistent T_dyn construction | `src/occupants/tdyn.py` |
| online T_dyn state | `src/occupants/tdyn.py` |
| checkpoint/result paths | `src/rlmpc_pipeline.py` |
| offline training entry point | `scripts/04_train_rlmpc_offline.py` |
| online optimization entry point | `scripts/05_run_rlmpc_online.py` |
| clean analysis | `analysis/04_rlmpc_results.ipynb` |

## Experiment Variants

| Variant | Meaning |
|---|---|
| `feedback_only` | Paper-style T_dyn updated only from real new override feedback. |
| `present_only_no_bspline` | Occupant can intervene only when actually present; no occupancy prediction. |
| `bspline_dynamic_comfort` | Optional extension using B-spline dynamic comfort/occupancy model. |

## Commands

Current scaffold:

```bash
python scripts/04_train_rlmpc_offline.py --scenario occupant_present --occupant occ1_tolerant --tdyn-variant feedback_only
python scripts/05_run_rlmpc_online.py --scenario occupant_present --occupant occ1_tolerant --tdyn-variant feedback_only
```

Next migration step: move `train_diffmpc_from_bundle`,
`train_diffmpc_from_bundle_with_tdyn`, `StochasticDiffMPCPolicyWithNorms`,
`StochasticDiffMPCPolicyWithNormsTDyn`, and
`run_gnu_rl_case_from_checkpoint_tdyn` out of the notebook into `src/`.
