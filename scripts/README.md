# Benchmark entry points

Run commands from the repository root. The numbered scripts are reusable
pipeline stages; their defaults alone do not reconstruct every final thesis
setting. The final sweep wrapper supplies the publication objective flags.

| Script | Role |
|---|---|
| `00_collect_january_pi_data.py` | Collect January PI commissioning trajectories for fitting and imitation. |
| `00_collect_january_id_data.py` | Optional excitation/identification data collection. |
| `01_fit_rc_model.py` | Fit and export thermal models from commissioning trajectories. |
| `03_run_mpc.py` | Deploy MPC, including the thesis 2R2C/Kalman controller. |
| `04_train_rlmpc_offline.py` | Initialize differentiable MPC through January imitation. |
| `05_run_rlmpc_online.py` | Deploy GNU-RL with fixed or adaptive dynamics. |
| `06_summarize_results.py` | Summarize generated controller artifacts. |
| `07_train_sac_merlin_january.py` | January SAC training stage. |
| `08_run_sac_merlin_online.py` | SAC training/deployment and continuation. |
| `09_run_citylearn_baseline.py` | Dataset/CityLearn baseline evaluation. |
| `run_ce1_thesis_rbc_safepi.py` | Final Safe-PI comparator with the same reference logic. |
| `run_ce1_occupant_p2s2_all_occ.py` | Final publication sweep: no occupant and four occupant types, February/March. Includes SAC initial and continuation stages. |
| `run_ce1_occupant_p2s2_all_occ_sac_parallel.py` | Bounded per-building SAC execution using the final wrapper's command builders. |
| `build_results_notebooks.py` | Build the three reading notebooks from published CSV/PNG exports; no simulation. |

There is no stage 02: numbering follows the original workflow.

Inspect final controller commands without running them:

```bash
python scripts/run_ce1_occupant_p2s2_all_occ.py --dry-run
```

The research-specific `append_*`, ablation runners, and notebook repair scripts
are in [`experiments/scripts/`](../experiments/scripts/). The original final
publication exporters are there too; unlike `build_results_notebooks.py`, they
expect local raw trajectories and can modify historical notebooks and thesis
LaTeX files. See [the reproduction guide](../docs/reproduction.md).
