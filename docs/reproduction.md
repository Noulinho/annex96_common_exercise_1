# Reproduction and result provenance

## Inspect the published results

The three reading notebooks under `analysis/` run entirely from versioned CSV
and PNG exports. They need no `results/` directory, trained weights, replay
buffers, simulator, or thesis LaTeX checkout.

```bash
python -m pip install -r requirements-analysis.txt
python scripts/build_results_notebooks.py --execute
```

This refreshes tables and embeds the existing figures in notebook outputs. It
does not regenerate the figures from raw trajectories or rerun the experiments.
The original full simulation environment is separate:

```bash
python -m pip install -r requirements.txt
python -m pip install scipy matplotlib nbformat tabulate cvxpy
```

The original requirements include upstream simulation tools and are not a
locked environment specification. Use the repository's local `citylearn` and
`diff_mpc` modules; do not replace them with independently installed versions.
No fresh-environment, bit-for-bit training reproduction is claimed by this
presentation cleanup.

## Final experiment identity

The supplied final thesis PDF was used to identify the reported results,
specifically Tables VII–IX and the February/March comfort figures. The PDF is
not redistributed in this repository.

| Item | Source |
|---|---|
| Final controller sweep | `scripts/run_ce1_occupant_p2s2_all_occ.py` |
| Run prefix | `ce1_occ_p2s2_all_occ` |
| Safe-PI comparator | `scripts/run_ce1_thesis_rbc_safepi.py` / `ce1_thesis_rbc_safepi` |
| Final compact export | `experiments/scripts/append_ce1_thesis_final_results.py` / `ce1_thesis_final_results` |
| Load-profile export | `experiments/scripts/append_ce1_thesis_clean_load_figures.py` |
| Original combined analysis | `experiments/analysis/12_simple_v4_comparison.ipynb`, section “CE1 Thesis Final Compact Results” |
| Dataset | `data/datasets/annex96_ce1_vt_neighborhood/` |
| Published artifact list and checksums | [`analysis/artifact_manifest.json`](../analysis/artifact_manifest.json) |

Final compact results were appended into six research notebooks, so filenames
such as `02_mpc_results` or `08_sac_results` do not imply exclusive ownership of
the thesis figures. The new reading notebooks present one consistent final set
and keep the complete research history in `experiments/`.

The curated CSVs and PNGs are byte-for-byte copies of the existing final
exports. Original CSV files and generated figures may still exist locally in
`analysis/` and `figures/`; those working outputs are ignored. To publish new
scientific results, regenerate the original exports, review the changed data,
copy the selected files into `analysis/data/` and `analysis/figures/`, update the
manifest checksums, then rebuild the notebooks.

## Simulation workflow

Run all simulation commands from the repository root. Use each script's
`--help` to inspect paths and settings before long training jobs.

1. **Commissioning:** `00_collect_january_pi_data.py` collects January PI
   trajectories. `00_collect_january_id_data.py` provides an optional
   identification/excitation variant. Outputs go into `results/datasets/`.
2. **Thermal models:** `01_fit_rc_model.py` supports `--model-type 1r1c` and
   `--model-type 2r2c`. The final MPC uses 2R2C and Kalman state correction;
   GNU-RL uses 1R1C. Model outputs go into `results/models/rc/`.
3. **GNU-RL initialization:** `04_train_rlmpc_offline.py` uses January data for
   imitation and saves per-building checkpoints and normalization bundles under
   `results/models/rlmpc/`. Occupant scenarios require their corresponding
   initialization paths; inspect the resolved commands and scenario slugs.
4. **Baseline:** `09_run_citylearn_baseline.py` evaluates the dataset baseline.
   `run_ce1_thesis_rbc_safepi.py` runs the separate final Safe-PI comparator.
5. **Deployment:** `run_ce1_occupant_p2s2_all_occ.py` builds the publication
   commands for MPC, fixed/adaptive GNU-RL, and SAC. The standalone
   `07_train_sac_merlin_january.py` is an additional training entry point; the
   final wrapper uses `08_run_sac_merlin_online.py` for its SAC stages.
6. **Export:** the historical final aggregation and publication scripts in
   `experiments/scripts/` consume saved raw trajectories. They can also append
   to historical notebooks or edit local LaTeX files. They are not needed to
   view the published results.

Preview the final sweep (creates only its log directory):

```bash
python scripts/run_ce1_occupant_p2s2_all_occ.py --dry-run
```

After preparing the commissioning models and initialization checkpoints,
run one final controller family without occupants:

```bash
python scripts/run_ce1_occupant_p2s2_all_occ.py \
  --controllers mpc --modes no_occupant --months feb mar
```

The wrapper uses your current Python interpreter. Its defaults sweep all 25
buildings, four occupant types, both occupant modes, and both deployment months.
It does **not** create missing RC fits or offline GNU-RL checkpoints. SAC
continuation expects the January-trained agent; March adaptive GNU-RL
continuation expects the February updated checkpoint. Keep stage order intact.
The optional parallel SAC wrapper uses the same command builders and labels.

## Metrics and interpretation

The final objective flags use `time_bias_guard`, tracking/cold/hot weights
0.1/0.7/0.2, 2 K scales, and a 1 K reference bias. Grid efficiency scores are
post hoc evaluation metrics, not an alternative training reward.

Load scores are normalized against the corresponding baseline **per building**
and then averaged. The Grid score is the mean of normalized ramping,
one-minus-load-factor, daily peak, and maximum peak. CityLearn ramping here
measures positive up-ramps. The baseline score of 1 applies to normalized load
metrics, not to raw comfort percentages or Kelvin-hours.

In the occupant-present comparison, band violations are auxiliary indicators;
preference shifts can raise them even while override counts fall. Report KH
and override events alongside these indicators. The occupancy validation
notebook distinguishes controller full-series fitting from January-only holdout
fitting; those fit scopes should not be conflated.

The result exports are included; raw training trajectories, model weights, and
replay buffers in `results/` remain local and are not downloadable from this
repository. The original requirements and training randomness also limit exact
reproduction of every reported number from a fresh clone.
