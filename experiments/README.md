# Research experiments and development history

Browse the [complete experiment inventory](INDEX.md) to locate individual trials.

This directory preserves the exploratory work behind the thesis, including
saved notebook outputs. Start with [the final results](../analysis/README.md)
for the completed comparison.

| Location | Contents |
|---|---|
| `analysis/` | Original large controller notebooks: MPC, GNU-RL, SAC, district profiles, and intermediate comparisons. Each contains many successive reward/controller experiments. |
| `notebooks/` | Deployment prototypes, scratch MPC, quickstart modifications, and occupant trials. |
| `scripts/` | Reward ablations, trial runners, append-to-notebook tools, diagnostic utilities, and original thesis publication exporters. |
| [moved_files.json](moved_files.json) | Exact old → new path mapping for all relocated tracked files. |

The final publication run is `ce1_occ_p2s2_all_occ`. Earlier names containing
`final`, `best`, `normD`, `D010702`, `p3`, `fair3`, or reward variants describe
research milestones, not the final thesis selection.

Useful original exporters:

- `scripts/append_ce1_occupant_p2s2_all_occ_analysis.py`: aggregate the final sweep.
- `scripts/append_ce1_thesis_final_results.py`: compact thesis tables and figures.
- `scripts/append_ce1_thesis_clean_load_figures.py`: publication load profiles.
- `scripts/append_ce1_supplementary_kpi_cost_audit.py`: evaluation-cost audit.

Run historical scripts as modules from the repository root, for example
`python -m experiments.scripts.append_ce1_thesis_final_results`. These tools
require original local `results/` artifacts and some also write to the author's
LaTeX working directory. They are retained for provenance; use
`python scripts/build_results_notebooks.py --execute` for the portable reading
notebooks.

Historical notebooks are preserved research records, not a maintained clean
reproduction pipeline. To inspect their code, start Jupyter from the repository
root. File paths in saved text or historical outputs may describe their old
locations; consult the relocation map. Unpublished raw rollouts, model weights,
and replay buffers are not bundled.
