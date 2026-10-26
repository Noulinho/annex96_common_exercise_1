#!/usr/bin/env python
"""Build the thesis reading notebooks from versioned result exports.

Run from any directory. No controller, training, raw rollout, or LaTeX export
is invoked. Use --execute to refresh the small saved table/image outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = '''from pathlib import Path
import pandas as pd
from IPython.display import Image, display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'analysis/artifact_manifest.json').is_file())
DATA = ROOT / 'analysis/data'
FIGURES = ROOT / 'analysis/figures'
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)
'''


def markdown(text):
    return nbf.v4.new_markdown_cell(text)


def table(filename, heading):
    return [markdown('## ' + heading), nbf.v4.new_code_cell(f"display(pd.read_csv(DATA / '{filename}'))")]


def figure(folder, filename, caption):
    return [markdown('## ' + caption), nbf.v4.new_code_cell(f"display(Image(filename=str(FIGURES / '{folder}/{filename}'))) ")]


def build(execute=False):
    prefix = 'ce1_thesis_final_results'
    final = 'ce1_thesis_final_results'
    clean = 'ce1_thesis_clean_load_figures'
    intro = '\n\nThese are the saved final thesis results for 25 Vermont homes, from `ce1_occ_p2s2_all_occ`. January supplies commissioning data; February and March are deployment months. Run all cells to read the versioned CSVs and figures. This notebook does not train or simulate controllers. See [reproduction and provenance](../docs/reproduction.md) and the [artifact manifest](artifact_manifest.json).'
    specs = []
    cells = [markdown('# Thesis results: controller comparison without occupants' + intro), markdown('MPC uses a fitted 2R2C model with Kalman correction. GNU-RL uses a differentiable 1R1C MPC policy with online adaptation. SAC receives January training and February continuation before March. Safe-PI/RBC is the controlled baseline; the CityLearn dataset baseline is a separate reference.\n\nTable VII averages normalized **building-level** scores after normalizing each building against its corresponding baseline. Lower scores are better; 1 means baseline for normalized load metrics. Comfort is a percentage of evaluated steps outside the band. KH measures reference-tracking Kelvin-hours and is not interchangeable with discomfort percentage. The Grid column averages ramping, 1-LF, daily peak, and maximum peak scores.'), nbf.v4.new_code_cell(SETUP)]
    for month in ['february', 'march']:
        cells += table(f'{prefix}_no_occupant_mean_building_kpis_{month}.csv', f'Table VII — {month.title()}')
    cells += [markdown('MPC has 1.10% and 1.17% total discomfort in February and March; GNU-RL has 3.94% and 3.75%. SAC improves from 27.09% to 5.70% after continuation. Safe-PI/RBC has 41.71% and 34.18%. Reduced electricity use must be interpreted alongside comfort: underheating can produce an apparently favorable energy score.')]
    cells += figure(final, 'thesis_no_occupant_selected_controllers_month.png', 'Representative buildings: February and March')
    cells += figure(final, 'thesis_no_occupant_selected_controllers_mean_daily.png', 'Representative buildings: daily profiles')
    cells += [markdown('The thesis fixes three illustrative buildings across comparisons: Building 12 (good), Building 15 (middle), and Building 8 (difficult). These are selected examples; the portfolio tables above cover all 25 homes.')]
    for month in ['february', 'march']:
        cells += table(f'{prefix}_no_occupant_selected_building_kpis_{month}.csv', f'Table VIII — selected buildings, {month.title()}')
        cells += figure(final, f'{month}_no_occupant_table_vii_histograms.png', f'Building-level distributions — {month.title()}')
    cells += table('ce1_occ_p2s2_all_occ_gnurl_adaptive_ppo_delta_summary.csv', 'GNU-RL online adaptation diagnostics')
    specs.append(('01_no_occupant_results.ipynb', cells))

    cells = [markdown('# Thesis results: occupant feedback and dynamic comfort' + intro), markdown('Four simulated occupant types (tolerant, sensitive, cold-preferring, and hot-preferring) generate stochastic thermostat overrides. `without_tdyn` uses the schedule and overrides; `bspline_tdyn` uses occupancy-gated, preference-anchored dynamic references.\n\nTable IX reports the B-spline dynamic-reference cases. Cold, hot, and comfort columns are auxiliary band-violation indicators. In these preference-shifted cases, KH and override starts are closer to the occupant feedback signal; a high band-violation percentage alone does not establish preference dissatisfaction. These are simulation models, not field-validated human-subject models.'), nbf.v4.new_code_cell(SETUP)]
    for month in ['february', 'march']:
        cells += table(f'{prefix}_occupant_controller_mean_kpis_{month}.csv', f'Table IX — {month.title()}')
    cells += table('ce1_occ_p2s2_all_occ_override_deltas.csv', 'Paired override comparisons for every building')
    cells += [nbf.v4.new_code_cell('''overrides = pd.read_csv(DATA / 'ce1_occ_p2s2_all_occ_override_deltas.csv')
summary = overrides.groupby(['month', 'controller'])[['overrides_without', 'overrides_bspline']].sum()
summary['reduction_pct_from_totals'] = 100 * (1 - summary['overrides_bspline'] / summary['overrides_without'])
display(summary)
'''), markdown('The calculation above uses pooled override counts per controller and month. Its averaging convention is explicit; it can differ from an average of individual percentage reductions. The thesis reports approximately 78% reduction for MPC and 72% for GNU-RL, versus 14% for Safe-PI/RBC.')]
    for filename, caption in [('thesis_mpc_occupant_selected_month.png','MPC occupant response in the same representative buildings'), ('thesis_mpc_occupant_selected_mean_daily.png','MPC daily comfort profiles by occupant'), ('thesis_mpc_occupant_mean_hourly_load_feb_mar.png','MPC occupant load profiles'), ('feb_mar_mpc_bspline_tdyn_table_viii_hist_by_occupant.png','MPC building-level distributions by occupant')]:
        cells += figure(final, filename, caption)
    cells += [markdown('The distribution figure retains its original export filename (`table_viii`); the final PDF labels the occupant comparison **Table IX**. For occupancy prediction validation, see [the dedicated validation notebook](09_bspline_occupancy_validation.ipynb), which distinguishes full-series fitting from January-only holdout fitting.')]
    specs.append(('02_occupant_results.ipynb', cells))

    cells = [markdown('# Thesis results: load shifting and deployment profiles' + intro), markdown('These figures show how the final controllers redistribute heating over the day. The stage objective uses normalized temperature tracking and asymmetric guards, with weights 0.1 / 0.7 / 0.2 and 2 K scales. A time-dependent reference bias induces load shifting. Evaluation grid and comfort costs are reported separately from this training/control objective.'), nbf.v4.new_code_cell(SETUP)]
    for filename, caption in [('time_bias_windows_vs_baseline.png','Reference-bias windows and baseline demand'), ('figure_A_no_occupant_feb_mar_combined.png','Controller daily load profiles: February and March'), ('no_occupant_deployment_daily_kpi_normalized.png','Deployment KPI trajectories'), ('figure_B_mpc_feb_occupant_impact.png','MPC occupant impact: February'), ('figure_B_mpc_mar_occupant_impact.png','MPC occupant impact: March')]:
        cells += figure(clean, filename, caption)
    cells += table('ce1_occ_p2s2_all_occ_supplementary_comfort_grid_costs_by_mode.csv', 'Supplementary comfort and grid evaluation audit')
    specs.append(('03_load_shifting_results.ipynb', cells))
    for filename, cells in specs:
        notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name':'Python 3', 'language':'python', 'name':'python3'}})
        if execute:
            from nbclient import NotebookClient
            NotebookClient(notebook, timeout=120, resources={'metadata': {'path':str(ROOT)}}).execute()
        path = ROOT / 'analysis' / filename
        nbf.write(notebook, path)
        print(path.relative_to(ROOT))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Refresh saved outputs using only committed artifacts.')
    build(parser.parse_args().execute)
