from __future__ import annotations

from pathlib import Path

import nbformat as nbf
import pandas as pd


PREFIX = "ce1_occ_p2s2_all_occ"
ANALYSIS_DIR = Path("analysis")
MARKER = f"## CE1 Supplementary KPI Cost Audit - `{PREFIX}`"
NOTEBOOKS = [
    ANALYSIS_DIR / "02_mpc_results.ipynb",
    ANALYSIS_DIR / "04_rlmpc_results.ipynb",
    ANALYSIS_DIR / "08_sac_results.ipynb",
    ANALYSIS_DIR / "10_peak_load_profiles.ipynb",
    ANALYSIS_DIR / "11_no_occupant_single_building_comparison.ipynb",
    ANALYSIS_DIR / "12_simple_v4_comparison.ipynb",
]

SCORE_COLUMNS = [
    "ramping_average",
    "daily_one_minus_load_factor_average",
    "daily_peak_average",
    "all_time_peak_average",
]

DISPLAY_NAMES = {
    "controller": "ctrl",
    "occupant": "occ",
    "comfort_cost": "S_c",
    "grid_efficiency_cost": "S_g",
    "ramping_average": "S_ramp",
    "daily_one_minus_load_factor_average": "S_1-LF",
    "daily_peak_average": "S_daypk",
    "all_time_peak_average": "S_peak",
}

CSS = """
<style>
.kpi-scroll {
  max-height: 520px;
  overflow: auto;
  border: 1px solid #444;
  margin: 0.6rem 0 1.0rem 0;
}
.kpi-scroll table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.86rem;
}
.kpi-scroll th,
.kpi-scroll td {
  padding: 4px 7px;
  border: 1px solid #555;
  white-space: nowrap;
}
.kpi-scroll th {
  position: sticky;
  top: 0;
  background: #1f1f1f;
  z-index: 2;
}
</style>
"""


def _fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def html_table(df: pd.DataFrame, digits: int = 3) -> str:
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []

    for _, row in df.iterrows():
        rows.append("<tr>" + "".join(f"<td>{_fmt(v, digits)}</td>" for v in row) + "</tr>")

    return CSS + "<div class='kpi-scroll'><table><thead><tr>" + headers + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"


def compute_cost_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = ANALYSIS_DIR / f"{PREFIX}_mean_building_kpis.csv"
    means = pd.read_csv(source)

    required = {"month", "controller", "mode", "occupant", "discomfort_proportion", *SCORE_COLUMNS}
    missing = sorted(required - set(means.columns))

    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")

    costs = means.copy()
    costs["comfort_cost"] = costs["discomfort_proportion"]
    costs["grid_efficiency_cost"] = costs[SCORE_COLUMNS].mean(axis=1)

    cost_columns = [
        "month",
        "controller",
        "mode",
        "occupant",
        "comfort_cost",
        "grid_efficiency_cost",
        *SCORE_COLUMNS,
    ]
    costs = costs[cost_columns].copy()
    costs.to_csv(ANALYSIS_DIR / f"{PREFIX}_supplementary_comfort_grid_costs.csv", index=False)

    summary = (
        costs.groupby(["month", "controller", "mode"], as_index=False)[
            ["comfort_cost", "grid_efficiency_cost", *SCORE_COLUMNS]
        ]
        .mean()
        .sort_values(["month", "mode", "controller"])
    )
    summary.to_csv(ANALYSIS_DIR / f"{PREFIX}_supplementary_comfort_grid_costs_by_mode.csv", index=False)

    return summary, costs


def compact_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns=DISPLAY_NAMES)
    out["ctrl"] = out["ctrl"].replace(
        {
            "GNU-RL adaptive model": "GNU adapt",
            "GNU-RL fixed model": "GNU fixed",
            "Safe-PI/RBC": "Safe-PI",
        }
    )
    return out


def append_to_notebooks(markdown: str) -> None:
    new_cell = nbf.v4.new_markdown_cell(markdown)

    for notebook in NOTEBOOKS:
        nb = nbf.read(notebook, as_version=4)
        nb.cells = [
            cell
            for cell in nb.cells
            if not (cell.cell_type == "markdown" and str(cell.source).startswith(MARKER))
        ]
        nb.cells.append(new_cell)
        nbf.write(nb, notebook)


def build_markdown(summary: pd.DataFrame, costs: pd.DataFrame) -> str:
    summary_display = compact_display(summary)
    cost_display = compact_display(costs)

    return f"""{MARKER}

Supplementary KPI audit using the definitions from *Supplementary materials for "Optimizing building energy systems for grid-interactivity, comfort and resilience"* and the formulas implemented in `citylearn/cost_function.py` and `CityLearnEnv.evaluate()`.

Important distinction: the controller stage reward is still the normalized temperature-tracking cost used in the control experiments. The new `S_c` and `S_g` below are **evaluation costs**, not a new controller reward.

Formula/code match:

- `S_c` comfort cost maps to CityLearn `discomfort_proportion`, averaged over the 25 buildings. With occupant counts, CityLearn evaluates discomfort over occupied/evaluated steps.
- `S_g = 0.25*(S_ramp + S_1-LF + S_daypk + S_peak)`.
- `S_ramp` maps to CityLearn `ramping_average`. In this CityLearn code path, `CostFunction.ramping(..., down_ramp=None)` defaults to `down_ramp=False`, so it is cumulative positive **up-ramp** normalized by baseline.
- `S_1-LF` maps to `daily_one_minus_load_factor_average`.
- `S_daypk` maps to `daily_peak_average`.
- `S_peak` maps to `all_time_peak_average`.
- The raw `ramping_abs_kwh` column remains a separate auxiliary metric and is not the CityLearn `S_ramp` score.

Cost summary by mode:

{html_table(summary_display, digits=3)}

Full comfort/grid-efficiency cost table:

{html_table(cost_display, digits=3)}

Saved tables:

- `analysis/{PREFIX}_supplementary_comfort_grid_costs.csv`
- `analysis/{PREFIX}_supplementary_comfort_grid_costs_by_mode.csv`
"""


def main() -> None:
    summary, costs = compute_cost_tables()
    append_to_notebooks(build_markdown(summary, costs))
    print(f"Saved {ANALYSIS_DIR / f'{PREFIX}_supplementary_comfort_grid_costs.csv'}")
    print(f"Saved {ANALYSIS_DIR / f'{PREFIX}_supplementary_comfort_grid_costs_by_mode.csv'}")
    print(f"Appended {len(NOTEBOOKS)} notebooks")


if __name__ == "__main__":
    main()
