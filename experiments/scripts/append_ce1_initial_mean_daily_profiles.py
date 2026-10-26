#!/usr/bin/env python
"""Add mean-daily CE1 district profile plots to the early CE1 notebook sections."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = REPO / "analysis"
FIGURES_DIR = REPO / "figures"
PROFILE_CSV = REPO / "results" / "summaries" / "ce1_mean_daily_profiles_by_controller.csv"

TARGET_PLOT = FIGURES_DIR / "ce1_district_target_mean_daily_profiles_all_controllers.png"
PEAK_SHIFT_PLOT = FIGURES_DIR / "ce1_peak_shift_mean_daily_profiles_all_controllers.png"

NOTEBOOKS = [
    ANALYSIS_DIR / "02_mpc_results.ipynb",
    ANALYSIS_DIR / "04_rlmpc_results.ipynb",
    ANALYSIS_DIR / "08_sac_results.ipynb",
]


def _plot_profile(data: pd.DataFrame, cases: list[str], labels: dict[str, str], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    for case in cases:
        group = data.loc[data["case"] == case].sort_values("hour")
        if group.empty:
            continue
        color = "black" if case == "CE1 district target" else None
        linewidth = 2.2 if case == "CE1 district target" else 1.8
        ax.plot(
            group["hour"],
            group["load"],
            marker="o",
            linewidth=linewidth,
            color=color,
            label=labels.get(case, case),
        )

    ax.set_title(title)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean district electricity [kWh]")
    ax.set_xticks(range(24))
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_plots() -> None:
    data = pd.read_csv(PROFILE_CSV)
    _plot_profile(
        data,
        [
            "CityLearn baseline",
            "MPC ce1_district_target",
            "GNU-RL / RLMPC ce1_district_target",
            "SAC ce1_district_target",
            "CE1 district target",
        ],
        {
            "MPC ce1_district_target": "MPC",
            "GNU-RL / RLMPC ce1_district_target": "GNU-RL / RLMPC",
            "SAC ce1_district_target": "SAC",
        },
        TARGET_PLOT,
        "CE1 mean daily district profile after control (ce1_district_target)",
    )
    _plot_profile(
        data,
        [
            "CityLearn baseline",
            "MPC ce1_district_target",
            "MPC ce1_peak_shift",
            "GNU-RL / RLMPC ce1_district_target",
            "GNU-RL / RLMPC ce1_peak_shift",
            "SAC ce1_district_target",
            "SAC ce1_peak_shift",
            "CE1 district target",
        ],
        {
            "MPC ce1_district_target": "MPC target-only",
            "MPC ce1_peak_shift": "MPC peak+shift",
            "GNU-RL / RLMPC ce1_district_target": "GNU-RL / RLMPC target-only",
            "GNU-RL / RLMPC ce1_peak_shift": "GNU-RL / RLMPC peak+shift",
            "SAC ce1_district_target": "SAC target-only",
            "SAC ce1_peak_shift": "SAC peak+shift",
        },
        PEAK_SHIFT_PLOT,
        "CE1 mean daily district profile after control (target-only vs peak+shift)",
    )


def _strip_existing_cells(cells: list, marker: str) -> list:
    return [cell for cell in cells if marker not in str(cell.get("source", ""))]


def _insert_before_heading(cells: list, heading: str, new_cells: list) -> list:
    for idx, cell in enumerate(cells):
        if cell.cell_type == "markdown" and str(cell.source).lstrip().startswith(heading):
            return cells[:idx] + new_cells + cells[idx:]
    return cells + new_cells


def update_notebooks() -> None:
    target_rel = TARGET_PLOT.relative_to(REPO).as_posix()
    peak_rel = PEAK_SHIFT_PLOT.relative_to(REPO).as_posix()
    target_marker = "ce1_district_target_mean_daily_profiles_all_controllers.png"
    peak_marker = "ce1_peak_shift_mean_daily_profiles_all_controllers.png"

    for notebook_path in NOTEBOOKS:
        nb = nbf.read(notebook_path, as_version=4)
        cells = list(nb.cells)
        cells = _strip_existing_cells(cells, target_marker)
        cells = _strip_existing_cells(cells, peak_marker)

        target_cells = [
            nbf.v4.new_markdown_cell(
                "### Mean Daily District Profile\n\n"
                f"![CE1 district-target mean daily profile](../{target_rel})"
            )
        ]
        peak_cells = [
            nbf.v4.new_markdown_cell(
                "### Mean Daily District Profile\n\n"
                f"![CE1 peak-shift mean daily profile](../{peak_rel})"
            )
        ]

        cells = _insert_before_heading(cells, "## CE1 Target Comfort Plots", target_cells)
        if any(
            cell.cell_type == "markdown"
            and str(cell.source).lstrip().startswith("## CE1 Exact-Setpoint Normalized Shift Run")
            for cell in cells
        ):
            cells = _insert_before_heading(cells, "## CE1 Exact-Setpoint Normalized Shift Run", peak_cells)
        else:
            cells = _insert_before_heading(cells, "## CE1 Asymmetric Comfort + Peak/Shift Run", peak_cells)

        nb.cells = cells
        nbf.write(nb, notebook_path)


def main() -> None:
    make_plots()
    update_notebooks()
    print(f"Wrote {TARGET_PLOT.relative_to(REPO)}")
    print(f"Wrote {PEAK_SHIFT_PLOT.relative_to(REPO)}")
    print("Updated CE1 mean-daily plot cells in analysis notebooks.")


if __name__ == "__main__":
    main()
