#!/usr/bin/env python
"""Replace the DiffMPC parity append with result-style comfort/load plots."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
PREFIX = "ce1_time_bias_guard_1c_2r2c_kalman_proxy_fair_p100k"
FIG_DIR = ROOT / "figures" / PREFIX
OLD_MARKER = "## CE1 2R2C+Kalman DiffMPC Action-Parity Gate"
MARKER = "## CE1 Exact-Proxy Fair 2R2C+Kalman MPC/RLMPC/SAC - Visual Result"


def _markdown_table(df: pd.DataFrame, digits: int = 3, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No data._"
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            data[column] = data[column].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
    rows = [list(data.columns)] + data.astype(str).values.tolist()
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))) + " |"

    return "\n".join(
        [fmt(rows[0]), "| " + " | ".join("-" * width for width in widths) + " |", *[fmt(row) for row in rows[1:]]]
    )


def _short_controller(name: str) -> str:
    mapping = {
        "CityLearn baseline": "CityLearn",
        "MPC 2R2C+Kalman exact proxy cost": "MPC",
        "RLMPC 2R2C+Kalman exact proxy reward": "RLMPC",
        "SAC safe-PI exact proxy reward (9 ep + bad +10)": "SAC hybrid",
    }
    return mapping.get(name, name)


def _load_tables() -> dict[str, pd.DataFrame]:
    mean = pd.read_csv(ANALYSIS / f"{PREFIX}_mean_building_kpis.csv")
    mean["controller"] = mean["controller"].map(_short_controller)
    comfort = mean.rename(
        columns={
            "electricity_consumption_total": "energy",
            "discomfort_%": "disc_%",
            "kelvin_hours": "KH",
        }
    )[["controller", "energy", "disc_%", "cold_%", "hot_%", "KH"]]
    operation = mean.rename(columns={"daily_peak_average": "daily_peak", "ramping_average": "ramping"})[
        ["controller", "daily_peak", "ramping", "feb_tin_max"]
    ].rename(columns={"feb_tin_max": "Tin_max"})

    district = pd.read_csv(ANALYSIS / f"{PREFIX}_district_kpis.csv")
    district["controller"] = district["controller"].map(_short_controller)
    load_shape = district.rename(
        columns={
            "total_vs_citylearn": "total_vs_CL",
            "peak_vs_citylearn": "peak_vs_CL",
            "daily_profile_peak_kwh": "daily_peak_kWh",
            "daily_profile_peak_hour": "daily_peak_hr",
        }
    )[["controller", "total_vs_CL", "peak_vs_CL", "daily_peak_kWh", "daily_peak_hr"]]
    tracking = district.rename(
        columns={
            "ramping_vs_citylearn": "ramp_vs_CL",
            "target_rmse_kwh": "target_RMSE",
            "above_target_kwh": "above_target",
        }
    )[["controller", "ramp_vs_CL", "target_RMSE", "above_target"]]

    terms = pd.read_csv(ANALYSIS / f"{PREFIX}_term_diagnostics.csv")
    terms["controller"] = terms["controller"].map(_short_controller)
    keep = ["tracking_target", "guard_cold", "guard_hot", "time_weighted_heat", "stage_cost", "guard_violation_%"]
    term_mean = (
        terms[terms["term"].isin(keep)]
        .pivot(index="controller", columns="term", values="mean")
        .reset_index()
        .rename(
            columns={
                "tracking_target": "track",
                "guard_cold": "cold_guard",
                "guard_hot": "hot_guard",
                "time_weighted_heat": "grid_heat",
                "stage_cost": "stage",
                "guard_violation_%": "guard_%",
            }
        )
    )
    term_max = (
        terms[terms["term"].isin(keep)]
        .pivot(index="controller", columns="term", values="max")
        .reset_index()
        .rename(
            columns={
                "tracking_target": "track",
                "guard_cold": "cold_guard",
                "guard_hot": "hot_guard",
                "time_weighted_heat": "grid_heat",
                "stage_cost": "stage",
                "guard_violation_%": "guard_%",
            }
        )
    )

    parity = pd.read_csv(ANALYSIS / "ce1_time_bias_guard_1c_2r2c_kalman_action_parity_gate_summary.csv")
    parity = parity.rename(
        columns={
            "action_mae_mean": "action_MAE",
            "action_rmse_mean": "action_RMSE",
            "action_max_abs_global": "max_abs",
            "qpth_warnings": "qpth_warn",
            "guard_penalty": "guard_pen",
        }
    )[["controller", "action_MAE", "action_RMSE", "max_abs", "qpth_warn", "guard_pen"]]

    return {
        "comfort": comfort,
        "operation": operation,
        "load_shape": load_shape,
        "tracking": tracking,
        "term_mean": term_mean,
        "term_max": term_max,
        "parity": parity,
    }


def _image(path: Path, title: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(f"![{title}](../{path.relative_to(ROOT).as_posix()})")


def _section_end(notebook: nbf.NotebookNode, start: int) -> int:
    end = start + 1
    while end < len(notebook.cells):
        source = str(notebook.cells[end].get("source", "")).strip()
        if source.startswith("## "):
            break
        if source.startswith("!["):
            end += 1
            continue
        break
    return end


def _append_once(path: Path, cells: list[nbf.NotebookNode]) -> None:
    notebook = nbf.read(path, as_version=4)
    for index, cell in enumerate(notebook.cells):
        source = str(cell.get("source", ""))
        if MARKER in source or OLD_MARKER in source:
            end = _section_end(notebook, index)
            notebook.cells[index:end] = cells
            nbf.write(notebook, path)
            print(f"Updated {MARKER} in {path.relative_to(ROOT)}")
            return
    notebook.cells.extend(cells)
    nbf.write(notebook, path)
    print(f"Appended {MARKER} to {path.relative_to(ROOT)}")


def main() -> None:
    tables = _load_tables()
    overview = f"""{MARKER}

Result-style view for the recovered 2R2C+Kalman exact-proxy experiment.

Fair comparison labels:

`ce1_time_bias_guard_1c_2r2c_kalman`: recovered MPC with January-only 2R2C+Kalman RC fit and `baseline_district_daily_mean_exp2_norm` grid stress.

`ce1_time_bias_guard_1c_2r2c_kalman_rlmpc_h24_west_exact_proxy_fair_p100k`: RLMPC/GNU-RL using the same 2R2C+Kalman model, horizon 24, Kalman/w_est observer, future setpoint/bias profiles, baseline-daily grid stress, action-proxy heat, and matched MPC soft guard constraints.

`ce1_time_bias_guard_1c_2r2c_kalman_sac_safepi9_proxy_fair_a1` plus `ce1_time_bias_guard_1c_2r2c_kalman_sac_safepi9_bad10_proxy_fair_a1`: SAC safe-PI bootstrap, same action-proxy reward/gridstress, actor range 0-1, with bad-building continuation.

Cost/reward:

`D_k = 3*((Tin-(Tsp+bias))/2)^2 + 60*(max(Tsp-2-Tin,0)/1.5)^2 + 15*(max(Tin-Tsp-2,0)/2)^2`

`J_k = 0.5*D_k + 0.5*GridStress*(p_RC2,b*u_k)/5`, `reward = -J_k`.

`p_RC2,b` is each building's January 2R2C+Kalman `power_per_unit_action`. `bias=+1 C` in preheat windows, `bias=-1 C` in reduce windows, and `GridStress = baseline_district_daily_mean_exp2_norm`.

MPC and RLMPC optimizers also include soft nonlinear guard constraints `Tin >= Tsp-2` and `Tin <= Tsp+2` with solver penalty `100000`. The SAC reward and the weighted-term diagnostics report the displayed `J_k` terms only.

Mean building comfort KPIs:

{_markdown_table(tables["comfort"], digits=3)}

Mean building operation KPIs:

{_markdown_table(tables["operation"], digits=3)}

District aggregate KPIs - load shape:

{_markdown_table(tables["load_shape"], digits=3)}

District aggregate KPIs - tracking:

{_markdown_table(tables["tracking"], digits=3)}

Weighted term diagnostics - mean:

{_markdown_table(tables["term_mean"], digits=3)}

Weighted term diagnostics - max:

{_markdown_table(tables["term_max"], digits=3)}

DiffMPC same-state action-parity sanity check:

{_markdown_table(tables["parity"], digits=3)}

Saved compact tables:

- `analysis/{PREFIX}_mean_building_kpis.csv`
- `analysis/{PREFIX}_district_kpis.csv`
- `analysis/{PREFIX}_term_diagnostics.csv`
- `analysis/ce1_time_bias_guard_1c_2r2c_kalman_action_parity_gate_summary.csv`
"""
    cells = [
        nbf.v4.new_markdown_cell(overview),
        _image(FIG_DIR / "mpc_comfort_all25.png", "MPC plot_comfort all buildings"),
        _image(FIG_DIR / "rlmpc_comfort_all25.png", "RLMPC plot_comfort all buildings"),
        _image(FIG_DIR / "sac_comfort_all25.png", "SAC hybrid plot_comfort all buildings"),
        _image(FIG_DIR / "district_load_timeseries.png", "District load time series"),
        _image(FIG_DIR / "district_mean_hourly_profile.png", "Daily average load profile"),
        _image(FIG_DIR / "weighted_terms_by_controller.png", "Weighted stage-cost terms"),
        _image(FIG_DIR / "hourly_bias_response.png", "Hourly response to time-bias schedule"),
        _image(FIG_DIR / "sac_safe_pi_january_episode0_comfort_all25.png", "SAC safe-PI January plot_comfort"),
        _image(FIG_DIR / "sac_online_training_reward.png", "SAC online reward"),
        _image(FIG_DIR / "sac_training_reset_fraction.png", "SAC reset fraction"),
    ]
    for notebook in [ANALYSIS / "02_mpc_results.ipynb", ANALYSIS / "04_rlmpc_results.ipynb"]:
        _append_once(notebook, cells)

    print("Updated result-style notebook section with comfort and daily load plots.")
    print(tables["comfort"].to_string(index=False))
    print(tables["load_shape"].to_string(index=False))


if __name__ == "__main__":
    main()
