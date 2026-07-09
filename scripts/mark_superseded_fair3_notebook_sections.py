from pathlib import Path

import nbformat


NOTEBOOKS = [
    Path("analysis/02_mpc_results.ipynb"),
    Path("analysis/04_rlmpc_results.ipynb"),
    Path("analysis/08_sac_results.ipynb"),
    Path("analysis/10_peak_load_profiles.ipynb"),
    Path("analysis/11_no_occupant_single_building_comparison.ipynb"),
    Path("analysis/12_simple_v4_comparison.ipynb"),
]

OLD_HEADING = "## CE1 Fair 2R2C+Kalman MPC/RLMPC/SAC - `ce1_time_bias_guard_1c_2r2c_kalman_fair3`"
WARNING = (
    "> **Superseded sanity note:** this `fair3` block is an older comparison and is **not** "
    "the current exact-cost fair comparison. Use the later "
    "`ce1_time_bias_guard_1c_2r2c_kalman_proxy_fair_p100k` section instead; that is the one "
    "matched to the recovered MPC setup with 2R2C+Kalman, horizon 24, "
    "`baseline_district_daily_mean_exp2_norm`, action-proxy heat, and the MPC/RLMPC "
    "soft guard constraint penalty.\n"
)


def mark_notebook(path: Path) -> bool:
    nb = nbformat.read(path, as_version=4)
    changed = False
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        if OLD_HEADING not in cell.source:
            continue
        if WARNING in cell.source:
            continue
        cell.source = cell.source.replace(OLD_HEADING, OLD_HEADING + "\n\n" + WARNING, 1)
        changed = True
    if changed:
        nbformat.write(nb, path)
    return changed


def main() -> None:
    for path in NOTEBOOKS:
        if mark_notebook(path):
            print(f"Marked superseded fair3 section in {path}")


if __name__ == "__main__":
    main()
