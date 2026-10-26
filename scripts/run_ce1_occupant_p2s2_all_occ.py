#!/usr/bin/env python
"""Run final CE1 p2 occupant-in-loop all-occupant controller experiments.

This is the thesis-final occupant sweep:

- no occupant on all 25 buildings;
- each fitted occupant type on all 25 buildings;
- raw occupant overrides and B-spline Tdyn modes;
- p2 normalized-D objective: weights 0.1/0.7/0.2, cold/hot scales both 2.0;
- no RC refit/offline imitation retraining.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PYTHON = Path(sys.executable)
PREFIX = "ce1_occ_p2s2_all_occ"
LOG_DIR = Path("logs") / PREFIX

MONTHS = {
    "feb": (744, 1415),
    "mar": (1416, 2159),
}

BUILDINGS_ALL = ",".join(str(i) for i in range(1, 26))
OCCUPANTS = ["occ1_tolerant", "occ2_sensitive", "occ3_cold", "occ4_hot"]
MODES = ["without_tdyn", "bspline_tdyn"]

COMMON_COMFORT_FLAGS = [
    "--comfort-mode", "time_bias_guard",
    "--comfort-weight", "1.0",
    "--energy-weight", "0.0",
    "--peak-weight", "0.0",
    "--ramp-weight", "0.0",
    "--shift-weight", "0.0",
    "--preheat-weight", "0.0",
    "--comfort-scale", "1.0",
    "--comfort-track-weight", "0.1",
    "--comfort-cold-weight", "0.7",
    "--comfort-hot-weight", "0.2",
    "--comfort-track-scale", "2.0",
    "--comfort-cold-scale", "2.0",
    "--comfort-hot-scale", "2.0",
    "--time-bias-amplitude", "1.0",
    "--time-bias-guard-band", "2.0",
    "--power-source", "heating",
    "--single-setpoint",
]

MPC_RLMPC_MATCHED_FLAGS = [
    "--time-bias-guard-penalty", "0.0",
]

SAC_TRAINING_FLAGS = [
    "--action-scaling-coefficient", "1.0",
    "--jumpstart-controller", "safe_pi",
    "--safe-pi-random-prob", "0.0",
    "--safe-pi-random-low", "0.0",
    "--automatic-entropy-tuning",
    "--entropy-lr", "0.001",
    "--target-entropy", "-1.0",
    "--comfort-guard-exponent", "2.0",
    "--robust-normalization",
    "--normalized-observation-clip", "5.0",
    "--normalized-reward-clip", "10.0",
]


def run(cmd: list[str], log_name: str, *, dry_run: bool = False) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{log_name}.log"
    print(f"\n>>> {log_name}")
    print(" ".join(cmd))
    print(f"log: {log_path}")
    if dry_run:
        return
    with log_path.open("w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(f"{log_name} failed with code {completed.returncode}\n\n{tail}")


def mpc_no_occupant_cmd(month: str) -> list[str]:
    start, end = MONTHS[month]
    return [
        str(PYTHON), "-u", "scripts/03_run_mpc.py",
        "--scenario", "no_occupant",
        "--building", BUILDINGS_ALL,
        "--label", f"{PREFIX}_mpc_{month}_no_occupant",
        "--objective", "balanced",
        "--rc-model-type", "2r2c",
        "--two-state-observer", "kalman",
        "--simulation-start", str(start),
        "--simulation-end", str(end),
        *COMMON_COMFORT_FLAGS,
        *MPC_RLMPC_MATCHED_FLAGS,
    ]


def mpc_occupant_cmd(month: str, occupant: str, mode: str | None = None) -> list[str]:
    start, end = MONTHS[month]
    occupant_mode = "both" if mode is None else mode
    label = f"{PREFIX}_mpc_{month}" if mode is None else f"{PREFIX}_mpc_{month}_{occupant}_{mode}"
    return [
        str(PYTHON), "-u", "scripts/03_run_mpc.py",
        "--scenario", "occupant_present",
        "--building", BUILDINGS_ALL,
        "--occupant", occupant,
        "--occupant-mode", occupant_mode,
        "--label", label,
        "--objective", "balanced",
        "--rc-model-type", "2r2c",
        "--two-state-observer", "kalman",
        "--simulation-start", str(start),
        "--simulation-end", str(end),
        *COMMON_COMFORT_FLAGS,
        *MPC_RLMPC_MATCHED_FLAGS,
    ]


def rlmpc_no_occupant_cmd(
    month: str,
    *,
    device: str,
    adaptive_dynamics: bool = False,
    continue_from_feb: bool = False,
) -> list[str]:
    start, end = MONTHS[month]
    kind = "adaptive" if adaptive_dynamics else "fixed"
    cmd = [
        str(PYTHON), "-u", "scripts/05_run_rlmpc_online.py",
        "--scenario", "no_occupant",
        "--building", BUILDINGS_ALL,
        "--label", f"{PREFIX}_rlmpc_{kind}_{month}_no_occupant",
        "--reward", "balanced",
        "--rc-model-type", "1r1c",
        "--simulation-start", str(start),
        "--simulation-end", str(end),
        "--device", device,
        *COMMON_COMFORT_FLAGS,
        *MPC_RLMPC_MATCHED_FLAGS,
    ]
    if continue_from_feb:
        cmd.extend(
            [
                "--checkpoint-template",
                (
                    "results/raw/rlmpc/{building_id}/no_occupant/"
                    f"{PREFIX}_rlmpc_adaptive_feb_no_occupant/online_updated_checkpoint.pt"
                ),
            ]
        )
    if adaptive_dynamics:
        cmd.append("--adaptive-dynamics")
    return cmd


def rlmpc_occupant_cmd(
    month: str,
    occupant: str,
    mode: str | None,
    *,
    device: str,
    adaptive_dynamics: bool = False,
    continue_from_feb: bool = False,
) -> list[str]:
    start, end = MONTHS[month]
    kind = "adaptive" if adaptive_dynamics else "fixed"
    occupant_mode = "both" if mode is None else mode
    if mode is None:
        label = f"{PREFIX}_rlmpc_{kind}_{month}"
    elif adaptive_dynamics and month == "mar" and continue_from_feb:
        # Preserve the existing March-adaptive artifact shape: the lower-level
        # runner appends occupant/mode to this label after loading the matching
        # February checkpoint.
        label = f"{PREFIX}_rlmpc_{kind}_{month}_{occupant}_{mode}"
    else:
        label = f"{PREFIX}_rlmpc_{kind}_{month}"
    cmd = [
        str(PYTHON), "-u", "scripts/05_run_rlmpc_online.py",
        "--scenario", "occupant_present",
        "--building", BUILDINGS_ALL,
        "--occupant", occupant,
        "--occupant-mode", occupant_mode,
        "--label", label,
        "--reward", "balanced",
        "--rc-model-type", "1r1c",
        "--simulation-start", str(start),
        "--simulation-end", str(end),
        "--device", device,
        *COMMON_COMFORT_FLAGS,
        *MPC_RLMPC_MATCHED_FLAGS,
    ]
    if continue_from_feb:
        if mode is None:
            raise ValueError("Adaptive March continuation needs one occupant mode at a time.")
        cmd.extend(
            [
                "--checkpoint-template",
                (
                    "results/raw/rlmpc/{building_id}/occupant_present/"
                    f"{PREFIX}_rlmpc_adaptive_feb_{occupant}_{mode}/online_updated_checkpoint.pt"
                ),
            ]
        )
    if adaptive_dynamics:
        cmd.append("--adaptive-dynamics")
    return cmd


def sac_no_occupant_initial_cmd(*, device: str) -> list[str]:
    return [
        str(PYTHON), "-u", "scripts/08_run_sac_merlin_online.py",
        "--scenario", "no_occupant",
        "--building", BUILDINGS_ALL,
        "--device", device,
        "--label", f"{PREFIX}_sac_jan20_feb_no_occupant",
        "--reward", "balanced",
        "--train-sim-start", "0",
        "--train-sim-end", "743",
        "--eval-sim-start", "744",
        "--eval-sim-end", "1415",
        "--training-episodes", "20",
        *COMMON_COMFORT_FLAGS,
        *SAC_TRAINING_FLAGS,
    ]


def sac_occupant_initial_cmd(occupant: str, mode: str, *, device: str) -> list[str]:
    return [
        str(PYTHON), "-u", "scripts/08_run_sac_merlin_online.py",
        "--scenario", "occupant_present",
        "--building", BUILDINGS_ALL,
        "--occupant", occupant,
        "--occupant-mode", mode,
        "--device", device,
        "--label", f"{PREFIX}_sac_jan20_feb_{occupant}_{mode}",
        "--reward", "balanced",
        "--train-sim-start", "0",
        "--train-sim-end", "743",
        "--eval-sim-start", "744",
        "--eval-sim-end", "1415",
        "--training-episodes", "20",
        *COMMON_COMFORT_FLAGS,
        *SAC_TRAINING_FLAGS,
    ]


def sac_no_occupant_continuation_cmd(*, device: str) -> list[str]:
    resume = f"results/models/sac_merlin/{{building_id}}/{PREFIX}_sac_jan20_feb_no_occupant_jan_feb_online_agent.pkl"
    return [
        str(PYTHON), "-u", "scripts/08_run_sac_merlin_online.py",
        "--scenario", "no_occupant",
        "--building", BUILDINGS_ALL,
        "--device", device,
        "--label", f"{PREFIX}_sac_feb10_mar_no_occupant",
        "--reward", "balanced",
        "--train-sim-start", "744",
        "--train-sim-end", "1415",
        "--eval-sim-start", "1416",
        "--eval-sim-end", "2159",
        "--training-episodes", "10",
        "--resume-agent-template", resume,
        "--resume-episode-offset", "20",
        *COMMON_COMFORT_FLAGS,
        *SAC_TRAINING_FLAGS,
    ]


def sac_occupant_continuation_cmd(occupant: str, mode: str, *, device: str) -> list[str]:
    resume = (
        "results/models/sac_merlin/{building_id}/"
        f"{PREFIX}_sac_jan20_feb_{occupant}_{mode}_jan_feb_online_agent.pkl"
    )
    cmd = [
        str(PYTHON), "-u", "scripts/08_run_sac_merlin_online.py",
        "--scenario", "occupant_present",
        "--building", BUILDINGS_ALL,
        "--occupant", occupant,
        "--occupant-mode", mode,
        "--device", device,
        "--label", f"{PREFIX}_sac_feb10_mar_{occupant}_{mode}",
        "--reward", "balanced",
        "--train-sim-start", "744",
        "--train-sim-end", "1415",
        "--eval-sim-start", "1416",
        "--eval-sim-end", "2159",
        "--training-episodes", "10",
        "--resume-agent-template", resume,
        "--resume-episode-offset", "20",
        *COMMON_COMFORT_FLAGS,
        *SAC_TRAINING_FLAGS,
    ]
    if mode == "bspline_tdyn":
        cmd.extend(["--resume-replay-keep-last", "1024"])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["mpc", "rlmpc_fixed", "rlmpc_adaptive", "sac_initial", "sac_continuation"],
        choices=["mpc", "rlmpc_fixed", "rlmpc_adaptive", "sac_initial", "sac_continuation"],
    )
    parser.add_argument("--months", nargs="+", default=["feb", "mar"], choices=["feb", "mar"])
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["no_occupant", "without_tdyn", "bspline_tdyn"],
        choices=["no_occupant", "without_tdyn", "bspline_tdyn"],
    )
    parser.add_argument("--occupants", nargs="+", default=OCCUPANTS, choices=OCCUPANTS)
    parser.add_argument("--rlmpc-device", default="cpu")
    parser.add_argument("--sac-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for controller in args.controllers:
        if controller == "mpc":
            for month in args.months:
                if "no_occupant" in args.modes:
                    run(mpc_no_occupant_cmd(month), f"mpc_{month}_no_occupant", dry_run=args.dry_run)
                for occupant in args.occupants:
                    occupant_modes = [m for m in args.modes if m != "no_occupant"]
                    if occupant_modes:
                        if set(occupant_modes) == set(MODES):
                            run(mpc_occupant_cmd(month, occupant), f"mpc_{month}_{occupant}", dry_run=args.dry_run)
                        else:
                            for mode in occupant_modes:
                                run(
                                    mpc_occupant_cmd(month, occupant, mode),
                                    f"mpc_{month}_{occupant}_{mode}",
                                    dry_run=args.dry_run,
                                )
        elif controller in {"rlmpc_fixed", "rlmpc_adaptive"}:
            for month in args.months:
                adaptive_dynamics = controller == "rlmpc_adaptive"
                if "no_occupant" in args.modes:
                    run(
                        rlmpc_no_occupant_cmd(
                            month,
                            device=args.rlmpc_device,
                            adaptive_dynamics=adaptive_dynamics,
                            continue_from_feb=adaptive_dynamics and month == "mar",
                        ),
                        f"{controller}_{month}_no_occupant",
                        dry_run=args.dry_run,
                    )
                for occupant in args.occupants:
                    if controller == "rlmpc_adaptive" and month == "mar":
                        for mode in [m for m in args.modes if m != "no_occupant"]:
                            run(
                                rlmpc_occupant_cmd(
                                    month,
                                    occupant,
                                    mode,
                                    device=args.rlmpc_device,
                                    adaptive_dynamics=True,
                                    continue_from_feb=True,
                                ),
                                f"{controller}_{month}_{occupant}_{mode}",
                                dry_run=args.dry_run,
                            )
                    else:
                        occupant_modes = [m for m in args.modes if m != "no_occupant"]
                        if occupant_modes:
                            if set(occupant_modes) == set(MODES):
                                run(
                                    rlmpc_occupant_cmd(
                                        month,
                                        occupant,
                                        None,
                                        device=args.rlmpc_device,
                                        adaptive_dynamics=adaptive_dynamics,
                                    ),
                                    f"{controller}_{month}_{occupant}",
                                    dry_run=args.dry_run,
                                )
                            else:
                                for mode in occupant_modes:
                                    run(
                                        rlmpc_occupant_cmd(
                                            month,
                                            occupant,
                                            mode,
                                            device=args.rlmpc_device,
                                            adaptive_dynamics=adaptive_dynamics,
                                        ),
                                        f"{controller}_{month}_{occupant}_{mode}",
                                        dry_run=args.dry_run,
                                    )
        elif controller == "sac_initial":
            if "no_occupant" in args.modes:
                run(sac_no_occupant_initial_cmd(device=args.sac_device), "sac_jan20_feb_no_occupant", dry_run=args.dry_run)
            for occupant in args.occupants:
                for mode in [m for m in args.modes if m != "no_occupant"]:
                    run(
                        sac_occupant_initial_cmd(occupant, mode, device=args.sac_device),
                        f"sac_jan20_feb_{mode}_{occupant}",
                        dry_run=args.dry_run,
                    )
        elif controller == "sac_continuation":
            if "no_occupant" in args.modes:
                run(
                    sac_no_occupant_continuation_cmd(device=args.sac_device),
                    "sac_feb10_mar_no_occupant",
                    dry_run=args.dry_run,
                )
            for occupant in args.occupants:
                for mode in [m for m in args.modes if m != "no_occupant"]:
                    run(
                        sac_occupant_continuation_cmd(occupant, mode, device=args.sac_device),
                        f"sac_feb10_mar_{mode}_{occupant}",
                        dry_run=args.dry_run,
                    )


if __name__ == "__main__":
    sys.exit(main())
