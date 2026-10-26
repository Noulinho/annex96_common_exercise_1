#!/usr/bin/env python
"""Run final CE1 occupant-in-loop D010702 controller experiments.

The occupant assignment is cyclic:
Building 1 -> occ1, Building 2 -> occ2, Building 3 -> occ3, Building 4 -> occ4,
then repeat through Building 25.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PYTHON = Path("/home/kostas/anaconda3/bin/python")
LOG_DIR = Path("logs/ce1_occ_D010702")

MONTHS = {
    "feb": (744, 1415),
    "mar": (1416, 2159),
}

GROUPS = {
    "occ1_tolerant": "1,5,9,13,17,21,25",
    "occ2_sensitive": "2,6,10,14,18,22",
    "occ3_cold": "3,7,11,15,19,23",
    "occ4_hot": "4,8,12,16,20,24",
}

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
    "--comfort-cold-scale", "1.5",
    "--comfort-hot-scale", "2.0",
    "--time-bias-amplitude", "1.0",
    "--time-bias-guard-band", "2.0",
    "--power-source", "heating",
    "--single-setpoint",
]

SAC_TRAINING_FLAGS = [
    "--action-scaling-coefficient", "1.0",
    "--jumpstart-controller", "safe_pi",
    "--safe-pi-random-prob", "0.0",
    "--safe-pi-random-low", "0.0",
    "--automatic-entropy-tuning",
    "--entropy-lr", "0.001",
    "--target-entropy", "-1.0",
    "--robust-normalization",
    "--normalized-observation-clip", "5.0",
    "--normalized-reward-clip", "10.0",
    "--training-temperature-reset",
    "--training-temperature-reset-replay-mode", "drop",
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


def mpc_cmd(month: str, occupant: str, buildings: str) -> list[str]:
    start, end = MONTHS[month]
    return [
        str(PYTHON), "-u", "scripts/03_run_mpc.py",
        "--scenario", "occupant_present",
        "--building", buildings,
        "--occupant", occupant,
        "--occupant-mode", "both",
        "--label", f"ce1_occ_D010702_mpc_{month}",
        "--objective", "balanced",
        "--rc-model-type", "2r2c",
        "--two-state-observer", "kalman",
        "--simulation-start", str(start),
        "--simulation-end", str(end),
        *COMMON_COMFORT_FLAGS,
    ]


def rlmpc_cmd(month: str, occupant: str, buildings: str, *, device: str, adaptive_dynamics: bool = False) -> list[str]:
    start, end = MONTHS[month]
    label = "ce1_occ_D010702_rlmpc_dyn" if adaptive_dynamics else "ce1_occ_D010702_rlmpc_fixed"
    cmd = [
        str(PYTHON), "-u", "scripts/05_run_rlmpc_online.py",
        "--scenario", "occupant_present",
        "--building", buildings,
        "--occupant", occupant,
        "--occupant-mode", "both",
        "--label", f"{label}_{month}",
        "--reward", "balanced",
        "--rc-model-type", "1r1c",
        "--simulation-start", str(start),
        "--simulation-end", str(end),
        "--device", device,
        *COMMON_COMFORT_FLAGS,
    ]
    if adaptive_dynamics:
        cmd.append("--adaptive-dynamics")
    return cmd


def sac_initial_cmd(occupant: str, buildings: str, mode: str, *, device: str) -> list[str]:
    return [
        str(PYTHON), "-u", "scripts/08_run_sac_merlin_online.py",
        "--scenario", "occupant_present",
        "--building", buildings,
        "--occupant", occupant,
        "--occupant-mode", mode,
        "--device", device,
        "--label", f"ce1_occ_D010702_sac_jan20_feb_{mode}",
        "--reward", "balanced",
        "--train-sim-start", "0",
        "--train-sim-end", "743",
        "--eval-sim-start", "744",
        "--eval-sim-end", "1415",
        "--training-episodes", "20",
        *COMMON_COMFORT_FLAGS,
        *SAC_TRAINING_FLAGS,
    ]


def sac_continuation_cmd(occupant: str, buildings: str, mode: str, *, device: str) -> list[str]:
    resume = (
        "results/models/sac_merlin/{building_id}/"
        f"ce1_occ_D010702_sac_jan20_feb_{mode}_jan_feb_online_agent.pkl"
    )
    return [
        str(PYTHON), "-u", "scripts/08_run_sac_merlin_online.py",
        "--scenario", "occupant_present",
        "--building", buildings,
        "--occupant", occupant,
        "--occupant-mode", mode,
        "--device", device,
        "--label", f"ce1_occ_D010702_sac_feb10_mar_{mode}",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["mpc", "rlmpc_fixed", "rlmpc_dyn", "sac_initial", "sac_continuation"],
        choices=["mpc", "rlmpc_fixed", "rlmpc_dyn", "sac_initial", "sac_continuation"],
    )
    parser.add_argument("--months", nargs="+", default=["feb", "mar"], choices=["feb", "mar"])
    parser.add_argument("--rlmpc-device", default="cpu")
    parser.add_argument("--sac-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for controller in args.controllers:
        if controller == "mpc":
            for month in args.months:
                for occupant, buildings in GROUPS.items():
                    run(mpc_cmd(month, occupant, buildings), f"mpc_{month}_{occupant}", dry_run=args.dry_run)
        elif controller in {"rlmpc_fixed", "rlmpc_dyn"}:
            for month in args.months:
                for occupant, buildings in GROUPS.items():
                    adaptive_dynamics = controller == "rlmpc_dyn"
                    run(
                        rlmpc_cmd(month, occupant, buildings, device=args.rlmpc_device, adaptive_dynamics=adaptive_dynamics),
                        f"{controller}_{month}_{occupant}",
                        dry_run=args.dry_run,
                    )
        elif controller == "sac_initial":
            for occupant, buildings in GROUPS.items():
                for mode in ("without_tdyn", "bspline_tdyn"):
                    run(
                        sac_initial_cmd(occupant, buildings, mode, device=args.sac_device),
                        f"sac_jan20_feb_{mode}_{occupant}",
                        dry_run=args.dry_run,
                    )
        elif controller == "sac_continuation":
            for occupant, buildings in GROUPS.items():
                for mode in ("without_tdyn", "bspline_tdyn"):
                    run(
                        sac_continuation_cmd(occupant, buildings, mode, device=args.sac_device),
                        f"sac_feb10_mar_{mode}_{occupant}",
                        dry_run=args.dry_run,
                    )


if __name__ == "__main__":
    sys.exit(main())
