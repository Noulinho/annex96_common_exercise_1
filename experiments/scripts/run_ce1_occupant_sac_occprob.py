#!/usr/bin/env python
"""Run SAC occupant-probability reward ablation for CE1 occupant experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PYTHON = Path("/home/kostas/anaconda3/bin/python")
LOG_DIR = Path("logs/ce1_occ_D010702_occprob")
PREFIX = "ce1_occ_D010702_sac_occprob"

GROUPS = {
    "occ1_tolerant": "1,5,9,13,17,21,25",
    "occ2_sensitive": "2,6,10,14,18,22",
    "occ3_cold": "3,7,11,15,19,23",
    "occ4_hot": "4,8,12,16,20,24",
}

REWARD_FLAGS = [
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
    "--occupant-probability-base", "0.002",
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
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-100:])
        raise RuntimeError(f"{log_name} failed with code {completed.returncode}\n\n{tail}")


def initial_cmd(occupant: str, buildings: str, mode: str, *, device: str) -> list[str]:
    return [
        str(PYTHON), "-u", "scripts/08_run_sac_merlin_online.py",
        "--scenario", "occupant_present",
        "--building", buildings,
        "--occupant", occupant,
        "--occupant-mode", mode,
        "--device", device,
        "--label", f"{PREFIX}_jan20_feb_{mode}",
        "--reward", "occupant_prob",
        "--train-sim-start", "0",
        "--train-sim-end", "743",
        "--eval-sim-start", "744",
        "--eval-sim-end", "1415",
        "--training-episodes", "20",
        *REWARD_FLAGS,
        *SAC_TRAINING_FLAGS,
    ]


def continuation_cmd(occupant: str, buildings: str, mode: str, *, device: str) -> list[str]:
    resume = (
        "results/models/sac_merlin/{building_id}/"
        f"{PREFIX}_jan20_feb_{mode}_jan_feb_online_agent.pkl"
    )
    return [
        str(PYTHON), "-u", "scripts/08_run_sac_merlin_online.py",
        "--scenario", "occupant_present",
        "--building", buildings,
        "--occupant", occupant,
        "--occupant-mode", mode,
        "--device", device,
        "--label", f"{PREFIX}_feb10_mar_{mode}",
        "--reward", "occupant_prob",
        "--train-sim-start", "744",
        "--train-sim-end", "1415",
        "--eval-sim-start", "1416",
        "--eval-sim-end", "2159",
        "--training-episodes", "10",
        "--resume-agent-template", resume,
        "--resume-episode-offset", "20",
        *REWARD_FLAGS,
        *SAC_TRAINING_FLAGS,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        nargs="+",
        default=["initial", "continuation"],
        choices=["initial", "continuation"],
    )
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for phase in args.phase:
        for occupant, buildings in GROUPS.items():
            for mode in ("without_tdyn", "bspline_tdyn"):
                if phase == "initial":
                    cmd = initial_cmd(occupant, buildings, mode, device=args.device)
                    log_name = f"initial_{mode}_{occupant}"
                else:
                    cmd = continuation_cmd(occupant, buildings, mode, device=args.device)
                    log_name = f"continuation_{mode}_{occupant}"
                run(cmd, log_name, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
