#!/usr/bin/env python
"""Resume no-reset SAC with a small January replay tail, train February, deploy March."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PYTHON = Path("/home/kostas/anaconda3/bin/python")
LOG_DIR = Path("logs/ce1_occ_D010702_sac_noreset_replaytail")

BUILDINGS_ALL = ",".join(str(i) for i in range(1, 26))
GROUPS = {
    "occ1_tolerant": "1,5,9,13,17,21,25",
    "occ2_sensitive": "2,6,10,14,18,22",
    "occ3_cold": "3,7,11,15,19,23",
    "occ4_hot": "4,8,12,16,20,24",
}

PREFIX = "ce1_occ_D010702_sac_noreset_replaytail1024_feb10_mar"

COMMON_COMFORT_FLAGS = [
    "--reward", "balanced",
    "--train-sim-start", "744",
    "--train-sim-end", "1415",
    "--eval-sim-start", "1416",
    "--eval-sim-end", "2159",
    "--training-episodes", "10",
    "--resume-episode-offset", "20",
    "--resume-replay-keep-last", "1024",
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

SAC_FLAGS = [
    "--device", "cuda",
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


def no_occupant_cmd() -> list[str]:
    return [
        str(PYTHON), "-u", "scripts/08_run_sac_merlin_online.py",
        "--scenario", "no_occupant",
        "--building", BUILDINGS_ALL,
        "--label", f"{PREFIX}_no_occupant",
        "--resume-agent-template",
        "results/models/sac_merlin/{building_id}/ce1_occ_D010702_sac_noreset_jan20_feb_no_occupant_jan_feb_online_agent.pkl",
        *COMMON_COMFORT_FLAGS,
        *SAC_FLAGS,
    ]


def bspline_tdyn_cmd(occupant: str, buildings: str) -> list[str]:
    return [
        str(PYTHON), "-u", "scripts/08_run_sac_merlin_online.py",
        "--scenario", "occupant_present",
        "--building", buildings,
        "--occupant", occupant,
        "--occupant-mode", "bspline_tdyn",
        "--label", f"{PREFIX}_bspline_tdyn",
        "--resume-agent-template",
        "results/models/sac_merlin/{building_id}/ce1_occ_D010702_sac_noreset_jan20_feb_bspline_tdyn_jan_feb_online_agent.pkl",
        *COMMON_COMFORT_FLAGS,
        *SAC_FLAGS,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["no_occupant", "bspline_tdyn"],
        choices=["no_occupant", "bspline_tdyn"],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if "no_occupant" in args.modes:
        run(no_occupant_cmd(), "sac_noreset_replaytail1024_feb10_mar_no_occupant", dry_run=args.dry_run)
    if "bspline_tdyn" in args.modes:
        for occupant, buildings in GROUPS.items():
            run(
                bspline_tdyn_cmd(occupant, buildings),
                f"sac_noreset_replaytail1024_feb10_mar_bspline_tdyn_{occupant}",
                dry_run=args.dry_run,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
