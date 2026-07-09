#!/usr/bin/env python
"""Train SAC on January+February, then deploy deterministically on March."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
from pathlib import Path


PYTHON = Path("/home/kostas/anaconda3/bin/python")
LOG_DIR = Path("logs/ce1_occ_D010702_sac_janfeb20_mar")

BUILDINGS_ALL = ",".join(str(i) for i in range(1, 26))
BUILDING_IDS = list(range(1, 26))
GROUPS = {
    "occ1_tolerant": "1,5,9,13,17,21,25",
    "occ2_sensitive": "2,6,10,14,18,22",
    "occ3_cold": "3,7,11,15,19,23",
    "occ4_hot": "4,8,12,16,20,24",
}

PREFIX = "ce1_occ_D010702_sac_janfeb20_mar"

COMMON_COMFORT_FLAGS = [
    "--reward", "balanced",
    "--train-sim-start", "0",
    "--train-sim-end", "1415",
    "--eval-sim-start", "1416",
    "--eval-sim-end", "2159",
    "--training-episodes", "20",
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


def rollout_exists(building: int, scenario: str, label: str) -> bool:
    path = (
        Path("results/raw/sac_merlin")
        / f"building_{int(building):02d}"
        / scenario
        / label
        / "rollout.csv"
    )
    return path.exists()


def no_occupant_cmd(buildings: str = BUILDINGS_ALL) -> list[str]:
    return [
        str(PYTHON), "-u", "scripts/08_run_sac_merlin_online.py",
        "--scenario", "no_occupant",
        "--building", buildings,
        "--label", f"{PREFIX}_no_occupant",
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
        *COMMON_COMFORT_FLAGS,
        *SAC_FLAGS,
    ]


def build_tasks(modes: list[str]) -> list[tuple[list[str], str, int, str, str]]:
    tasks: list[tuple[list[str], str, int, str, str]] = []
    if "no_occupant" in modes:
        label = f"{PREFIX}_no_occupant"
        for building in BUILDING_IDS:
            if rollout_exists(building, "no_occupant", label):
                print(f"skip no_occupant building {building}: rollout exists")
                continue
            tasks.append((
                no_occupant_cmd(str(building)),
                f"sac_janfeb20_mar_no_occupant_b{building:02d}",
                building,
                "no_occupant",
                label,
            ))
    if "bspline_tdyn" in modes:
        label = f"{PREFIX}_bspline_tdyn"
        for occupant, buildings in GROUPS.items():
            for building_str in buildings.split(","):
                building = int(building_str)
                if rollout_exists(building, "occupant_present", label):
                    print(f"skip bspline_tdyn building {building}: rollout exists")
                    continue
                tasks.append((
                    bspline_tdyn_cmd(occupant, str(building)),
                    f"sac_janfeb20_mar_bspline_tdyn_b{building:02d}_{occupant}",
                    building,
                    "bspline_tdyn",
                    label,
                ))
    return tasks


def run_many(tasks: list[tuple[list[str], str, int, str, str]], *, workers: int, dry_run: bool) -> None:
    if not tasks:
        print("No SAC tasks to run; all requested rollouts already exist.")
        return
    workers = max(1, min(int(workers), len(tasks)))
    if workers == 1:
        for cmd, log_name, _, _, _ in tasks:
            run(cmd, log_name, dry_run=dry_run)
        return
    print(f"Running {len(tasks)} per-building SAC tasks with {workers} workers.")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(run, cmd, log_name, dry_run=dry_run): (building, mode, log_name)
            for cmd, log_name, building, mode, _ in tasks
        }
        for future in as_completed(future_map):
            building, mode, log_name = future_map[future]
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"{mode} building {building} failed in {log_name}") from exc
            print(f"finished {mode} building {building}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["no_occupant", "bspline_tdyn"],
        choices=["no_occupant", "bspline_tdyn"],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of per-building SAC subprocesses to run in parallel.",
    )
    args = parser.parse_args()

    run_many(build_tasks(args.modes), workers=args.workers, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
