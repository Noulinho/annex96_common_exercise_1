#!/usr/bin/env python
"""Run the CE1 p2 all-occupants SAC sweep per building with bounded parallelism.

The main final-run wrapper is intentionally simple and serial.  SAC is much
slower than MPC/RLMPC in that form, so this helper preserves the exact command
builders and labels from ``run_ce1_occupant_p2s2_all_occ.py`` while splitting
the building list into one process per building.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from run_ce1_occupant_p2s2_all_occ import (
    LOG_DIR,
    OCCUPANTS,
    PREFIX,
    MODES,
    sac_no_occupant_initial_cmd,
    sac_no_occupant_continuation_cmd,
    sac_occupant_initial_cmd,
    sac_occupant_continuation_cmd,
)


BUILDINGS = list(range(1, 26))
RESULTS = Path("results")
PARALLEL_LOG_DIR = Path("logs") / f"{PREFIX}_sac_parallel"


@dataclass(frozen=True)
class Task:
    stage: str
    building: int
    label: str
    scenario: str
    cmd: tuple[str, ...]
    rollout_path: Path
    agent_path: Path

    @property
    def name(self) -> str:
        return f"{self.stage}_b{self.building:02d}"

    @property
    def done(self) -> bool:
        return self.rollout_path.exists() and self.agent_path.exists()

    def fresh_after(self, cutoff: datetime | None) -> bool:
        if not self.done:
            return False
        if cutoff is None:
            return True
        rollout_mtime = datetime.fromtimestamp(self.rollout_path.stat().st_mtime)
        agent_mtime = datetime.fromtimestamp(self.agent_path.stat().st_mtime)
        return rollout_mtime >= cutoff and agent_mtime >= cutoff


def replace_arg(cmd: list[str], flag: str, value: str) -> list[str]:
    updated = list(cmd)
    index = updated.index(flag)
    updated[index + 1] = value
    return updated


def building_id(building: int) -> str:
    return f"building_{building:02d}"


def raw_rollout_path(building: int, scenario: str, label: str) -> Path:
    return RESULTS / "raw" / "sac_merlin" / building_id(building) / scenario / label / "rollout.csv"


def agent_path(building: int, label: str) -> Path:
    return (
        RESULTS
        / "models"
        / "sac_merlin"
        / building_id(building)
        / f"{label}_jan_feb_online_agent.pkl"
    )


def task_from_cmd(stage: str, building: int, label: str, scenario: str, cmd: list[str]) -> Task:
    single = replace_arg(cmd, "--building", str(building))
    return Task(
        stage=stage,
        building=building,
        label=label,
        scenario=scenario,
        cmd=tuple(single),
        rollout_path=raw_rollout_path(building, scenario, label),
        agent_path=agent_path(building, label),
    )


def initial_tasks(device: str, occupants: list[str], modes: list[str]) -> list[Task]:
    tasks: list[Task] = []
    if "no_occupant" in modes:
        label = f"{PREFIX}_sac_jan20_feb_no_occupant"
        base = sac_no_occupant_initial_cmd(device=device)
        tasks.extend(
            task_from_cmd("initial_no_occupant", building, label, "no_occupant", base)
            for building in BUILDINGS
        )
    for occupant in occupants:
        for mode in [m for m in modes if m != "no_occupant"]:
            label = f"{PREFIX}_sac_jan20_feb_{occupant}_{mode}"
            base = sac_occupant_initial_cmd(occupant, mode, device=device)
            tasks.extend(
                task_from_cmd(f"initial_{occupant}_{mode}", building, label, "occupant_present", base)
                for building in BUILDINGS
            )
    return tasks


def continuation_tasks(device: str, occupants: list[str], modes: list[str]) -> list[Task]:
    tasks: list[Task] = []
    if "no_occupant" in modes:
        label = f"{PREFIX}_sac_feb10_mar_no_occupant"
        base = sac_no_occupant_continuation_cmd(device=device)
        tasks.extend(
            task_from_cmd("continuation_no_occupant", building, label, "no_occupant", base)
            for building in BUILDINGS
        )
    for occupant in occupants:
        for mode in [m for m in modes if m != "no_occupant"]:
            label = f"{PREFIX}_sac_feb10_mar_{occupant}_{mode}"
            base = sac_occupant_continuation_cmd(occupant, mode, device=device)
            tasks.extend(
                task_from_cmd(f"continuation_{occupant}_{mode}", building, label, "occupant_present", base)
                for building in BUILDINGS
            )
    return tasks


def run_task(task: Task) -> tuple[Task, int]:
    PARALLEL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = PARALLEL_LOG_DIR / f"{task.name}.log"
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")
    with log_path.open("w") as log:
        log.write(" ".join(task.cmd) + "\n\n")
        log.flush()
        completed = subprocess.run(
            list(task.cmd),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    return task, int(completed.returncode)


def run_stage(
    tasks: list[Task],
    max_workers: int,
    dry_run: bool = False,
    force: bool = False,
    rerun_before: datetime | None = None,
) -> None:
    pending = list(tasks) if force else [task for task in tasks if not task.fresh_after(rerun_before)]
    skipped = 0 if force else len(tasks) - len(pending)
    print(f"Stage tasks: {len(tasks)} total, {skipped} skipped, {len(pending)} pending")
    if dry_run:
        for task in pending[:20]:
            print("DRY", task.name, " ".join(task.cmd))
        if len(pending) > 20:
            print(f"... {len(pending) - 20} more")
        return

    running: set[Future[tuple[Task, int]]] = set()
    failures: list[tuple[Task, int]] = []

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending_iter = iter(pending)

        def submit_next() -> bool:
            try:
                task = next(pending_iter)
            except StopIteration:
                return False
            print(f"START {task.name}")
            running.add(executor.submit(run_task, task))
            return True

        for _ in range(max_workers):
            if not submit_next():
                break

        completed_count = skipped
        while running:
            done, running_now = wait(running, return_when=FIRST_COMPLETED)
            running = set(running_now)
            for future in done:
                task, code = future.result()
                completed_count += 1
                if code == 0 and task.done:
                    print(f"DONE  {task.name} ({completed_count}/{len(tasks)})")
                else:
                    print(f"FAIL  {task.name} code={code} ({completed_count}/{len(tasks)})")
                    failures.append((task, code))
                if not failures:
                    submit_next()
            if failures:
                break

    if failures:
        task, code = failures[0]
        log_path = PARALLEL_LOG_DIR / f"{task.name}.log"
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(f"{task.name} failed with code {code}\n\n{tail}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["initial", "continuation"],
        choices=["initial", "continuation"],
    )
    parser.add_argument("--occupants", nargs="+", default=OCCUPANTS, choices=OCCUPANTS)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["no_occupant", *MODES],
        choices=["no_occupant", *MODES],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun tasks even when rollout/agent artifacts already exist.")
    parser.add_argument(
        "--rerun-before",
        default=None,
        help="Rerun tasks whose rollout or agent artifact is older than this ISO timestamp.",
    )
    args = parser.parse_args()
    rerun_before = datetime.fromisoformat(args.rerun_before) if args.rerun_before else None

    if "initial" in args.stages:
        print("=== SAC initial Jan20 -> Feb ===")
        run_stage(
            initial_tasks(args.device, args.occupants, args.modes),
            args.max_workers,
            args.dry_run,
            args.force,
            rerun_before,
        )
    if "continuation" in args.stages:
        print("=== SAC continuation Feb10 -> Mar ===")
        run_stage(
            continuation_tasks(args.device, args.occupants, args.modes),
            args.max_workers,
            args.dry_run,
            args.force,
            rerun_before,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
