#!/usr/bin/env python
"""Run CE1 occupant RLMPC adaptive-dynamics March continuation.

Starts each March occupant/mode run from the corresponding February
adaptive-dynamics online checkpoint.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.run_ce1_occupant_D010702_full import COMMON_COMFORT_FLAGS, GROUPS, LOG_DIR, PYTHON


def run(cmd: list[str], log_name: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{log_name}.log"
    print(f"\n>>> {log_name}")
    print(" ".join(cmd))
    print(f"log: {log_path}")
    with log_path.open("w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(f"{log_name} failed with code {completed.returncode}\n\n{tail}")


def cmd(occupant: str, buildings: str, mode: str) -> list[str]:
    checkpoint_template = (
        "results/raw/rlmpc/{building_id}/occupant_present/"
        f"ce1_occ_D010702_rlmpc_dyn_feb_{occupant}_{mode}/online_updated_checkpoint.pt"
    )
    return [
        str(PYTHON), "-u", "scripts/05_run_rlmpc_online.py",
        "--scenario", "occupant_present",
        "--building", buildings,
        "--occupant", occupant,
        "--occupant-mode", mode,
        "--label", "ce1_occ_D010702_rlmpc_dyn_mar",
        "--reward", "balanced",
        "--rc-model-type", "1r1c",
        "--checkpoint-template", checkpoint_template,
        "--simulation-start", "1416",
        "--simulation-end", "2159",
        "--device", "cpu",
        "--adaptive-dynamics",
        *COMMON_COMFORT_FLAGS,
    ]


def main() -> int:
    for occupant, buildings in GROUPS.items():
        for mode in ("without_tdyn", "bspline_tdyn"):
            run(cmd(occupant, buildings, mode), f"rlmpc_dyn_mar_{mode}_{occupant}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
