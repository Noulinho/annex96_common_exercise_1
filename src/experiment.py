"""Command entry point for MERLIN-style experiment execution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scenarios import get_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controller experiments.")
    parser.add_argument("controller", choices=["rbc", "mpc", "rl", "rlmpc", "gnu_rl", "compare"])
    parser.add_argument("scenario", nargs="?", default="no_occupant")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "summarize"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.controller == "gnu_rl":
        args.controller = "rlmpc"

    scenario = get_scenario(args.scenario)
    print(
        f"{args.command}: controller={args.controller}, scenario={scenario.name}, "
        f"occupant={scenario.occupant_enabled}, peak_flattening={scenario.peak_flattening_enabled}"
    )
    if args.controller == "mpc" and args.command == "run":
        raise SystemExit(
            "MPC execution is scaffolded. Move the working do-mpc loop from "
            "notebooks/my_mpc.ipynb into src/controllers/mpc.py, then wire it here."
        )
    if args.controller == "rlmpc" and args.command == "run":
        raise SystemExit(
            "RLMPC execution is scaffolded. Use scripts/04_train_rlmpc_offline.py "
            "and scripts/05_run_rlmpc_online.py while migrating the notebook loops."
        )


if __name__ == "__main__":
    main()
