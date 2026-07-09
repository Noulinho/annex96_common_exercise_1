#!/usr/bin/env python
"""Run CityLearn's no-control BaselineAgent for the full CE1 district."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.buildings import DATASET_NAME
from src.district_analysis import FEB_END, FEB_START, run_citylearn_baseline_district


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/raw/citylearn_baseline/no_occupant/baseline")
    parser.add_argument("--summary-dir", default="results/summaries/citylearn_baseline/no_occupant")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--simulation-start", type=int, default=FEB_START)
    parser.add_argument("--simulation-end", type=int, default=FEB_END)
    parser.add_argument("--seed", type=int, default=49)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = run_citylearn_baseline_district(
        output_dir=args.output_dir,
        summary_dir=args.summary_dir,
        dataset_name=args.dataset_name,
        simulation_start=args.simulation_start,
        simulation_end=args.simulation_end,
        seed=args.seed,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
