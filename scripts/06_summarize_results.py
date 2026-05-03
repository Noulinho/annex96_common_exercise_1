#!/usr/bin/env python
"""Summarize saved rollout metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", required=True, help="CSV or parquet rollout file.")
    parser.add_argument("--output", required=True, help="Metrics JSON output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import pandas as pd

    from src.evaluation import evaluate_rollout, save_metrics

    path = Path(args.rollout)
    if path.suffix == ".parquet":
        history = pd.read_parquet(path)
    else:
        history = pd.read_csv(path)

    metrics = evaluate_rollout(history)
    save_metrics(metrics, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
