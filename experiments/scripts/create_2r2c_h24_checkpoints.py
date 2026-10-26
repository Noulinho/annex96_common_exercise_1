#!/usr/bin/env python3
"""Clone 2R2C DiffMPC checkpoints with a horizon-24 model config.

The matched-cost 2R2C RLMPC runs overwrite the learned q/r tracking cost with
the explicit MPC stage cost, so the learned scalar weights are not the relevant
part of the checkpoint.  This utility preserves the fitted 2R2C dynamics,
normalization, indices, and q/r state while changing only the DiffMPC horizon.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-label", default="ce1_time_bias_guard_1c_2r2c_kalman")
    parser.add_argument("--target-label", default="ce1_time_bias_guard_1c_2r2c_kalman_h24")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--root", default="results/models/rlmpc/offline")
    parser.add_argument("--bundle-root", default="results/models/rlmpc/bundles")
    parser.add_argument("--building", default="all")
    return parser


def selected_buildings(root: Path, selector: str) -> list[Path]:
    if selector == "all":
        return sorted(p for p in root.glob("building_*") if p.is_dir())
    values = [value.strip() for value in selector.split(",") if value.strip()]
    out: list[Path] = []
    for value in values:
        if value.startswith("building_"):
            bid = value
        else:
            bid = f"building_{int(value):02d}"
        out.append(root / bid)
    return out


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root)
    bundle_root = Path(args.bundle_root)
    rows = []

    for building_dir in selected_buildings(root, args.building):
        source = building_dir / f"{args.source_label}.pt"
        if not source.exists():
            raise FileNotFoundError(source)
        target = building_dir / f"{args.target_label}.pt"
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        checkpoint = deepcopy(checkpoint)
        checkpoint.setdefault("model_config", {})
        old_horizon = checkpoint["model_config"].get("horizon")
        checkpoint["model_config"]["horizon"] = int(args.horizon)
        checkpoint["horizon_override_source"] = str(source)
        checkpoint["horizon_override_old_horizon"] = int(old_horizon)
        checkpoint["horizon_override_new_horizon"] = int(args.horizon)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, target)

        bundle_dir = bundle_root / building_dir.name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            key: value
            for key, value in checkpoint.items()
            if isinstance(value, (str, int, float, bool, dict, list, type(None)))
        }
        bundle["checkpoint_path"] = str(target)
        bundle_path = bundle_dir / f"{args.target_label}.json"
        bundle_path.write_text(json.dumps(bundle, indent=2, default=str))

        rows.append((building_dir.name, old_horizon, args.horizon, target))

    for building, old_horizon, new_horizon, target in rows:
        print(f"{building}: horizon {old_horizon} -> {new_horizon} | {target}")


if __name__ == "__main__":
    main()
