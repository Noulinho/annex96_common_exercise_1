#!/usr/bin/env python
"""Offline imitation stage for Gnu-RL/RLMPC.

This is the script version of the working notebook section:

1. load the January PI rollout dataset,
2. train the differentiable MPC by imitating actions and one-step temperature,
3. save the learned DiffMPC checkpoint plus normalization bundle,
4. optionally run the pretrained raw policy on January as a sanity check.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rlmpc_pipeline import (
    OccupantComfortReward,
    RLMPCExperiment,
    run_january_raw_rollout,
    train_offline_diffmpc,
    train_offline_2r2c_diffmpc,
)
from src.buildings import DATASET_NAME, building_metadata, format_building_path, resolve_building_entries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default="no_occupant",
        choices=["no_occupant", "occupant_present", "occupant_present_peak_flattening"],
    )
    parser.add_argument("--occupant", default=None)
    parser.add_argument(
        "--tdyn-variant",
        default=None,
        choices=["feedback_only", "present_only_no_bspline", "bspline_dynamic_comfort"],
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Single dataset path or template. For multiple buildings, include placeholders or use the default.",
    )
    parser.add_argument(
        "--dataset-template",
        default="results/datasets/january_pi/{building_id}/january_pi_dataset.npz",
        help="Per-building January PI dataset path template. Used unless --dataset is supplied.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path or template. For multiple buildings, include placeholders or use the default.",
    )
    parser.add_argument(
        "--checkpoint-template",
        default="results/models/rlmpc/offline/{building_id}/{slug}.pt",
        help="Per-building offline checkpoint template. Supports {slug} plus building placeholders.",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Bundle path or template. For multiple buildings, include placeholders or use the default.",
    )
    parser.add_argument(
        "--bundle-template",
        default="results/models/rlmpc/bundles/{building_id}/{slug}.json",
        help="Per-building offline bundle template. Supports {slug} plus building placeholders.",
    )
    parser.add_argument("--summary", default="results/models/rlmpc/offline/offline_training_summary.csv")
    parser.add_argument(
        "--building",
        default="all",
        help="Building selector: all, building_01, numeric index, short ResStock id, or full schema name.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0, help="Training/shuffle seed, matching notebook rng=0 default.")
    parser.add_argument("--rollout-seed", type=int, default=49)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda-dyn", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--eta", type=float, default=10.0)
    parser.add_argument("--rho-u", type=float, default=1.0)
    parser.add_argument("--u-min", type=float, default=0.0)
    parser.add_argument("--u-max", type=float, default=1.0)
    parser.add_argument("--q-reg", type=float, default=1e-4)
    parser.add_argument("--qp-max-iter", type=int, default=200)
    parser.add_argument("--qp-eps", type=float, default=1e-6)
    parser.add_argument("--max-samples", type=int, default=None, help="Short smoke-test subset of the dataset.")
    parser.add_argument("--observation-preset", default="auto", choices=["auto", "notebook_mpc_28"])
    parser.add_argument(
        "--reward",
        default="occupant_comfort",
        choices=["comfort", "occupant_comfort"],
        help="Reward used only for the optional raw January rollout check.",
    )
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--max-rollout-steps", type=int, default=None)
    parser.add_argument(
        "--rc-model-type",
        default="1r1c",
        choices=["1r1c", "2r2c"],
        help="Thermal model type. '2r2c' trains TwoStateDiffMPC with fixed 2R2C dynamics.",
    )
    parser.add_argument(
        "--rc-model-2r2c",
        default="results/models/rc/2r2c_{short_name}_january.json",
        help="Path template for 2R2C model JSON (used when --rc-model-type 2r2c).",
    )
    return parser


def _has_format_fields(path: str | Path) -> bool:
    text = str(path)
    return "{" in text and "}" in text


def _format_path(template: str | Path, building: dict, experiment: RLMPCExperiment) -> Path:
    values = dict(building)
    values.update(
        {
            "building_id": building["result_id"],
            "building_name": building["name"],
            "building": building["name"],
            "slug": experiment.result_slug,
        }
    )
    return Path(str(template).format(**values))


def _resolve_path(
    *,
    explicit: str | Path | None,
    template: str | Path,
    building: dict,
    experiment: RLMPCExperiment,
    multiple_buildings: bool,
    label: str,
) -> Path:
    value = template if explicit is None else explicit
    if multiple_buildings and explicit is not None and not _has_format_fields(explicit):
        raise ValueError(
            f"--{label} points to one fixed path while --building selects multiple buildings. "
            f"Use --{label}-template or include a placeholder such as {{building_id}}."
        )
    return _format_path(value, building, experiment)


def _resolve_dataset_path(args: argparse.Namespace, building: dict, experiment: RLMPCExperiment, multiple: bool) -> Path:
    return _resolve_path(
        explicit=args.dataset,
        template=args.dataset_template,
        building=building,
        experiment=experiment,
        multiple_buildings=multiple,
        label="dataset",
    )


def main() -> None:
    import json

    args = build_parser().parse_args()
    rc_model_type = getattr(args, "rc_model_type", "1r1c")

    # For 2R2C, use a dedicated checkpoint slug so it doesn't overwrite 1R1C checkpoints
    experiment = RLMPCExperiment(
        controller="rlmpc",
        scenario=args.scenario,
        occupant_label=args.occupant,
        tdyn_variant=args.tdyn_variant,
    )
    buildings = resolve_building_entries(args.building, DATASET_NAME)
    multiple = len(buildings) > 1
    rows = []

    for building in buildings:
        dataset_path = _resolve_dataset_path(args, building, experiment, multiple)

        if rc_model_type == "2r2c":
            rc2_path = Path(format_building_path(args.rc_model_2r2c, building))
            if not rc2_path.exists():
                raise FileNotFoundError(f"Missing 2R2C model for {building['display']}: {rc2_path}")
            rc2_params = json.loads(rc2_path.read_text())
            building_id = building["result_id"]
            slug_2r2c = experiment.result_slug + "_2r2c"
            if args.checkpoint is not None:
                checkpoint_path = Path(args.checkpoint)
            else:
                checkpoint_path = Path(
                    args.checkpoint_template.format(
                        building_id=building_id,
                        building_name=building["name"],
                        building=building["name"],
                        slug=slug_2r2c,
                    )
                )
            if args.bundle is not None:
                bundle_path = Path(args.bundle)
            else:
                bundle_path = Path(
                    args.bundle_template.format(
                        building_id=building_id,
                        building_name=building["name"],
                        building=building["name"],
                        slug=slug_2r2c,
                    )
                )
        else:
            rc2_params = None
            checkpoint_path = _resolve_path(
                explicit=args.checkpoint,
                template=args.checkpoint_template,
                building=building,
                experiment=experiment,
                multiple_buildings=multiple,
                label="checkpoint",
            )
            bundle_path = _resolve_path(
                explicit=args.bundle,
                template=args.bundle_template,
                building=building,
                experiment=experiment,
                multiple_buildings=multiple,
                label="bundle",
            )

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Missing January PI dataset for {building['display']}: {dataset_path}. "
                "Run scripts/00_collect_january_pi_data.py first."
            )

        train_seed = int(args.seed + 1000 * (int(building["index"]) - 1))
        rollout_seed = int(args.rollout_seed + 1000 * (int(building["index"]) - 1))
        metadata = building_metadata(building)
        print(f"\n=== {building['display']}: {building['name']} ===")
        print(f"Dataset: {dataset_path}")
        print(f"Checkpoint: {checkpoint_path}")

        if rc_model_type == "2r2c":
            print(f"RC model type: 2R2C ({rc2_path})")
            checkpoint = train_offline_2r2c_diffmpc(
                dataset_path=dataset_path,
                rc2_params=rc2_params,
                checkpoint_path=checkpoint_path,
                bundle_path=bundle_path,
                horizon=args.horizon,
                eta=args.eta,
                rho_u=args.rho_u,
                u_min=args.u_min,
                u_max=args.u_max,
                q_reg=args.q_reg,
                qp_max_iter=args.qp_max_iter,
                qp_eps=args.qp_eps,
                epochs=args.epochs,
                lr=args.lr,
                lambda_dyn=args.lambda_dyn,
                batch_size=args.batch_size,
                patience=args.patience,
                seed=train_seed,
                device=args.device,
                max_samples=args.max_samples,
                observation_preset=args.observation_preset,
                building_name=str(building["name"]),
                building_metadata=metadata,
            )
        else:
            checkpoint = train_offline_diffmpc(
                dataset_path=dataset_path,
                checkpoint_path=checkpoint_path,
                bundle_path=bundle_path,
                horizon=args.horizon,
                eta=args.eta,
                rho_u=args.rho_u,
                u_min=args.u_min,
                u_max=args.u_max,
                q_reg=args.q_reg,
                qp_max_iter=args.qp_max_iter,
                qp_eps=args.qp_eps,
                epochs=args.epochs,
                lr=args.lr,
                lambda_dyn=args.lambda_dyn,
                batch_size=args.batch_size,
                patience=args.patience,
                seed=train_seed,
                device=args.device,
                max_samples=args.max_samples,
                observation_preset=args.observation_preset,
                building_name=str(building["name"]),
                building_metadata=metadata,
            )

        rollout_reward = None
        rollout_steps = None
        if not args.skip_rollout and rc_model_type != "2r2c":
            reward_function = OccupantComfortReward if args.reward == "occupant_comfort" else None
            rollout = run_january_raw_rollout(
                checkpoint_path=checkpoint_path,
                seed=rollout_seed,
                reward_function=reward_function,
                device=args.device,
                max_steps=args.max_rollout_steps,
                building_name=str(building["name"]),
            )
            rollout_reward = rollout["reward_sum"]
            rollout_steps = rollout["steps"]

        rows.append(
            {
                **metadata,
                "dataset_path": str(dataset_path),
                "checkpoint_path": str(checkpoint_path),
                "bundle_path": str(bundle_path),
                "best_val": checkpoint["best_val"],
                "train_seed": train_seed,
                "rollout_seed": rollout_seed,
                "rollout_reward": rollout_reward,
                "rollout_steps": rollout_steps,
            }
        )

    if rows:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0])
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved offline training summary to {summary_path}")


if __name__ == "__main__":
    main()
