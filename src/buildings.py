"""Dataset building selection helpers for experiment scripts and notebooks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DATASET_NAME = "annex96_ce1_vt_neighborhood"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_dir(dataset_name: str = DATASET_NAME) -> Path:
    return repo_root() / "data" / "datasets" / dataset_name


def schema_path(dataset_name: str = DATASET_NAME) -> Path:
    return dataset_dir(dataset_name) / "schema.json"


def dataset_building_names(dataset_name: str = DATASET_NAME) -> list[str]:
    schema = json.loads(schema_path(dataset_name).read_text())
    return sorted(schema["buildings"])


def building_short_name(building_name: str) -> str:
    match = re.search(r"(\d+)$", building_name)
    return match.group(1) if match else re.sub(r"[^A-Za-z0-9]+", "_", building_name).strip("_")


def dataset_building_entries(dataset_name: str = DATASET_NAME) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, name in enumerate(dataset_building_names(dataset_name), start=1):
        short_name = building_short_name(name)
        result_id = f"building_{index:02d}"
        entries.append(
            {
                "index": index,
                "display": f"Building {index}",
                "result_id": result_id,
                "name": name,
                "short_name": short_name,
            }
        )
    return entries


def resolve_building_entries(selector: str, dataset_name: str = DATASET_NAME) -> list[dict[str, Any]]:
    entries = dataset_building_entries(dataset_name)
    selector = str(selector).strip()
    if selector.lower() == "all":
        return entries

    if "," in selector:
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for part in selector.split(","):
            for entry in resolve_building_entries(part.strip(), dataset_name):
                if entry["result_id"] not in seen:
                    selected.append(entry)
                    seen.add(entry["result_id"])
        return selected

    matches = []
    for entry in entries:
        aliases = {
            entry["name"],
            entry["result_id"],
            entry["display"],
            entry["display"].lower().replace(" ", "_"),
            entry["short_name"],
            str(entry["index"]),
        }
        if selector in aliases:
            matches.append(entry)

    if not matches:
        available = ", ".join(f"{e['result_id']}={e['name']}" for e in entries)
        raise ValueError(f"Unknown building selector '{selector}'. Use 'all' or one of: {available}")
    return matches


def format_building_path(template: str | Path, building: dict[str, Any]) -> Path:
    values = {
        "building": building["name"],
        "building_name": building["name"],
        "building_id": building["result_id"],
        "index": building["index"],
        "short_name": building["short_name"],
    }
    return Path(str(template).format(**values))


def building_metadata(building: dict[str, Any], *, occupancy_building_name: str | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "building": building["display"],
        "building_id": building["result_id"],
        "building_index": int(building["index"]),
        "building_name": building["name"],
    }
    if occupancy_building_name is not None:
        metadata["occupancy_building_name"] = occupancy_building_name
    return metadata
