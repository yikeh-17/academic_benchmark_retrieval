from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INVENTORY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "metadata_label_inventory.json"
)

MAPPING_PATH = (
    PROJECT_ROOT
    / "configs"
    / "evalagent"
    / "label_mapping.json"
)

FIELD_MAPPING = {
    "domain_mapping": "domain",
    "task_mapping": "task_type",
    "ability_mapping": "ability",
    "difficulty_mapping": "difficulty",
    "dataset_mapping": "dataset_name",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_valid_labels(
    inventory: dict[str, Any],
    field_name: str,
) -> set[str]:
    rows = inventory["fields"].get(field_name, [])

    return {
        str(row["label"])
        for row in rows
        if row["label"]
        not in {
            "<missing>",
            "<empty>",
            "<empty_list>",
        }
    }


def main() -> None:
    inventory = load_json(INVENTORY_PATH)
    mapping = load_json(MAPPING_PATH)

    has_error = False

    print("=" * 70)
    print("EvalAgent label mapping validation")
    print("=" * 70)

    for mapping_name, metadata_field in FIELD_MAPPING.items():
        current_mapping = mapping.get(mapping_name, {})
        valid_labels = extract_valid_labels(
            inventory,
            metadata_field,
        )

        print(
            f"\n[{mapping_name}] "
            f"against [{metadata_field}]"
        )

        invalid_targets = []

        for user_term, target_label in current_mapping.items():
            if target_label not in valid_labels:
                invalid_targets.append(
                    (user_term, target_label)
                )

        if invalid_targets:
            has_error = True
            print("Invalid mappings:")

            for user_term, target_label in invalid_targets:
                print(
                    f"  {user_term!r} -> "
                    f"{target_label!r}"
                )
        else:
            print("OK")

    print("\n" + "=" * 70)

    if has_error:
        print("Validation failed.")
        sys.exit(1)

    print("All mappings are valid.")
    print("=" * 70)


if __name__ == "__main__":
    main()