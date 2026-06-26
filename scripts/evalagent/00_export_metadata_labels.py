#!/usr/bin/env python3
"""
Export the actual metadata labels used by the unified benchmark dataset.

Output:
    reports/metadata_label_inventory.txt
    reports/metadata_label_inventory.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "academic_5_merged.jsonl"
)

TEXT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "metadata_label_inventory.txt"
)

JSON_OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "metadata_label_inventory.json"
)


def normalize_scalar(value: Any) -> str:
    """Convert a scalar metadata value into a clean string."""
    if value is None:
        return "<missing>"

    text = str(value).strip()
    return text if text else "<empty>"


def update_multi_value_counter(
    counter: Counter[str],
    value: Any,
) -> None:
    """
    Update a counter for a field that may be:
    - a list
    - a scalar string
    - missing
    """
    if value is None:
        counter["<missing>"] += 1
        return

    if isinstance(value, list):
        if not value:
            counter["<empty_list>"] += 1
            return

        for item in value:
            counter[normalize_scalar(item)] += 1
        return

    counter[normalize_scalar(value)] += 1


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    TEXT_OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    counters: dict[str, Counter[str]] = {
        "dataset_name": Counter(),
        "domain": Counter(),
        "task_type": Counter(),
        "image_type": Counter(),
        "difficulty": Counter(),
        "modality": Counter(),
        "ability": Counter(),
    }

    total_records = 0
    invalid_lines = 0

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_lines += 1
                print(
                    f"[Warning] Invalid JSON at line "
                    f"{line_number}: {exc}"
                )
                continue

            total_records += 1

            for field_name in [
                "dataset_name",
                "domain",
                "task_type",
                "image_type",
                "difficulty",
                "modality",
            ]:
                counters[field_name][
                    normalize_scalar(item.get(field_name))
                ] += 1

            update_multi_value_counter(
                counters["ability"],
                item.get("ability"),
            )

    inventory_json: dict[str, Any] = {
        "source_file": str(INPUT_PATH),
        "total_records": total_records,
        "invalid_lines": invalid_lines,
        "fields": {},
    }

    text_lines = [
        "EvalAgent Metadata Label Inventory",
        "=" * 70,
        f"Source file: {INPUT_PATH}",
        f"Valid records: {total_records}",
        f"Invalid JSON lines: {invalid_lines}",
    ]

    for field_name, counter in counters.items():
        text_lines.append("")
        text_lines.append(f"[{field_name}]")
        text_lines.append("-" * 70)

        sorted_items = counter.most_common()

        inventory_json["fields"][field_name] = [
            {
                "label": label,
                "count": count,
            }
            for label, count in sorted_items
        ]

        for label, count in sorted_items:
            text_lines.append(f"{label}\t{count}")

    TEXT_OUTPUT_PATH.write_text(
        "\n".join(text_lines) + "\n",
        encoding="utf-8",
    )

    JSON_OUTPUT_PATH.write_text(
        json.dumps(
            inventory_json,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 70)
    print("Metadata inventory completed.")
    print(f"Records: {total_records}")
    print(f"Text output: {TEXT_OUTPUT_PATH}")
    print(f"JSON output: {JSON_OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()