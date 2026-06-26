#!/usr/bin/env python3
"""
Build a retrieval query from a structured EvalAgent profile.

Input:
    A JSON file produced by 01_parse_requirement.py

Output:
    A JSON file containing:
    - query_profile
    - retrieval_query
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation_runs"
    / "parsed_requirement.json"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation_runs"
    / "retrieval_query.json"
)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def join_values(
    values: list[str],
    fallback: str,
) -> str:
    """Join a list into readable retrieval text."""
    cleaned = [
        str(value).strip()
        for value in values
        if str(value).strip()
    ]

    if not cleaned:
        return fallback

    return ", ".join(cleaned)


def build_retrieval_query(
    profile: dict[str, Any],
) -> str:
    """Convert a query profile into an embedding-friendly query."""
    domains = join_values(
        profile.get("target_domains", []),
        "cross-disciplinary academic research",
    )

    tasks = join_values(
        profile.get("target_tasks", []),
        "multimodal visual question answering",
    )

    abilities = join_values(
        profile.get("target_abilities", []),
        "visual understanding and reasoning",
    )

    difficulties = join_values(
        profile.get("preferred_difficulties", []),
        "mixed difficulty",
    )

    datasets = join_values(
        profile.get("preferred_datasets", []),
        "multiple academic benchmarks",
    )

    model_description = str(
        profile.get("model_description", "")
    ).strip()

    query_parts = [
        (
            "Retrieve multimodal academic benchmark samples "
            "for evaluating a research model."
        ),
        f"Target domains: {domains}.",
        f"Target tasks: {tasks}.",
        f"Target abilities: {abilities}.",
        f"Preferred difficulty: {difficulties}.",
        f"Preferred benchmark sources: {datasets}.",
    ]

    if model_description:
        query_parts.append(
            f"Original requirement: {model_description}"
        )

    return " ".join(query_parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an embedding retrieval query "
            "from a structured EvalAgent profile."
        )
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="Input profile JSON path.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output retrieval query JSON path.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    output_path = Path(args.output)

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    input_data = load_json(input_path)

    profile = input_data.get(
        "query_profile",
        input_data,
    )

    retrieval_query = build_retrieval_query(profile)

    result = {
        "query_builder_version": "query_builder_v1",
        "query_profile": profile,
        "retrieval_query": retrieval_query,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 70)
    print("EvalAgent retrieval query construction completed")
    print("=" * 70)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("=" * 70)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()