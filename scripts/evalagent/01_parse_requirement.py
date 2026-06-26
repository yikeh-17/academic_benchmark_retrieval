#!/usr/bin/env python3
"""
Rule-based requirement parser for EvalAgent Phase 1.

Input:
    Natural-language Chinese or English evaluation requirement.

Output:
    Structured query profile JSON.

Example:
    python scripts/evalagent/01_parse_requirement.py \
      --requirement "我想评测量子实验图理解，重点关注趋势分析，预算20条。" \
      --output data/evaluation_runs/demo_quantum_profile.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MAPPING_PATH = (
    PROJECT_ROOT
    / "configs"
    / "evalagent"
    / "label_mapping.json"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation_runs"
    / "parsed_requirement.json"
)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def unique_preserve_order(values: list[str]) -> list[str]:
    """Remove duplicate values while preserving order."""
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)

    return result


def match_mapping(
    requirement_lower: str,
    mapping: dict[str, str],
) -> list[str]:
    """
    Match user expressions against a mapping dictionary.

    Longer expressions are checked first so that phrases such as
    '实验状态判断' are matched before '实验状态'.
    """
    matched: list[str] = []

    sorted_mapping = sorted(
        mapping.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for source_term, target_label in sorted_mapping:
        source_term_lower = str(source_term).lower()

        if source_term_lower in requirement_lower:
            matched.append(str(target_label))

    return unique_preserve_order(matched)


def extract_budget(
    requirement: str,
    default_budget: int = 20,
) -> int:
    """Extract the requested evaluation sample budget."""
    patterns = [
        r"预算\s*(?:为|是|=|:|：)?\s*(\d+)\s*条",
        r"预算\s*(?:为|是|=|:|：)?\s*(\d+)",
        r"测试\s*(\d+)\s*条",
        r"选择\s*(\d+)\s*条",
        r"推荐\s*(\d+)\s*条",
        r"抽取\s*(\d+)\s*条",
        r"需要\s*(\d+)\s*条",
        r"(\d+)\s*条(?:样本|题目|数据)?",
        r"budget\s*(?:is|=|:)?\s*(\d+)",
        r"top[-_\s]?k\s*(?:is|=|:)?\s*(\d+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            requirement,
            flags=re.IGNORECASE,
        )

        if match:
            budget = int(match.group(1))

            # Prevent unreasonable input values.
            return max(1, min(budget, 500))

    return default_budget


def extract_model_name(requirement: str) -> str | None:
    """Try to extract an explicitly named model."""
    patterns = [
        r"模型名称\s*(?:为|是|=|:|：)?\s*([A-Za-z0-9_.\-/]+)",
        r"模型\s*(?:为|是|=|:|：)\s*([A-Za-z0-9_.\-/]+)",
        r"model\s+name\s*(?:is|=|:)?\s*([A-Za-z0-9_.\-/]+)",
        r"model\s*(?:is|=|:)\s*([A-Za-z0-9_.\-/]+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            requirement,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1).strip()

    return None


def detect_general_model_description(requirement: str) -> str:
    """
    Keep the complete user requirement as model/evaluation description.

    Phase 1 does not attempt complex semantic summarization.
    """
    return requirement.strip()


def parse_requirement(requirement: str) -> dict[str, Any]:
    """Parse a natural-language requirement into a structured profile."""
    mapping_config = load_json(MAPPING_PATH)
    requirement_lower = requirement.lower()

    domain_mapping = mapping_config.get("domain_mapping", {})
    task_mapping = mapping_config.get("task_mapping", {})
    ability_mapping = mapping_config.get("ability_mapping", {})
    difficulty_mapping = mapping_config.get(
        "difficulty_mapping",
        {},
    )
    dataset_mapping = mapping_config.get("dataset_mapping", {})

    normalized_dataset_mapping = {
        str(key).lower(): str(value)
        for key, value in dataset_mapping.items()
    }

    profile = {
        "model_name": extract_model_name(requirement),
        "model_description": detect_general_model_description(
            requirement
        ),
        "target_domains": match_mapping(
            requirement_lower,
            domain_mapping,
        ),
        "target_tasks": match_mapping(
            requirement_lower,
            task_mapping,
        ),
        "target_abilities": match_mapping(
            requirement_lower,
            ability_mapping,
        ),
        "preferred_difficulties": match_mapping(
            requirement_lower,
            difficulty_mapping,
        ),
        "preferred_datasets": match_mapping(
            requirement_lower,
            normalized_dataset_mapping,
        ),
        "evaluation_budget": extract_budget(requirement),
    }

    return profile


def validate_profile(profile: dict[str, Any]) -> None:
    """Perform basic structural validation."""
    required_list_fields = [
        "target_domains",
        "target_tasks",
        "target_abilities",
        "preferred_difficulties",
        "preferred_datasets",
    ]

    for field_name in required_list_fields:
        if not isinstance(profile.get(field_name), list):
            raise TypeError(
                f"{field_name} must be a list, "
                f"got {type(profile.get(field_name)).__name__}"
            )

    budget = profile.get("evaluation_budget")

    if not isinstance(budget, int):
        raise TypeError("evaluation_budget must be an integer")

    if not 1 <= budget <= 500:
        raise ValueError(
            "evaluation_budget must be between 1 and 500"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse a natural-language evaluation requirement "
            "into an EvalAgent query profile."
        )
    )

    parser.add_argument(
        "--requirement",
        required=True,
        help="Natural-language evaluation requirement.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output JSON file path.",
    )

    args = parser.parse_args()

    requirement = args.requirement.strip()

    if not requirement:
        raise ValueError("Requirement cannot be empty.")

    profile = parse_requirement(requirement)
    validate_profile(profile)

    result = {
        "parser_version": "rule_parser_v1",
        "raw_requirement": requirement,
        "query_profile": profile,
    }

    output_path = Path(args.output)

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

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
    print("EvalAgent requirement parsing completed")
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