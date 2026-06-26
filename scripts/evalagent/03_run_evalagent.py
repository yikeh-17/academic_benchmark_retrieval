#!/usr/bin/env python3
"""
EvalAgent Phase 1 main workflow.

Workflow:
1. Parse a natural-language evaluation requirement.
2. Build an embedding-friendly retrieval query.
3. Generate a query embedding with qwen3-vl-embedding.
4. Search the existing Academic-5 Zvec collection.
5. Rerank candidates using structured requirements.
6. Aggregate dataset and capability coverage.
7. Save the recommendation result as JSON.

Example:
    python scripts/evalagent/03_run_evalagent.py \
      --request-id demo_quantum \
      --requirement "我想评测量子实验图理解，重点关注趋势分析、参数提取和科学推理，预算20条。"
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import zvec


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PARSE_SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "evalagent"
    / "01_parse_requirement.py"
)

QUERY_BUILDER_SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "evalagent"
    / "02_build_retrieval_query.py"
)

SEARCH_SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "09_search_zvec.py"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation_runs"
)


def load_python_module(
    module_name: str,
    path: Path,
) -> ModuleType:
    """Load a Python script as a module."""
    if not path.exists():
        raise FileNotFoundError(
            f"Python script not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load module from: {path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete EvalAgent Phase 1 "
            "benchmark recommendation workflow."
        )
    )

    parser.add_argument(
        "--request-id",
        type=str,
        required=True,
        help="Unique ID for this evaluation request.",
    )

    parser.add_argument(
        "--requirement",
        type=str,
        required=True,
        help="Natural-language evaluation requirement.",
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional local query image.",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=None,
        help=(
            "Number of Zvec candidates retrieved before reranking. "
            "Default: max(evaluation_budget * 3, 50)."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path.",
    )

    return parser.parse_args()


def normalize_scalar(value: Any) -> str:
    """Normalize a scalar metadata value."""
    if value is None:
        return ""

    return str(value).strip()


def normalize_ability(value: Any) -> list[str]:
    """
    Normalize the ability metadata field.

    It may be stored as:
    - a Python list
    - a JSON string
    - a Python-list-like string
    - a comma-separated string
    - a single string
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [
            normalize_scalar(item)
            for item in value
            if normalize_scalar(item)
        ]

    text = normalize_scalar(value)

    if not text:
        return []

    try:
        parsed_json = json.loads(text)

        if isinstance(parsed_json, list):
            return [
                normalize_scalar(item)
                for item in parsed_json
                if normalize_scalar(item)
            ]
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        parsed_python = ast.literal_eval(text)

        if isinstance(parsed_python, list):
            return [
                normalize_scalar(item)
                for item in parsed_python
                if normalize_scalar(item)
            ]
    except (ValueError, SyntaxError):
        pass

    if "," in text:
        return [
            item.strip()
            for item in text.split(",")
            if item.strip()
        ]

    return [text]


def safe_similarity(
    search_module: ModuleType,
    distance: float,
) -> float:
    """Use the similarity conversion from the original search script."""
    return float(
        search_module.safe_similarity(distance)
    )


def calculate_requirement_score(
    fields: dict[str, Any],
    profile: dict[str, Any],
) -> tuple[float, list[str]]:
    """
    Calculate a metadata-based requirement matching score.

    This score supplements the vector similarity and makes the final
    recommendation reflect the user's structured requirements.
    """
    score = 0.0
    reasons: list[str] = []

    dataset_name = normalize_scalar(
        fields.get("dataset_name")
    )
    domain = normalize_scalar(
        fields.get("domain")
    )
    task_type = normalize_scalar(
        fields.get("task_type")
    )
    difficulty = normalize_scalar(
        fields.get("difficulty")
    )
    abilities = set(
        normalize_ability(fields.get("ability"))
    )

    preferred_datasets = set(
        profile.get("preferred_datasets", [])
    )
    target_domains = set(
        profile.get("target_domains", [])
    )
    target_tasks = set(
        profile.get("target_tasks", [])
    )
    target_abilities = set(
        profile.get("target_abilities", [])
    )
    preferred_difficulties = set(
        profile.get("preferred_difficulties", [])
    )

    if preferred_datasets and dataset_name in preferred_datasets:
        score += 1.5
        reasons.append(
            f"matches preferred dataset {dataset_name}"
        )

    if target_domains and domain in target_domains:
        score += 2.0
        reasons.append(
            f"matches target domain {domain}"
        )

    if target_tasks and task_type in target_tasks:
        score += 2.0
        reasons.append(
            f"matches target task {task_type}"
        )

    matched_abilities = sorted(
        abilities.intersection(target_abilities)
    )

    if matched_abilities:
        score += min(
            2.0,
            0.75 * len(matched_abilities),
        )
        reasons.append(
            "matches abilities: "
            + ", ".join(matched_abilities)
        )

    if (
        preferred_difficulties
        and difficulty in preferred_difficulties
    ):
        score += 1.0
        reasons.append(
            f"matches preferred difficulty {difficulty}"
        )

    return score, reasons


def convert_result(
    rank: int,
    result: Any,
    profile: dict[str, Any],
    search_module: ModuleType,
) -> dict[str, Any]:
    """Convert a Zvec query result into JSON-safe data."""
    fields = dict(result.fields or {})

    distance = float(result.score)
    similarity = safe_similarity(
        search_module,
        distance,
    )

    metadata_score, reasons = (
        calculate_requirement_score(
            fields=fields,
            profile=profile,
        )
    )

    # Vector similarity remains the primary signal.
    # Metadata matching is used as an additional reranking signal.
    final_score = similarity + 0.05 * metadata_score

    if not reasons:
        reasons.append(
            "selected primarily by semantic vector similarity"
        )

    return {
        "candidate_rank": rank,
        "zvec_id": str(result.id),
        "sample_id": normalize_scalar(
            fields.get("original_id")
        ),
        "dataset_name": normalize_scalar(
            fields.get("dataset_name")
        ),
        "question": normalize_scalar(
            fields.get("question")
        ),
        "image_path": normalize_scalar(
            fields.get("image_path")
        ),
        "task_type": normalize_scalar(
            fields.get("task_type")
        ),
        "domain": normalize_scalar(
            fields.get("domain")
        ),
        "difficulty": normalize_scalar(
            fields.get("difficulty")
        ),
        "ability": normalize_ability(
            fields.get("ability")
        ),
        "distance": round(distance, 6),
        "similarity_score": round(
            similarity,
            6,
        ),
        "metadata_match_score": round(
            metadata_score,
            4,
        ),
        "final_score": round(
            final_score,
            6,
        ),
        "recommendation_reason": "; ".join(
            reasons
        ),
    }


def rerank_candidates(
    candidates: list[dict[str, Any]],
    budget: int,
) -> list[dict[str, Any]]:
    """Sort candidates and select the final requested number."""
    ranked = sorted(
        candidates,
        key=lambda item: (
            item["final_score"],
            item["similarity_score"],
        ),
        reverse=True,
    )

    selected = ranked[:budget]

    for final_rank, item in enumerate(
        selected,
        start=1,
    ):
        item["rank"] = final_rank

    return selected


def aggregate_dataset_summary(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate recommendation statistics by dataset."""
    counts: Counter[str] = Counter()
    similarities: dict[str, list[float]] = defaultdict(list)

    for sample in samples:
        dataset_name = (
            sample.get("dataset_name")
            or "Unknown"
        )

        counts[dataset_name] += 1
        similarities[dataset_name].append(
            float(sample["similarity_score"])
        )

    total = len(samples)
    summary: dict[str, Any] = {}

    for dataset_name, count in counts.most_common():
        dataset_scores = similarities[dataset_name]

        summary[dataset_name] = {
            "count": count,
            "proportion": (
                round(count / total, 4)
                if total
                else 0.0
            ),
            "average_similarity": round(
                sum(dataset_scores)
                / len(dataset_scores),
                6,
            ),
            "best_similarity": round(
                max(dataset_scores),
                6,
            ),
        }

    return summary


def count_nonempty(
    samples: list[dict[str, Any]],
    field_name: str,
) -> dict[str, int]:
    """Count non-empty scalar values."""
    counter: Counter[str] = Counter()

    for sample in samples:
        value = normalize_scalar(
            sample.get(field_name)
        )

        if value:
            counter[value] += 1

    return dict(counter.most_common())


def count_abilities(
    samples: list[dict[str, Any]],
) -> dict[str, int]:
    """Count all ability labels in selected samples."""
    counter: Counter[str] = Counter()

    for sample in samples:
        counter.update(
            sample.get("ability", [])
        )

    return dict(counter.most_common())


def build_coverage_summary(
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the final capability coverage summary."""
    return {
        "datasets": count_nonempty(
            samples,
            "dataset_name",
        ),
        "domains": count_nonempty(
            samples,
            "domain",
        ),
        "tasks": count_nonempty(
            samples,
            "task_type",
        ),
        "abilities": count_abilities(samples),
        "difficulties": count_nonempty(
            samples,
            "difficulty",
        ),
    }


def build_recommendation_summary(
    samples: list[dict[str, Any]],
    dataset_summary: dict[str, Any],
    profile: dict[str, Any],
) -> str:
    """Generate a compact human-readable recommendation summary."""
    if not samples:
        return "No benchmark samples were recommended."

    main_datasets = [
        f"{name} ({info['count']})"
        for name, info in list(
            dataset_summary.items()
        )[:3]
    ]

    target_parts: list[str] = []

    if profile.get("target_domains"):
        target_parts.append(
            "domains="
            + ", ".join(
                profile["target_domains"]
            )
        )

    if profile.get("target_tasks"):
        target_parts.append(
            "tasks="
            + ", ".join(
                profile["target_tasks"]
            )
        )

    if profile.get("target_abilities"):
        target_parts.append(
            "abilities="
            + ", ".join(
                profile["target_abilities"]
            )
        )

    target_text = (
        "; ".join(target_parts)
        if target_parts
        else "general multimodal academic evaluation"
    )

    return (
        f"Selected {len(samples)} benchmark samples "
        f"for {target_text}. "
        f"The main recommended sources are "
        f"{', '.join(main_datasets)}."
    )


def main() -> None:
    args = parse_args()

    requirement = args.requirement.strip()

    if not requirement:
        raise ValueError(
            "--requirement cannot be empty."
        )

    request_id = args.request_id.strip()

    if not request_id:
        raise ValueError(
            "--request-id cannot be empty."
        )

    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set. "
            "Please export the API key before running EvalAgent."
        )

    print("=" * 80)
    print("EvalAgent Phase 1")
    print("=" * 80)
    print(f"Request ID: {request_id}")
    print(f"Requirement: {requirement}")

    print("\n[1/6] Loading reusable modules...")

    parser_module = load_python_module(
        "evalagent_requirement_parser",
        PARSE_SCRIPT_PATH,
    )

    query_builder_module = load_python_module(
        "evalagent_query_builder",
        QUERY_BUILDER_SCRIPT_PATH,
    )

    search_module = load_python_module(
        "academic_zvec_search",
        SEARCH_SCRIPT_PATH,
    )

    print("[2/6] Parsing requirement...")

    profile = parser_module.parse_requirement(
        requirement
    )

    parser_module.validate_profile(profile)

    budget = int(
        profile.get("evaluation_budget", 20)
    )

    candidate_k = (
        args.candidate_k
        if args.candidate_k is not None
        else max(budget * 3, 50)
    )

    candidate_k = max(
        budget,
        min(candidate_k, 500),
    )

    print(
        json.dumps(
            profile,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n[3/6] Building retrieval query...")

    retrieval_query = (
        query_builder_module.build_retrieval_query(
            profile
        )
    )

    print(retrieval_query)

    image_path: Path | None = None

    if args.image:
        image_path = Path(
            args.image
        ).expanduser().resolve()

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Query image not found: {image_path}"
            )

    print("\n[4/6] Generating query embedding...")

    query_vector = (
        search_module.call_query_embedding_api(
            api_key=api_key,
            text=retrieval_query,
            image_path=image_path,
        )
    )

    print(
        f"Embedding dimension: "
        f"{query_vector.shape[0]}"
    )

    collection_path = (
        search_module.COLLECTION_PATH
    )

    if not collection_path.exists():
        raise FileNotFoundError(
            f"Zvec Collection not found: "
            f"{collection_path}"
        )

    print("\n[5/6] Searching Zvec candidates...")

    collection = zvec.open(
        path=str(collection_path),
        option=zvec.CollectionOption(
            read_only=True,
            enable_mmap=True,
        ),
    )

    query = zvec.Query(
        field_name="embedding",
        vector=query_vector.tolist(),
    )

    raw_results = collection.query(
        queries=query,
        topk=candidate_k,
    )

    print(
        f"Retrieved {len(raw_results)} candidates."
    )

    candidates = [
        convert_result(
            rank=rank,
            result=result,
            profile=profile,
            search_module=search_module,
        )
        for rank, result in enumerate(
            raw_results,
            start=1,
        )
    ]

    selected_samples = rerank_candidates(
        candidates=candidates,
        budget=budget,
    )

    print("\n[6/6] Aggregating and saving results...")

    dataset_summary = aggregate_dataset_summary(
        selected_samples
    )

    coverage_summary = build_coverage_summary(
        selected_samples
    )

    recommendation_summary = (
        build_recommendation_summary(
            samples=selected_samples,
            dataset_summary=dataset_summary,
            profile=profile,
        )
    )

    result = {
        "evalagent_version": "phase1_v1",
        "request_id": request_id,
        "raw_requirement": requirement,
        "query_profile": profile,
        "retrieval_query": retrieval_query,
        "retrieval": {
            "embedding_model": search_module.MODEL_NAME,
            "embedding_dimension": (
                search_module.EMBEDDING_DIMENSION
            ),
            "candidate_k": candidate_k,
            "returned_candidates": len(raw_results),
            "selected_samples": len(
                selected_samples
            ),
            "query_image": (
                str(image_path)
                if image_path
                else None
            ),
        },
        "recommended_datasets": dataset_summary,
        "coverage_summary": coverage_summary,
        "recommended_samples": selected_samples,
        "recommendation_summary": recommendation_summary,
    }

    if args.output:
        output_path = Path(args.output)

        if not output_path.is_absolute():
            output_path = (
                PROJECT_ROOT
                / output_path
            )
    else:
        output_path = (
            DEFAULT_OUTPUT_DIR
            / f"{request_id}_recommendation.json"
        )

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

    print("\n" + "=" * 80)
    print("EvalAgent completed")
    print("=" * 80)
    print(recommendation_summary)
    print(f"Output: {output_path}")

    print("\nDataset recommendation summary:")

    for dataset_name, info in dataset_summary.items():
        print(
            f"- {dataset_name}: "
            f"{info['count']} samples, "
            f"proportion={info['proportion']:.2%}, "
            f"avg_similarity="
            f"{info['average_similarity']:.4f}"
        )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nEvalAgent interrupted by user.")

    except Exception as error:
        print("\n" + "=" * 80)
        print("EvalAgent failed")
        print("=" * 80)
        print(
            f"Error type: {type(error).__name__}"
        )
        print(f"Error message: {error}")
        raise