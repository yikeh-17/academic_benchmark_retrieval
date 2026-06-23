import json
import random
from pathlib import Path
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data/raw/qcaleval/qcaleval_api_augmented_sample_500.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/qcaleval_sample_index.jsonl"
)

SOURCE_IMAGE_ROOT = Path(
    "/home/tju/qcaleval_project/data/images/qcaleval"
)

RANDOM_SEED = 42
TARGET_SIZE = 500


DEFAULT_ABILITIES = [
    "scientific_figure_understanding",
    "plot_understanding",
    "scientific_reasoning",
    "parameter_extraction",
    "experiment_diagnosis",
]


TASK_ABILITY_MAP = {
    "trend_analysis": [
        "trend_analysis",
        "plot_understanding",
        "scientific_reasoning",
    ],
    "figure_description": [
        "scientific_figure_understanding",
        "visual_description",
        "plot_understanding",
    ],
    "parameter_extraction": [
        "parameter_extraction",
        "data_extraction",
        "scientific_figure_understanding",
    ],
    "fit_assessment": [
        "fit_assessment",
        "scientific_reasoning",
        "plot_understanding",
    ],
    "experiment_status": [
        "experiment_diagnosis",
        "scientific_reasoning",
        "status_classification",
    ],
    "scientific_reasoning": [
        "scientific_reasoning",
        "experiment_understanding",
        "plot_understanding",
    ],
    "experiment_status_decision": [
        "experiment_diagnosis",
        "decision_making",
        "scientific_reasoning",
    ],
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    return " ".join(text.split())


def resolve_image_path(image_value: str) -> Path | None:
    if not image_value:
        return None

    filename = Path(image_value).name
    image_path = SOURCE_IMAGE_ROOT / filename

    if image_path.exists():
        return image_path

    return None


def get_abilities(task_type: str) -> list[str]:
    abilities = TASK_ABILITY_MAP.get(task_type)

    if abilities:
        return abilities

    return DEFAULT_ABILITIES


def build_text_for_embedding(
    question: str,
    reasoning: str,
    task_type: str,
    image_type: str,
    domain: str,
    difficulty: str,
    abilities: list[str],
    experiment_family: str,
    experiment_background: str,
) -> str:
    ability_text = ", ".join(abilities)

    parts = [
        "Dataset: QCalEval.",
        f"Task: {task_type}.",
        f"Domain: {domain}.",
        f"Image type: {image_type}.",
        f"Abilities: {ability_text}.",
    ]

    if experiment_family:
        parts.append(f"Experiment family: {experiment_family}.")

    if experiment_background:
        parts.append(
            f"Experiment background: {experiment_background}"
        )

    parts.append(f"Question: {question}")

    if reasoning:
        parts.append(f"Reasoning context: {reasoning}")

    parts.extend(
        [
            "Answer type: open-ended answer.",
            f"Difficulty: {difficulty}.",
        ]
    )

    return " ".join(parts)


def load_records(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                print(
                    f"Skipping invalid JSON at line "
                    f"{line_number}: {error}"
                )
                continue

            records.append(item)

    return records


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(INPUT_PATH)

    print(f"Loaded records: {len(records)}")

    if len(records) > TARGET_SIZE:
        random.seed(RANDOM_SEED)
        records = random.sample(records, TARGET_SIZE)

    kept = 0
    missing_image = 0
    missing_question = 0
    missing_answer = 0
    duplicate_ids = 0

    seen_ids: set[str] = set()

    task_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    image_type_counts: dict[str, int] = {}

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for item in tqdm(records, desc="Preparing QCalEval"):
            sample_id = clean_text(item.get("id"))

            if not sample_id:
                sample_id = f"qcaleval_{kept:06d}"

            if sample_id in seen_ids:
                duplicate_ids += 1
                continue

            question = clean_text(item.get("question"))
            answer = clean_text(item.get("answer"))
            reasoning = clean_text(item.get("reasoning"))

            if not question:
                missing_question += 1
                continue

            if not answer:
                missing_answer += 1
                continue

            image_path = resolve_image_path(
                clean_text(item.get("image"))
            )

            if image_path is None:
                missing_image += 1
                continue

            task_type = clean_text(
                item.get("task_type")
            ) or "scientific_reasoning"

            image_type = clean_text(
                item.get("image_type")
            ) or "scientific_figure"

            domain = clean_text(
                item.get("domain")
            ) or "quantum_calibration"

            difficulty = clean_text(
                item.get("difficulty")
            ) or "unknown"

            metadata = item.get("metadata") or {}

            experiment_family = clean_text(
                metadata.get("experiment_family")
            )

            experiment_background = clean_text(
                metadata.get("experiment_background")
            )

            original_id = clean_text(
                metadata.get("original_id")
            )

            original_qa_id = clean_text(
                metadata.get("original_qa_id")
            )

            question_index = metadata.get("question_index")

            abilities = get_abilities(task_type)

            text_for_embedding = build_text_for_embedding(
                question=question,
                reasoning=reasoning,
                task_type=task_type,
                image_type=image_type,
                domain=domain,
                difficulty=difficulty,
                abilities=abilities,
                experiment_family=experiment_family,
                experiment_background=experiment_background,
            )

            output_item = {
                "id": sample_id,
                "dataset_name": "QCalEval",
                "image": str(image_path),
                "question": question,
                "answer": answer,
                "reasoning": reasoning,
                "choices": [],
                "modality": "image+text",
                "task_type": task_type,
                "image_type": image_type,
                "domain": domain,
                "ability": abilities,
                "difficulty": difficulty,
                "answer_type": "open_ended",
                "source_split": "api_augmented",
                "is_science_related": bool(
                    item.get("is_science_related", True)
                ),
                "augment_type": clean_text(
                    item.get("augment_type")
                ),
                "original_id": original_id,
                "original_qa_id": original_qa_id,
                "question_index": question_index,
                "experiment_family": experiment_family,
                "experiment_background": experiment_background,
                "text_for_embedding": text_for_embedding,
            }

            output_file.write(
                json.dumps(
                    output_item,
                    ensure_ascii=False,
                )
                + "\n"
            )

            seen_ids.add(sample_id)
            kept += 1

            task_counts[task_type] = (
                task_counts.get(task_type, 0) + 1
            )
            difficulty_counts[difficulty] = (
                difficulty_counts.get(difficulty, 0) + 1
            )
            image_type_counts[image_type] = (
                image_type_counts.get(image_type, 0) + 1
            )

    print("\n" + "=" * 60)
    print("QCalEval sample-level preparation finished")
    print("=" * 60)
    print(f"Input records:      {len(records)}")
    print(f"Kept records:       {kept}")
    print(f"Missing images:     {missing_image}")
    print(f"Missing questions:  {missing_question}")
    print(f"Missing answers:    {missing_answer}")
    print(f"Duplicate IDs:      {duplicate_ids}")
    print(f"Output file:        {OUTPUT_PATH}")

    print("\nTask counts:")
    for key, value in sorted(
        task_counts.items(),
        key=lambda pair: pair[1],
        reverse=True,
    ):
        print(f"  {key}: {value}")

    print("\nDifficulty counts:")
    for key, value in sorted(
        difficulty_counts.items(),
        key=lambda pair: pair[1],
        reverse=True,
    ):
        print(f"  {key}: {value}")

    print("\nImage type counts:")
    for key, value in sorted(
        image_type_counts.items(),
        key=lambda pair: pair[1],
        reverse=True,
    ):
        print(f"  {key}: {value}")

    print("=" * 60)


if __name__ == "__main__":
    main()