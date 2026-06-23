import json
from pathlib import Path
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data/raw/chartqa/chartqa_sample_500.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/chartqa_sample_index.jsonl"
)

CHARTQA_IMAGE_ROOT = (
    PROJECT_ROOT
    / "data/raw/chartqa/data/chartqa_5000"
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    return " ".join(str(value).strip().split())


def get_message(
    conversations: list[dict],
    target_role: str,
) -> str:
    for message in conversations:
        if message.get("role") == target_role:
            return clean_text(message.get("content"))

    return ""


def clean_question(user_content: str) -> str:
    return clean_text(
        user_content.replace("<image>", "")
    )


def resolve_image_path(
    image_value: str,
) -> Path | None:
    if not image_value:
        return None

    image_path = CHARTQA_IMAGE_ROOT / image_value

    if image_path.exists():
        return image_path.resolve()

    return None


def infer_answer_type(answer: str) -> str:
    normalized = answer.strip().lower()

    if normalized in {"yes", "no"}:
        return "yes_no"

    try:
        float(
            normalized
            .replace(",", "")
            .replace("%", "")
        )
        return "numeric"
    except ValueError:
        pass

    if len(normalized.split()) <= 5:
        return "short_answer"

    return "open_ended"


def infer_abilities(
    question: str,
    answer_type: str,
) -> list[str]:
    combined = question.lower()

    abilities = [
        "chart_understanding",
        "visual_question_answering",
    ]

    numeric_keywords = [
        "value",
        "how many",
        "how much",
        "difference",
        "sum",
        "total",
        "average",
        "percent",
        "percentage",
        "ratio",
        "greater",
        "less",
        "highest",
        "lowest",
        "maximum",
        "minimum",
    ]

    comparison_keywords = [
        "more than",
        "less than",
        "higher",
        "lower",
        "greater",
        "smaller",
        "largest",
        "smallest",
    ]

    trend_keywords = [
        "increase",
        "decrease",
        "trend",
        "change",
        "over time",
        "from",
        "between",
    ]

    if (
        answer_type == "numeric"
        or any(
            keyword in combined
            for keyword in numeric_keywords
        )
    ):
        abilities.extend(
            [
                "numerical_reasoning",
                "data_extraction",
            ]
        )

    if any(
        keyword in combined
        for keyword in comparison_keywords
    ):
        abilities.append("numerical_comparison")

    if any(
        keyword in combined
        for keyword in trend_keywords
    ):
        abilities.append("trend_analysis")

    if answer_type == "yes_no":
        abilities.append("chart_fact_verification")

    return list(dict.fromkeys(abilities))


def infer_task_type(
    question: str,
    answer_type: str,
) -> str:
    combined = question.lower()

    if answer_type == "yes_no":
        return "chart_fact_verification"

    if any(
        keyword in combined
        for keyword in [
            "highest",
            "lowest",
            "maximum",
            "minimum",
            "greater",
            "less",
            "difference",
            "compare",
        ]
    ):
        return "chart_comparison"

    if any(
        keyword in combined
        for keyword in [
            "increase",
            "decrease",
            "trend",
            "change",
            "over time",
        ]
    ):
        return "chart_trend_analysis"

    if answer_type == "numeric":
        return "chart_numeric_qa"

    return "chart_qa"


def build_text_for_embedding(
    question: str,
    answer_type: str,
    task_type: str,
    abilities: list[str],
) -> str:
    return " ".join(
        [
            "Dataset: ChartQA.",
            f"Task: {task_type}.",
            "Domain: chart reasoning.",
            "Image type: chart.",
            f"Abilities: {', '.join(abilities)}.",
            f"Question: {question}",
            f"Answer type: {answer_type}.",
        ]
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise TypeError(
            f"Expected input to be a list, got {type(records)}"
        )

    kept = 0
    missing_image = 0
    missing_question = 0
    missing_answer = 0
    duplicate_ids = 0

    seen_ids: set[str] = set()

    task_counts: dict[str, int] = {}
    answer_type_counts: dict[str, int] = {}

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for item in tqdm(
            records,
            desc="Preparing ChartQA",
        ):
            sample_id = clean_text(
                item.get("id")
            )

            if not sample_id:
                sample_id = f"chartqa_{kept:06d}"

            if sample_id in seen_ids:
                duplicate_ids += 1
                continue

            conversations = (
                item.get("conversations") or []
            )

            user_content = get_message(
                conversations,
                "user",
            )

            assistant_content = get_message(
                conversations,
                "assistant",
            )

            question = clean_question(
                user_content
            )

            answer = clean_text(
                assistant_content
            )

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

            answer_type = infer_answer_type(
                answer
            )

            task_type = infer_task_type(
                question,
                answer_type,
            )

            abilities = infer_abilities(
                question,
                answer_type,
            )

            text_for_embedding = (
                build_text_for_embedding(
                    question=question,
                    answer_type=answer_type,
                    task_type=task_type,
                    abilities=abilities,
                )
            )

            output_item = {
                "id": sample_id,
                "dataset_name": "ChartQA",
                "image": str(image_path),
                "question": question,
                "answer": answer,
                "reasoning": "",
                "choices": [],
                "modality": "image+text",
                "task_type": task_type,
                "image_type": "chart",
                "domain": "chart_reasoning",
                "ability": abilities,
                "difficulty": "unknown",
                "answer_type": answer_type,
                "source_split": "unknown",
                "text_for_embedding": (
                    text_for_embedding
                ),
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
                task_counts.get(
                    task_type,
                    0,
                )
                + 1
            )

            answer_type_counts[
                answer_type
            ] = (
                answer_type_counts.get(
                    answer_type,
                    0,
                )
                + 1
            )

    print("\n" + "=" * 60)
    print(
        "ChartQA sample-level preparation finished"
    )
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

    print("\nAnswer type counts:")
    for key, value in sorted(
        answer_type_counts.items(),
        key=lambda pair: pair[1],
        reverse=True,
    ):
        print(f"  {key}: {value}")

    print("=" * 60)


if __name__ == "__main__":
    main()