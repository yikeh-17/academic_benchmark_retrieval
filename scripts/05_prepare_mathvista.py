import json
from pathlib import Path


PROJECT_ROOT = Path.home() / "academic_benchmark_retrieval"

INPUT_JSONL = (
    PROJECT_ROOT
    / "data/raw/mathvista/mathvista_testmini_sample_500.jsonl"
)

OUTPUT_JSONL = (
    PROJECT_ROOT
    / "data/processed/mathvista_sample_index.jsonl"
)


def build_question_text(item):
    question = str(item.get("question", "")).strip()
    choices = item.get("choices")

    if choices and isinstance(choices, list):
        valid_choices = [
            str(choice).strip()
            for choice in choices
            if choice is not None and str(choice).strip()
        ]

        if valid_choices:
            choice_text = "\n".join(
                f"{chr(65 + i)}. {choice}"
                for i, choice in enumerate(valid_choices)
            )
            return f"{question}\nChoices:\n{choice_text}"

    return question


def find_image_path(item):
    possible_fields = [
        "image_path",
        "saved_image_path",
        "local_image_path",
        "image",
    ]

    for field in possible_fields:
        value = item.get(field)

        if not isinstance(value, str) or not value.strip():
            continue

        image_path = Path(value)

        if image_path.is_absolute():
            return image_path

        project_relative = PROJECT_ROOT / image_path
        if project_relative.exists():
            return project_relative

        raw_relative = PROJECT_ROOT / "data/raw/mathvista" / image_path
        if raw_relative.exists():
            return raw_relative

    pid = str(item.get("pid", "")).strip()

    if pid:
        image_dir = PROJECT_ROOT / "data/raw/mathvista/images"

        for extension in [".png", ".jpg", ".jpeg", ".webp"]:
            candidate = image_dir / f"{pid}{extension}"
            if candidate.exists():
                return candidate

    return None


def make_relative_path(path):
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main():
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    saved = 0
    missing_image = 0
    empty_question = 0

    with INPUT_JSONL.open("r", encoding="utf-8") as fin, \
            OUTPUT_JSONL.open("w", encoding="utf-8") as fout:

        for line_number, line in enumerate(fin, start=1):
            line = line.strip()

            if not line:
                continue

            total += 1
            item = json.loads(line)

            question_text = build_question_text(item)

            if not question_text:
                empty_question += 1
                continue

            image_path = find_image_path(item)

            if image_path is None or not image_path.exists():
                missing_image += 1
                print(
                    f"Missing image at line {line_number}, "
                    f"pid={item.get('pid')}"
                )
                continue

            pid = str(item.get("pid", saved))

            answer = item.get("answer")
            answer_text = "" if answer is None else str(answer).strip()

            metadata = item.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"original_metadata": metadata}

            output_item = {
                "id": f"mathvista_{pid}",
                "dataset": "MathVista",
                "split": "testmini",
                "image_path": make_relative_path(image_path),
                "question": question_text,
                "answer": answer_text,
                "question_type": item.get("question_type"),
                "answer_type": item.get("answer_type"),
                "text_for_embedding": (
                    "Dataset: MathVista\n"
                    f"Question: {question_text}"
                ),
                "metadata": {
                    **metadata,
                    "pid": item.get("pid"),
                    "unit": item.get("unit"),
                    "precision": item.get("precision"),
                },
            }

            fout.write(
                json.dumps(output_item, ensure_ascii=False) + "\n"
            )
            saved += 1

    print("Input records:", total)
    print("Saved records:", saved)
    print("Missing images:", missing_image)
    print("Empty questions:", empty_question)
    print("Output:", OUTPUT_JSONL)


if __name__ == "__main__":
    main()
