import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data/raw/scienceqa/scienceqa_all_image_sample_500.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/scienceqa_sample_index.jsonl"
)

SCIENCEQA_IMAGE_ROOT = Path(
    "/home/tju/scienceqa_demo/data/raw"
)


def clean_text(value: Any) -> str:
    """Convert a value to a clean single-line string."""
    if value is None:
        return ""

    return " ".join(str(value).strip().split())


def get_message(
    conversations: list[dict],
    target_role: str,
) -> str:
    """Extract the content corresponding to a conversation role."""
    for message in conversations:
        role = message.get("role", "")
        if role == target_role:
            return clean_text(message.get("content"))

    return ""


def parse_question_and_choices(
    user_content: str,
) -> tuple[str, list[str]]:
    """
    Parse ScienceQA user content.

    Example input:
        <image>
        Question: Which state is farthest north?
        Options:
        A. West Virginia
        B. Louisiana
        C. Arizona
        D. Oklahoma
    """
    text = user_content.replace("<image>", "").strip()

    question_match = re.search(
        r"Question:\s*(.*?)(?:\s*Options:|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if question_match:
        question = clean_text(question_match.group(1))
    else:
        question = clean_text(text)

    choices: list[str] = []

    options_match = re.search(
        r"Options:\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if options_match:
        options_text = options_match.group(1)

        option_matches = re.findall(
            r"(?:^|\s)([A-Z])\.\s*(.*?)(?=\s+[A-Z]\.\s*|\Z)",
            options_text,
            flags=re.DOTALL,
        )

        choices = [
            clean_text(option_text)
            for _, option_text in option_matches
        ]

    return question, choices


def parse_answer(
    assistant_content: str,
    choices: list[str],
) -> tuple[str, int | None, str]:
    """
    Convert an answer such as 'A. West Virginia' into:
        answer_text = 'West Virginia'
        answer_index = 0
        answer_label = 'A'
    """
    raw_answer = clean_text(assistant_content)

    match = re.match(
        r"^\s*([A-Z])[\.\):\-]?\s*(.*)$",
        raw_answer,
    )

    if not match:
        return raw_answer, None, ""

    answer_label = match.group(1).upper()
    answer_index = ord(answer_label) - ord("A")
    answer_text = clean_text(match.group(2))

    if choices and 0 <= answer_index < len(choices):
        answer_text = choices[answer_index]

    return answer_text, answer_index, answer_label


def resolve_image_path(image_value: str) -> Path | None:
    """
    Original image field:
        images/scienceqa_000000.png

    Actual image:
        /home/tju/scienceqa_demo/data/raw/images/scienceqa_000000.png
    """
    if not image_value:
        return None

    image_path = SCIENCEQA_IMAGE_ROOT / image_value

    if image_path.exists():
        return image_path.resolve()

    return None


def infer_difficulty(grade: str) -> str:
    """
    Simple prototype-level difficulty mapping based on grade.
    """
    match = re.search(r"\d+", grade)

    if not match:
        return "unknown"

    grade_number = int(match.group())

    if grade_number <= 3:
        return "easy"

    if grade_number <= 6:
        return "medium"

    return "hard"


def infer_image_type(
    subject: str,
    topic: str,
    skill: str,
) -> str:
    combined = f"{subject} {topic} {skill}".lower()

    if "map" in combined or "geography" in combined:
        return "map"

    if "graph" in combined or "chart" in combined:
        return "chart"

    if "geometry" in combined or "shape" in combined:
        return "geometry_diagram"

    if "diagram" in combined:
        return "scientific_diagram"

    return "educational_image"


def infer_abilities(
    subject: str,
    topic: str,
    skill: str,
) -> list[str]:
    combined = f"{subject} {topic} {skill}".lower()

    abilities = [
        "science_knowledge",
        "visual_reasoning",
        "question_answering",
    ]

    if "map" in combined or "geography" in combined:
        abilities.extend(
            [
                "map_understanding",
                "spatial_reasoning",
            ]
        )

    if "diagram" in combined:
        abilities.append("diagram_understanding")

    if "graph" in combined or "chart" in combined:
        abilities.extend(
            [
                "chart_understanding",
                "data_interpretation",
            ]
        )

    if "math" in combined:
        abilities.append("mathematical_reasoning")

    return list(dict.fromkeys(abilities))


def build_text_for_embedding(
    question: str,
    choices: list[str],
    lecture: str,
    reasoning: str,
    grade: str,
    subject: str,
    topic: str,
    category: str,
    skill: str,
    image_type: str,
    abilities: list[str],
    difficulty: str,
) -> str:
    """
    Construct the textual portion used together with the image
    for multimodal embedding.
    """
    parts = [
        "Dataset: ScienceQA.",
        "Task: science visual question answering.",
        "Domain: science education.",
        f"Subject: {subject}.",
        f"Topic: {topic}.",
        f"Category: {category}.",
        f"Skill: {skill}.",
        f"Grade: {grade}.",
        f"Image type: {image_type}.",
        f"Abilities: {', '.join(abilities)}.",
        f"Question: {question}",
    ]

    if choices:
        formatted_choices = "; ".join(
            f"{chr(65 + index)}. {choice}"
            for index, choice in enumerate(choices)
        )
        parts.append(f"Options: {formatted_choices}")

    if lecture:
        parts.append(f"Knowledge context: {lecture}")

    if reasoning:
        parts.append(f"Reasoning context: {reasoning}")

    parts.extend(
        [
            "Answer type: multiple choice.",
            f"Difficulty: {difficulty}.",
        ]
    )

    return " ".join(parts)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {INPUT_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise TypeError(
            f"Expected input data to be a list, got {type(records)}"
        )

    print(f"Loaded records: {len(records)}")

    kept = 0
    missing_image = 0
    missing_question = 0
    missing_answer = 0
    duplicate_ids = 0

    seen_ids: set[str] = set()

    subject_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    image_type_counts: dict[str, int] = {}

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for item in tqdm(
            records,
            desc="Preparing ScienceQA",
        ):
            sample_id = clean_text(item.get("id"))

            if not sample_id:
                sample_id = f"scienceqa_{kept:06d}"

            if sample_id in seen_ids:
                duplicate_ids += 1
                continue

            conversations = item.get("conversations") or []

            user_content = get_message(
                conversations,
                "user",
            )

            assistant_content = get_message(
                conversations,
                "assistant",
            )

            question, choices = parse_question_and_choices(
                user_content
            )

            answer, answer_index, answer_label = parse_answer(
                assistant_content,
                choices,
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

            metadata = item.get("metadata") or {}

            source_split = clean_text(
                metadata.get("source_split")
            ) or "unknown"

            source_index = metadata.get("source_index")

            grade = clean_text(
                metadata.get("grade")
            ) or "unknown"

            subject = clean_text(
                metadata.get("subject")
            ) or "science"

            topic = clean_text(
                metadata.get("topic")
            ) or "unknown"

            category = clean_text(
                metadata.get("category")
            ) or "unknown"

            skill = clean_text(
                metadata.get("skill")
            ) or "unknown"

            lecture = clean_text(
                metadata.get("lecture")
            )

            reasoning = clean_text(
                metadata.get("solution")
            )

            difficulty = infer_difficulty(grade)

            image_type = infer_image_type(
                subject=subject,
                topic=topic,
                skill=skill,
            )

            abilities = infer_abilities(
                subject=subject,
                topic=topic,
                skill=skill,
            )

            text_for_embedding = build_text_for_embedding(
                question=question,
                choices=choices,
                lecture=lecture,
                reasoning=reasoning,
                grade=grade,
                subject=subject,
                topic=topic,
                category=category,
                skill=skill,
                image_type=image_type,
                abilities=abilities,
                difficulty=difficulty,
            )

            output_item = {
                "id": sample_id,
                "dataset_name": "ScienceQA",
                "image": str(image_path),
                "question": question,
                "answer": answer,
                "reasoning": reasoning,
                "choices": choices,
                "answer_index": answer_index,
                "answer_label": answer_label,
                "modality": "image+text",
                "task_type": "science_visual_qa",
                "image_type": image_type,
                "domain": "science_education",
                "ability": abilities,
                "difficulty": difficulty,
                "answer_type": "multiple_choice",
                "source_split": source_split,
                "source_index": source_index,
                "grade": grade,
                "subject": subject,
                "topic": topic,
                "category": category,
                "skill": skill,
                "lecture": lecture,
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

            subject_counts[subject] = (
                subject_counts.get(subject, 0) + 1
            )

            difficulty_counts[difficulty] = (
                difficulty_counts.get(difficulty, 0) + 1
            )

            image_type_counts[image_type] = (
                image_type_counts.get(image_type, 0) + 1
            )

    print("\n" + "=" * 60)
    print("ScienceQA sample-level preparation finished")
    print("=" * 60)
    print(f"Input records:      {len(records)}")
    print(f"Kept records:       {kept}")
    print(f"Missing images:     {missing_image}")
    print(f"Missing questions:  {missing_question}")
    print(f"Missing answers:    {missing_answer}")
    print(f"Duplicate IDs:      {duplicate_ids}")
    print(f"Output file:        {OUTPUT_PATH}")

    print("\nSubject counts:")
    for key, value in sorted(
        subject_counts.items(),
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