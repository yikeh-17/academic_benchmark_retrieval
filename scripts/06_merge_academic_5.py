import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path.home() / "academic_benchmark_retrieval"

INPUT_FILES = [
    PROJECT_ROOT / "data/processed/qcaleval_sample_index.jsonl",
    PROJECT_ROOT / "data/processed/scienceqa_sample_index.jsonl",
    PROJECT_ROOT / "data/processed/chartqa_sample_index.jsonl",
    PROJECT_ROOT / "data/processed/plotqa_sample_index.jsonl",
    PROJECT_ROOT / "data/processed/mathvista_sample_index.jsonl",
]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data/processed/academic_5_merged.jsonl"
)


# 当原始记录没有 dataset_name 时，
# 根据输入文件名补充规范的数据集名称
DATASET_NAME_BY_FILE = {
    "qcaleval_sample_index.jsonl": "QCalEval",
    "scienceqa_sample_index.jsonl": "ScienceQA",
    "chartqa_sample_index.jsonl": "ChartQA",
    "plotqa_sample_index.jsonl": "PlotQA",
    "mathvista_sample_index.jsonl": "MathVista",
}


def get_dataset_name(item: dict, input_file: Path) -> str:
    """
    优先使用记录本身的 dataset_name；
    其次使用已有 dataset；
    最后根据输入文件名推断。
    """
    dataset_name = item.get("dataset_name")

    if isinstance(dataset_name, str) and dataset_name.strip():
        return dataset_name.strip()

    dataset_value = item.get("dataset")

    if isinstance(dataset_value, str) and dataset_value.strip():
        dataset_value = dataset_value.strip()

        # 避免使用 qcaleval_sample_index 这类文件名作为正式数据集名
        if not dataset_value.endswith("_sample_index"):
            return dataset_value

    return DATASET_NAME_BY_FILE.get(
        input_file.name,
        input_file.stem,
    )


def validate_item(
    item: dict,
    input_file: Path,
    line_number: int,
) -> None:
    """
    检查后续 embedding 所需的关键字段。
    """
    sample_id = item.get("id")

    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(
            f"Missing or invalid id: "
            f"{input_file}, line {line_number}"
        )

    text_for_embedding = item.get("text_for_embedding")

    if (
        not isinstance(text_for_embedding, str)
        or not text_for_embedding.strip()
    ):
        raise ValueError(
            f"Missing or empty text_for_embedding: "
            f"{input_file}, line {line_number}, id={sample_id}"
        )

    image = item.get("image")

    if not isinstance(image, str) or not image.strip():
        raise ValueError(
            f"Missing or invalid image path: "
            f"{input_file}, line {line_number}, id={sample_id}"
        )


def main() -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    seen_ids = set()

    total_written = 0
    duplicate_ids = 0
    missing_images = 0

    dataset_counts = Counter()
    input_file_counts = Counter()

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as fout:

        for input_file in INPUT_FILES:
            if not input_file.exists():
                raise FileNotFoundError(
                    f"Input file not found: {input_file}"
                )

            current_file_written = 0

            with input_file.open(
                "r",
                encoding="utf-8",
            ) as fin:

                for line_number, line in enumerate(
                    fin,
                    start=1,
                ):
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON: "
                            f"{input_file}, line {line_number}: {exc}"
                        ) from exc

                    if not isinstance(item, dict):
                        raise ValueError(
                            f"Record is not a JSON object: "
                            f"{input_file}, line {line_number}"
                        )

                    validate_item(
                        item=item,
                        input_file=input_file,
                        line_number=line_number,
                    )

                    sample_id = item["id"].strip()

                    if sample_id in seen_ids:
                        duplicate_ids += 1
                        print(
                            f"Warning: duplicate id skipped: "
                            f"{sample_id}"
                        )
                        continue

                    seen_ids.add(sample_id)

                    dataset_name = get_dataset_name(
                        item=item,
                        input_file=input_file,
                    )

                    # 统一两个字段，避免后续聚合时名称不一致
                    item["dataset"] = dataset_name
                    item["dataset_name"] = dataset_name

                    # 清理关键字符串字段首尾空格
                    item["id"] = sample_id
                    item["image"] = item["image"].strip()
                    item["text_for_embedding"] = (
                        item["text_for_embedding"].strip()
                    )

                    # 检查图片是否真实存在，但不直接跳过
                    image_path = Path(item["image"])

                    if not image_path.is_absolute():
                        image_path = PROJECT_ROOT / image_path

                    if not image_path.exists():
                        missing_images += 1
                        print(
                            f"Warning: image not found: "
                            f"id={sample_id}, path={image_path}"
                        )

                    fout.write(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    total_written += 1
                    current_file_written += 1

                    dataset_counts[dataset_name] += 1
                    input_file_counts[input_file.name] += 1

            print(
                f"Merged {current_file_written} records "
                f"from {input_file.name}"
            )

    print("\nMerge completed.")
    print("Output file:", OUTPUT_FILE)
    print("Total records written:", total_written)
    print("Duplicate IDs skipped:", duplicate_ids)
    print("Missing image files:", missing_images)

    print("\nDataset counts:")
    for dataset_name, count in sorted(
        dataset_counts.items()
    ):
        print(f"  {dataset_name}: {count}")

    print("\nInput file counts:")
    for file_name, count in input_file_counts.items():
        print(f"  {file_name}: {count}")


if __name__ == "__main__":
    main()