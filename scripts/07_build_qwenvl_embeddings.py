import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data/processed/academic_5_merged.jsonl"
)

VECTOR_DIR = PROJECT_ROOT / "data/vector_db"

RAW_EMBEDDINGS_PATH = (
    VECTOR_DIR
    / "academic_5_embeddings_raw.jsonl"
)

FINAL_EMBEDDINGS_PATH = (
    VECTOR_DIR
    / "academic_5_embeddings.npy"
)

METADATA_PATH = (
    VECTOR_DIR
    / "academic_5_metadata.jsonl"
)

FAILED_PATH = (
    VECTOR_DIR
    / "academic_5_embedding_failed.jsonl"
)

API_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/"
    "embeddings/multimodal-embedding/multimodal-embedding"
)

MODEL_NAME = "qwen3-vl-embedding"
EMBEDDING_DIMENSION = 1024

MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 5
REQUEST_TIMEOUT = 180


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build multimodal embeddings for the merged "
            "academic benchmark dataset."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N pending records.",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete previous embedding outputs before running.",
    )

    return parser.parse_args()


def image_to_data_url(image_path: Path) -> str:
    """
    Convert a local image file into a Base64 data URL.
    """
    mime_type, _ = mimetypes.guess_type(image_path.name)

    if mime_type is None:
        mime_type = "image/png"

    with image_path.open("rb") as file:
        encoded = base64.b64encode(
            file.read()
        ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def load_records(
    path: Path,
) -> list[dict[str, Any]]:
    """
    Load all JSONL records.
    """
    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line "
                    f"{line_number}: {error}"
                ) from error

            records.append(item)

    return records


def load_completed_ids(
    path: Path,
) -> set[str]:
    """
    Read IDs that already have successful embeddings.
    """
    completed_ids: set[str] = set()

    if not path.exists():
        return completed_ids

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"at line {line_number}: {error}"
                ) from error

            sample_id = item.get("id")

            if sample_id:
                completed_ids.add(sample_id)

    return completed_ids


def call_embedding_api(
    api_key: str,
    image_path: Path,
    text: str,
) -> list[float]:
    """
    Call the DashScope multimodal embedding API.
    """
    payload = {
        "model": MODEL_NAME,
        "input": {
            "contents": [
                {
                    "text": text,
                },
                {
                    "image": image_to_data_url(
                        image_path
                    ),
                },
            ]
        },
        "parameters": {
            "enable_fusion": True,
            "dimension": EMBEDDING_DIMENSION,
        },
    }

    last_error: Exception | None = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": (
                        f"Bearer {api_key}"
                    ),
                    "Content-Type": (
                        "application/json"
                    ),
                },
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 429:
                wait_time = (
                    RETRY_WAIT_SECONDS
                    * attempt
                )

                print(
                    f"Rate limited. "
                    f"Waiting {wait_time}s..."
                )

                time.sleep(wait_time)
                continue

            if not response.ok:
                raise RuntimeError(
                    f"HTTP "
                    f"{response.status_code}: "
                    f"{response.text}"
                )

            try:
                result = response.json()
            except ValueError as error:
                raise RuntimeError(
                    "API returned a non-JSON "
                    f"response: {response.text}"
                ) from error

            embeddings = (
                result
                .get("output", {})
                .get("embeddings", [])
            )

            if not embeddings:
                raise RuntimeError(
                    "No embedding returned by API: "
                    f"{result}"
                )

            embedding = embeddings[0].get(
                "embedding"
            )

            if embedding is None:
                raise RuntimeError(
                    "Unexpected API response "
                    f"format: {result}"
                )

            if not isinstance(
                embedding,
                list,
            ):
                raise RuntimeError(
                    "Embedding is not a list: "
                    f"{type(embedding)}"
                )

            if len(embedding) != (
                EMBEDDING_DIMENSION
            ):
                raise RuntimeError(
                    f"Expected "
                    f"{EMBEDDING_DIMENSION} "
                    f"dimensions, got "
                    f"{len(embedding)}"
                )

            return embedding

        except Exception as error:
            last_error = error

            if attempt == MAX_RETRIES:
                break

            wait_time = (
                RETRY_WAIT_SECONDS
                * attempt
            )

            print(
                f"Attempt {attempt} failed: "
                f"{error}. "
                f"Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        f"API failed after "
        f"{MAX_RETRIES} attempts: "
        f"{last_error}"
    )


def build_final_files() -> None:
    """
    Convert raw JSONL embeddings into:
    1. NumPy embedding matrix
    2. Metadata JSONL
    """
    if not RAW_EMBEDDINGS_PATH.exists():
        print(
            "No raw embeddings file found."
        )
        return

    embeddings: list[list[float]] = []
    metadata_records: list[
        dict[str, Any]
    ] = []

    with RAW_EMBEDDINGS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Invalid JSON in raw "
                    f"embeddings at line "
                    f"{line_number}: {error}"
                ) from error

            embedding = item.get(
                "embedding"
            )

            if embedding is None:
                raise ValueError(
                    "Missing embedding at line "
                    f"{line_number}"
                )

            if len(embedding) != (
                EMBEDDING_DIMENSION
            ):
                raise ValueError(
                    f"Invalid embedding "
                    f"dimension at line "
                    f"{line_number}: "
                    f"{len(embedding)}"
                )

            embeddings.append(embedding)

            metadata_records.append(
                {
                    key: value
                    for key, value
                    in item.items()
                    if key != "embedding"
                }
            )

    if not embeddings:
        print(
            "No embeddings were generated."
        )
        return

    matrix = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if np.isnan(matrix).any():
        raise ValueError(
            "Embedding matrix contains NaN."
        )

    if np.isinf(matrix).any():
        raise ValueError(
            "Embedding matrix contains Inf."
        )

    np.save(
        FINAL_EMBEDDINGS_PATH,
        matrix,
    )

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for item in metadata_records:
            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(
        "Final embedding matrix:",
        matrix.shape,
    )

    print(
        "Saved:",
        FINAL_EMBEDDINGS_PATH,
    )

    print(
        "Saved:",
        METADATA_PATH,
    )


def reset_outputs() -> None:
    """
    Delete previous output files.
    """
    for path in [
        RAW_EMBEDDINGS_PATH,
        FINAL_EMBEDDINGS_PATH,
        METADATA_PATH,
        FAILED_PATH,
    ]:
        if path.exists():
            path.unlink()
            print("Deleted:", path)


def write_failed_record(
    failed_file: Any,
    sample_id: str,
    error: str,
    image: str | None = None,
) -> None:
    """
    Write one failed record to the failure log.
    """
    record: dict[str, Any] = {
        "id": sample_id,
        "error": error,
    }

    if image is not None:
        record["image"] = image

    failed_file.write(
        json.dumps(
            record,
            ensure_ascii=False,
        )
        + "\n"
    )

    failed_file.flush()


def main() -> None:
    args = parse_args()

    api_key = os.getenv(
        "DASHSCOPE_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY "
            "is not set."
        )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: "
            f"{INPUT_PATH}"
        )

    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.reset:
        reset_outputs()

    records = load_records(
        INPUT_PATH
    )

    completed_ids = load_completed_ids(
        RAW_EMBEDDINGS_PATH
    )

    pending_records = [
        item
        for item in records
        if item["id"]
        not in completed_ids
    ]

    if args.limit is not None:
        if args.limit < 0:
            raise ValueError(
                "--limit must be "
                "greater than or equal to 0."
            )

        pending_records = (
            pending_records[:args.limit]
        )

    print(
        "Total records:",
        len(records),
    )

    print(
        "Already completed:",
        len(completed_ids),
    )

    print(
        "This run:",
        len(pending_records),
    )

    with RAW_EMBEDDINGS_PATH.open(
        "a",
        encoding="utf-8",
    ) as embedding_file, \
         FAILED_PATH.open(
             "a",
             encoding="utf-8",
         ) as failed_file:

        for item in tqdm(
            pending_records,
            desc=(
                "Building multimodal "
                "embeddings"
            ),
        ):
            sample_id = item["id"]

            image_value = item.get(
                "image"
            )

            text = item.get(
                "text_for_embedding"
            )

            if not image_value:
                write_failed_record(
                    failed_file=failed_file,
                    sample_id=sample_id,
                    error=(
                        "missing_image_field"
                    ),
                )
                continue

            image_path = Path(
                image_value
            )

            if not image_path.is_file():
                write_failed_record(
                    failed_file=failed_file,
                    sample_id=sample_id,
                    error=(
                        "image_path_is_not_a_file"
                    ),
                    image=str(image_path),
                )
                continue

            if not text:
                write_failed_record(
                    failed_file=failed_file,
                    sample_id=sample_id,
                    error=(
                        "missing_text_for_embedding"
                    ),
                    image=str(image_path),
                )
                continue

            try:
                embedding = (
                    call_embedding_api(
                        api_key=api_key,
                        image_path=image_path,
                        text=text,
                    )
                )

                output_item = {
                    "id": sample_id,
                    "dataset_name": (
                        item.get(
                            "dataset_name"
                        )
                        or item.get(
                            "dataset"
                        )
                    ),
                    "image": str(
                        image_path
                    ),
                    "question": item.get(
                        "question"
                    ),
                    "task_type": item.get(
                        "task_type"
                    ),
                    "domain": item.get(
                        "domain"
                    ),
                    "ability": item.get(
                        "ability",
                        [],
                    ),
                    "embedding": embedding,
                }

                embedding_file.write(
                    json.dumps(
                        output_item,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                embedding_file.flush()

                print(
                    f"\nSuccess: "
                    f"id={sample_id}, "
                    f"dimension="
                    f"{len(embedding)}"
                )

                print(
                    "First 10 values:",
                    embedding[:10],
                )

            except Exception as error:
                print(
                    f"\nFailed: "
                    f"{sample_id}: "
                    f"{error}"
                )

                write_failed_record(
                    failed_file=failed_file,
                    sample_id=sample_id,
                    error=str(error),
                    image=str(image_path),
                )

    build_final_files()


if __name__ == "__main__":
    main()