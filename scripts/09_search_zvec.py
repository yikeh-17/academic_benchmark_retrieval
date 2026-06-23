"""
09_search_zvec.py

作用：
1. 接收用户的文本查询，可选查询图片
2. 使用 qwen3-vl-embedding 生成 1024 维查询向量
3. 打开 Academic-5 Zvec Collection
4. 检索最相似的 Top-K 评测样本
5. 输出样本信息和数据集命中统计

示例：

纯文本查询：
python scripts/09_search_zvec.py \
    --query "scientific plots, oscillation counting and trend analysis" \
    --topk 10

文本 + 图片查询：
python scripts/09_search_zvec.py \
    --query "find benchmarks requiring understanding of this plot" \
    --image /path/to/query_image.png \
    --topk 10
"""

import argparse
import base64
import mimetypes
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import requests
import zvec


# ============================================================
# 1. 路径和 API 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COLLECTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "vector_db"
    / "academic_5_zvec"
    / "academic_collection"
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


# ============================================================
# 2. 命令行参数
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a query embedding and search "
            "the Academic-5 Zvec database."
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Text describing the benchmark or capability needed.",
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional local image path for multimodal search.",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of nearest samples to return.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Optional dataset filter, such as "
            "QCalEval, ScienceQA, ChartQA, PlotQA or MathVista."
        ),
    )

    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Optional domain filter.",
    )

    parser.add_argument(
        "--task-type",
        type=str,
        default=None,
        help="Optional task_type filter.",
    )

    return parser.parse_args()


# ============================================================
# 3. 图片处理
# ============================================================

def image_to_data_url(image_path: Path) -> str:
    """
    Convert a local image into a Base64 data URL.
    """
    mime_type, _ = mimetypes.guess_type(image_path.name)

    if mime_type is None:
        mime_type = "image/png"

    with image_path.open("rb") as file:
        encoded = base64.b64encode(
            file.read()
        ).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


# ============================================================
# 4. 生成查询向量
# ============================================================

def call_query_embedding_api(
    api_key: str,
    text: str,
    image_path: Path | None = None,
) -> np.ndarray:
    """
    Generate a 1024-dimensional query embedding.

    The model, dimension and fusion settings are kept
    consistent with 07_build_qwenvl_embeddings.py.
    """

    contents: list[dict[str, str]] = [
        {
            "text": text,
        }
    ]

    if image_path is not None:
        contents.append(
            {
                "image": image_to_data_url(image_path),
            }
        )

    payload = {
        "model": MODEL_NAME,
        "input": {
            "contents": contents,
        },
        "parameters": {
            "enable_fusion": True,
            "dimension": EMBEDDING_DIMENSION,
        },
    }

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 429:
                wait_time = RETRY_WAIT_SECONDS * attempt

                print(
                    f"API rate limited. Waiting {wait_time}s..."
                )

                time.sleep(wait_time)
                continue

            if not response.ok:
                raise RuntimeError(
                    f"HTTP {response.status_code}: "
                    f"{response.text}"
                )

            try:
                result = response.json()
            except ValueError as error:
                raise RuntimeError(
                    "API returned a non-JSON response: "
                    f"{response.text}"
                ) from error

            embeddings = (
                result
                .get("output", {})
                .get("embeddings", [])
            )

            if not embeddings:
                raise RuntimeError(
                    f"No embedding returned by API: {result}"
                )

            embedding = embeddings[0].get("embedding")

            if embedding is None:
                raise RuntimeError(
                    f"Unexpected API response format: {result}"
                )

            if not isinstance(embedding, list):
                raise RuntimeError(
                    "Embedding is not a list: "
                    f"{type(embedding)}"
                )

            if len(embedding) != EMBEDDING_DIMENSION:
                raise RuntimeError(
                    f"Expected {EMBEDDING_DIMENSION} dimensions, "
                    f"got {len(embedding)}"
                )

            query_vector = np.asarray(
                embedding,
                dtype=np.float32,
            )

            if not np.isfinite(query_vector).all():
                raise RuntimeError(
                    "Query embedding contains NaN or Inf."
                )

            if np.linalg.norm(query_vector) == 0:
                raise RuntimeError(
                    "Query embedding is an all-zero vector."
                )

            return query_vector

        except Exception as error:
            last_error = error

            if attempt == MAX_RETRIES:
                break

            wait_time = RETRY_WAIT_SECONDS * attempt

            print(
                f"Embedding attempt {attempt} failed: {error}"
            )
            print(
                f"Retrying in {wait_time}s..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        f"Embedding API failed after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )


# ============================================================
# 5. 构造 Zvec 过滤条件
# ============================================================

def escape_filter_string(value: str) -> str:
    """
    Escape single quotes used in Zvec filter strings.
    """
    return value.replace("'", "''")


def build_filter(
    dataset_name: str | None,
    domain: str | None,
    task_type: str | None,
) -> str | None:
    """
    Build an optional scalar filter.
    """
    conditions: list[str] = []

    if dataset_name:
        safe_value = escape_filter_string(
            dataset_name.strip()
        )
        conditions.append(
            f"dataset_name = '{safe_value}'"
        )

    if domain:
        safe_value = escape_filter_string(
            domain.strip()
        )
        conditions.append(
            f"domain = '{safe_value}'"
        )

    if task_type:
        safe_value = escape_filter_string(
            task_type.strip()
        )
        conditions.append(
            f"task_type = '{safe_value}'"
        )

    if not conditions:
        return None

    return " AND ".join(conditions)


# ============================================================
# 6. 打印查询结果
# ============================================================

def safe_similarity(distance: float) -> float:
    """
    Convert cosine distance to an approximate similarity.

    For the current COSINE configuration:
        identical vector -> distance 0
        similarity ≈ 1 - distance
    """
    similarity = 1.0 - float(distance)

    return max(-1.0, min(1.0, similarity))


def print_result(
    rank: int,
    result: Any,
) -> None:
    fields = result.fields or {}

    distance = float(result.score)
    similarity = safe_similarity(distance)

    print("\n" + "=" * 80)
    print(f"Rank:       {rank}")
    print(f"Zvec ID:    {result.id}")
    print(f"Distance:   {distance:.6f}")
    print(f"Similarity: {similarity:.6f}")
    print(f"Dataset:    {fields.get('dataset_name', '')}")
    print(f"Original ID:{fields.get('original_id', '')}")
    print(f"Task type:  {fields.get('task_type', '')}")
    print(f"Domain:     {fields.get('domain', '')}")
    print(f"Difficulty: {fields.get('difficulty', '')}")
    print(f"Ability:    {fields.get('ability', '')}")
    print(f"Question:   {fields.get('question', '')}")
    print(f"Image:      {fields.get('image_path', '')}")


def print_dataset_summary(
    results: list[Any],
) -> None:
    dataset_counts = Counter()

    dataset_similarities: dict[
        str,
        list[float],
    ] = {}

    for result in results:
        fields = result.fields or {}

        dataset_name = fields.get(
            "dataset_name",
            "Unknown",
        )

        similarity = safe_similarity(
            float(result.score)
        )

        dataset_counts[dataset_name] += 1

        dataset_similarities.setdefault(
            dataset_name,
            [],
        ).append(similarity)

    print("\n" + "=" * 80)
    print("Dataset hit summary")
    print("=" * 80)

    summary_rows = []

    for dataset_name, count in dataset_counts.items():
        similarities = dataset_similarities[
            dataset_name
        ]

        average_similarity = (
            sum(similarities)
            / len(similarities)
        )

        best_similarity = max(similarities)

        summary_rows.append(
            (
                dataset_name,
                count,
                average_similarity,
                best_similarity,
            )
        )

    summary_rows.sort(
        key=lambda item: (
            item[1],
            item[2],
        ),
        reverse=True,
    )

    for index, row in enumerate(
        summary_rows,
        start=1,
    ):
        (
            dataset_name,
            count,
            average_similarity,
            best_similarity,
        ) = row

        print(
            f"{index}. {dataset_name:<15} "
            f"hits={count:<3} "
            f"avg_similarity={average_similarity:.4f} "
            f"best_similarity={best_similarity:.4f}"
        )


# ============================================================
# 7. 主程序
# ============================================================

def main() -> None:
    args = parse_args()

    if args.topk <= 0:
        raise ValueError(
            "--topk must be greater than 0."
        )

    query_text = args.query.strip()

    if not query_text:
        raise ValueError(
            "--query cannot be empty."
        )

    api_key = os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set."
        )

    if not COLLECTION_PATH.exists():
        raise FileNotFoundError(
            f"Zvec Collection not found: "
            f"{COLLECTION_PATH}"
        )

    image_path: Path | None = None

    if args.image:
        image_path = Path(
            args.image
        ).expanduser().resolve()

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Query image not found: "
                f"{image_path}"
            )

    print("=" * 80)
    print("Academic-5 Zvec Search")
    print("=" * 80)
    print(f"Query: {query_text}")
    print(
        f"Image: "
        f"{image_path if image_path else 'None'}"
    )
    print(f"Top-K: {args.topk}")

    filter_string = build_filter(
        dataset_name=args.dataset,
        domain=args.domain,
        task_type=args.task_type,
    )

    print(
        f"Filter: "
        f"{filter_string if filter_string else 'None'}"
    )

    print("\n[1/3] Generating query embedding...")

    query_vector = call_query_embedding_api(
        api_key=api_key,
        text=query_text,
        image_path=image_path,
    )

    print(
        f"Query embedding dimension: "
        f"{query_vector.shape[0]}"
    )

    print("\n[2/3] Opening Zvec Collection...")

    collection = zvec.open(
        path=str(COLLECTION_PATH),
        option=zvec.CollectionOption(
            read_only=True,
            enable_mmap=True,
        ),
    )

    print("Collection opened successfully.")

    print("\n[3/3] Searching...")

    query = zvec.Query(
        field_name="embedding",
        vector=query_vector.tolist(),
    )

    query_kwargs: dict[str, Any] = {
        "queries": query,
        "topk": args.topk,
    }

    if filter_string:
        query_kwargs["filter"] = filter_string

    results = collection.query(
        **query_kwargs
    )

    print(
        f"\nReturned {len(results)} results."
    )

    if not results:
        print(
            "No matching records were found."
        )
        return

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print_result(
            rank=rank,
            result=result,
        )

    print_dataset_summary(results)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")

    except Exception as error:
        print("\n" + "=" * 80)
        print("Search failed")
        print("=" * 80)
        print(
            f"Error type: {type(error).__name__}"
        )
        print(
            f"Error message: {error}"
        )
        raise