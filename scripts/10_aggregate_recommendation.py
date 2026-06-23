"""
10_aggregate_recommendation.py

作用：
1. 接收用户的评测需求
2. 使用 09_search_zvec.py 中相同的方法生成查询向量
3. 从 Zvec 检索 Top-K 相似样本
4. 按 dataset_name 聚合
5. 输出最终评测集推荐排名
6. 可将结果保存为 JSON

运行示例：

python scripts/10_aggregate_recommendation.py \
    --query "scientific plot understanding, oscillation counting and trend analysis" \
    --topk 50 \
    --top-datasets 5

文本 + 图片：

python scripts/10_aggregate_recommendation.py \
    --query "find benchmarks that require understanding this scientific plot" \
    --image /path/to/query.png \
    --topk 50 \
    --top-datasets 5

保存结果：

python scripts/10_aggregate_recommendation.py \
    --query "chart numerical reasoning and visual question answering" \
    --topk 50 \
    --output data/vector_db/recommendation_result.json
"""

import argparse
import importlib.util
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import zvec


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEARCH_SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "09_search_zvec.py"
)

COLLECTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "vector_db"
    / "academic_5_zvec"
    / "academic_collection"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "vector_db"
    / "recommendation_result.json"
)


# ============================================================
# 2. 加载 09 脚本
# ============================================================

def load_search_module() -> Any:
    """
    动态加载 scripts/09_search_zvec.py。

    因为文件名以数字开头，不能直接写：
        import 09_search_zvec

    所以使用 importlib 动态加载。
    """
    if not SEARCH_SCRIPT_PATH.exists():
        raise FileNotFoundError(
            f"找不到查询脚本：{SEARCH_SCRIPT_PATH}"
        )

    spec = importlib.util.spec_from_file_location(
        "search_zvec_module",
        SEARCH_SCRIPT_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"无法加载脚本：{SEARCH_SCRIPT_PATH}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required_functions = [
        "call_query_embedding_api",
        "build_filter",
        "safe_similarity",
    ]

    for function_name in required_functions:
        if not hasattr(module, function_name):
            raise AttributeError(
                f"09_search_zvec.py 缺少函数：{function_name}"
            )

    return module


# ============================================================
# 3. 参数配置
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve similar benchmark samples and aggregate "
            "them into dataset-level recommendations."
        )
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Description of the desired benchmark capability.",
    )

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional local query image.",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help=(
            "Number of similar samples used for aggregation. "
            "Default: 50."
        ),
    )

    parser.add_argument(
        "--top-datasets",
        type=int,
        default=5,
        help="Number of recommended datasets to display.",
    )

    parser.add_argument(
        "--examples-per-dataset",
        type=int,
        default=3,
        help=(
            "Number of representative samples displayed "
            "for each dataset."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset_name filter.",
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

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Optional JSON output path. "
            "For example: data/vector_db/result.json"
        ),
    )

    return parser.parse_args()


# ============================================================
# 4. 数据清理辅助函数
# ============================================================

def safe_string(
    value: Any,
    default: str = "Unknown",
) -> str:
    """
    将值安全转换为非空字符串。
    """
    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    return text


def parse_ability(value: Any) -> list[str]:
    """
    把 ability 字段统一转换成字符串列表。

    Zvec 中的 ability 当前可能以 JSON 字符串保存：
        '["trend_analysis", "plot_understanding"]'
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        try:
            parsed = json.loads(value)

            if isinstance(parsed, list):
                return [
                    str(item).strip()
                    for item in parsed
                    if str(item).strip()
                ]
        except json.JSONDecodeError:
            pass

        return [value]

    return [str(value)]


# ============================================================
# 5. 结果聚合
# ============================================================

def aggregate_results(
    results: list[Any],
    safe_similarity_function: Any,
    examples_per_dataset: int,
) -> list[dict[str, Any]]:
    """
    将样本级检索结果聚合成数据集级推荐。

    聚合指标：

    1. hit_count
       该数据集在 Top-K 中出现的次数。

    2. hit_share
       该数据集命中数占全部 Top-K 的比例。

    3. average_similarity
       该数据集所有命中样本的平均相似度。

    4. weighted_similarity
       加入排名权重后的相似度。
       排名越靠前，权重越高。

    5. recommendation_score
       最终推荐分数：

       60% 排名加权相似度
       25% 平均相似度
       15% 命中比例

    这个分数用于数据集之间排序，不代表模型准确率。
    """
    if not results:
        return []

    grouped_results: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    total_results = len(results)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        fields = result.fields or {}

        dataset_name = safe_string(
            fields.get("dataset_name"),
            default="Unknown",
        )

        distance = float(result.score)

        similarity = float(
            safe_similarity_function(distance)
        )

        # 排名越靠前，权重越高。
        # Rank 1 权重为 1。
        rank_weight = 1.0 / math.log2(rank + 1)

        weighted_similarity = (
            similarity * rank_weight
        )

        grouped_results[dataset_name].append(
            {
                "rank": rank,
                "zvec_id": str(result.id),
                "original_id": safe_string(
                    fields.get("original_id"),
                    default="",
                ),
                "distance": distance,
                "similarity": similarity,
                "rank_weight": rank_weight,
                "weighted_similarity": weighted_similarity,
                "question": safe_string(
                    fields.get("question"),
                    default="",
                ),
                "task_type": safe_string(
                    fields.get("task_type"),
                ),
                "domain": safe_string(
                    fields.get("domain"),
                ),
                "difficulty": safe_string(
                    fields.get("difficulty"),
                ),
                "ability": parse_ability(
                    fields.get("ability")
                ),
                "image_path": safe_string(
                    fields.get("image_path"),
                    default="",
                ),
            }
        )

    aggregated: list[dict[str, Any]] = []

    for dataset_name, samples in grouped_results.items():
        hit_count = len(samples)
        hit_share = hit_count / total_results

        similarities = [
            sample["similarity"]
            for sample in samples
        ]

        average_similarity = (
            sum(similarities)
            / len(similarities)
        )

        best_similarity = max(similarities)

        weighted_similarity_sum = sum(
            sample["weighted_similarity"]
            for sample in samples
        )

        rank_weight_sum = sum(
            sample["rank_weight"]
            for sample in samples
        )

        normalized_weighted_similarity = (
            weighted_similarity_sum
            / rank_weight_sum
            if rank_weight_sum > 0
            else 0.0
        )

        # 统计任务类型、领域和能力
        task_counter = Counter(
            sample["task_type"]
            for sample in samples
            if sample["task_type"] != "Unknown"
        )

        domain_counter = Counter(
            sample["domain"]
            for sample in samples
            if sample["domain"] != "Unknown"
        )

        ability_counter: Counter[str] = Counter()

        for sample in samples:
            ability_counter.update(
                sample["ability"]
            )

        # 最终推荐分数
        recommendation_score = (
            0.60 * normalized_weighted_similarity
            + 0.25 * average_similarity
            + 0.15 * hit_share
        )

        # 按原始检索排名选择代表样本
        representative_samples = sorted(
            samples,
            key=lambda item: item["rank"],
        )[:examples_per_dataset]

        aggregated.append(
            {
                "dataset_name": dataset_name,
                "recommendation_score": recommendation_score,
                "hit_count": hit_count,
                "hit_share": hit_share,
                "average_similarity": average_similarity,
                "best_similarity": best_similarity,
                "weighted_similarity": (
                    normalized_weighted_similarity
                ),
                "top_task_types": [
                    {
                        "name": name,
                        "count": count,
                    }
                    for name, count
                    in task_counter.most_common(5)
                ],
                "top_domains": [
                    {
                        "name": name,
                        "count": count,
                    }
                    for name, count
                    in domain_counter.most_common(5)
                ],
                "top_abilities": [
                    {
                        "name": name,
                        "count": count,
                    }
                    for name, count
                    in ability_counter.most_common(8)
                ],
                "representative_samples": (
                    representative_samples
                ),
            }
        )

    aggregated.sort(
        key=lambda item: (
            item["recommendation_score"],
            item["hit_count"],
            item["best_similarity"],
        ),
        reverse=True,
    )

    for rank, item in enumerate(
        aggregated,
        start=1,
    ):
        item["recommendation_rank"] = rank

    return aggregated


# ============================================================
# 6. 打印推荐结果
# ============================================================

def format_counter_items(
    items: list[dict[str, Any]],
    limit: int = 5,
) -> str:
    """
    将统计结果格式化为：
        trend_analysis(5), plot_understanding(3)
    """
    if not items:
        return "None"

    return ", ".join(
        f"{item['name']}({item['count']})"
        for item in items[:limit]
    )


def print_recommendations(
    recommendations: list[dict[str, Any]],
    top_datasets: int,
) -> None:
    """
    打印数据集级推荐排名。
    """
    print("\n" + "=" * 90)
    print("Benchmark Dataset Recommendations")
    print("=" * 90)

    if not recommendations:
        print("No dataset recommendations were generated.")
        return

    selected = recommendations[:top_datasets]

    for item in selected:
        print("\n" + "-" * 90)
        print(
            f"Rank {item['recommendation_rank']}: "
            f"{item['dataset_name']}"
        )
        print("-" * 90)

        print(
            f"Recommendation score: "
            f"{item['recommendation_score']:.4f}"
        )

        print(
            f"Top-K hits:           "
            f"{item['hit_count']}"
        )

        print(
            f"Hit share:            "
            f"{item['hit_share']:.2%}"
        )

        print(
            f"Average similarity:   "
            f"{item['average_similarity']:.4f}"
        )

        print(
            f"Best similarity:      "
            f"{item['best_similarity']:.4f}"
        )

        print(
            f"Weighted similarity:  "
            f"{item['weighted_similarity']:.4f}"
        )

        print(
            "Main task types:       "
            + format_counter_items(
                item["top_task_types"]
            )
        )

        print(
            "Main domains:          "
            + format_counter_items(
                item["top_domains"]
            )
        )

        print(
            "Main abilities:        "
            + format_counter_items(
                item["top_abilities"],
                limit=8,
            )
        )

        print("\nRepresentative samples:")

        for sample in item[
            "representative_samples"
        ]:
            print(
                f"  Retrieval rank {sample['rank']} | "
                f"similarity={sample['similarity']:.4f}"
            )

            print(
                f"  Task: {sample['task_type']} | "
                f"Domain: {sample['domain']}"
            )

            print(
                f"  Question: {sample['question']}"
            )

            if sample["image_path"]:
                print(
                    f"  Image: {sample['image_path']}"
                )

            print()


# ============================================================
# 7. 保存 JSON
# ============================================================

def save_output(
    output_path: Path,
    query_text: str,
    query_image: str | None,
    topk: int,
    filter_string: str | None,
    recommendations: list[dict[str, Any]],
) -> None:
    """
    保存查询信息和推荐结果。
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_data = {
        "query": query_text,
        "query_image": query_image,
        "topk": topk,
        "filter": filter_string,
        "recommendations": recommendations,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nRecommendation result saved to: "
        f"{output_path}"
    )


# ============================================================
# 8. 主程序
# ============================================================

def main() -> None:
    args = parse_args()

    if args.topk <= 0:
        raise ValueError(
            "--topk must be greater than 0."
        )

    if args.top_datasets <= 0:
        raise ValueError(
            "--top-datasets must be greater than 0."
        )

    if args.examples_per_dataset < 0:
        raise ValueError(
            "--examples-per-dataset cannot be negative."
        )

    query_text = args.query.strip()

    if not query_text:
        raise ValueError(
            "--query cannot be empty."
        )

    api_key = os.getenv(
        "DASHSCOPE_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is not set."
        )

    if not COLLECTION_PATH.exists():
        raise FileNotFoundError(
            f"Zvec Collection not found: "
            f"{COLLECTION_PATH}"
        )

    search_module = load_search_module()

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

    filter_string = search_module.build_filter(
        dataset_name=args.dataset,
        domain=args.domain,
        task_type=args.task_type,
    )

    print("=" * 90)
    print("Academic-5 Benchmark Recommendation")
    print("=" * 90)
    print(f"Query: {query_text}")
    print(
        f"Image: "
        f"{image_path if image_path else 'None'}"
    )
    print(f"Search Top-K: {args.topk}")
    print(
        f"Filter: "
        f"{filter_string if filter_string else 'None'}"
    )

    # --------------------------------------------------------
    # 生成查询向量
    # --------------------------------------------------------

    print("\n[1/4] Generating query embedding...")

    query_vector = (
        search_module.call_query_embedding_api(
            api_key=api_key,
            text=query_text,
            image_path=image_path,
        )
    )

    print(
        f"Query embedding dimension: "
        f"{query_vector.shape[0]}"
    )

    # --------------------------------------------------------
    # 打开 Zvec
    # --------------------------------------------------------

    print("\n[2/4] Opening Zvec Collection...")

    collection = zvec.open(
        path=str(COLLECTION_PATH),
        option=zvec.CollectionOption(
            read_only=True,
            enable_mmap=True,
        ),
    )

    print("Collection opened successfully.")

    # --------------------------------------------------------
    # 搜索
    # --------------------------------------------------------

    print("\n[3/4] Retrieving similar samples...")

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
        f"Retrieved {len(results)} samples."
    )

    if not results:
        print(
            "No matching samples were found."
        )
        return

    # --------------------------------------------------------
    # 聚合
    # --------------------------------------------------------

    print("\n[4/4] Aggregating dataset recommendations...")

    recommendations = aggregate_results(
        results=results,
        safe_similarity_function=(
            search_module.safe_similarity
        ),
        examples_per_dataset=(
            args.examples_per_dataset
        ),
    )

    print_recommendations(
        recommendations=recommendations,
        top_datasets=args.top_datasets,
    )

    # --------------------------------------------------------
    # 保存结果
    # --------------------------------------------------------

    if args.output:
        output_path = Path(
            args.output
        ).expanduser()

        if not output_path.is_absolute():
            output_path = (
                PROJECT_ROOT
                / output_path
            )

        output_path = output_path.resolve()

        save_output(
            output_path=output_path,
            query_text=query_text,
            query_image=(
                str(image_path)
                if image_path
                else None
            ),
            topk=args.topk,
            filter_string=filter_string,
            recommendations=recommendations,
        )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nRecommendation interrupted by user."
        )

    except Exception as error:
        print("\n" + "=" * 90)
        print("Recommendation failed")
        print("=" * 90)
        print(
            f"Error type: "
            f"{type(error).__name__}"
        )
        print(
            f"Error message: {error}"
        )
        raise