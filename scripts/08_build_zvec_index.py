"""
08_build_zvec_index.py

作用：
1. 读取 academic_5_embeddings.npy 中的多模态向量
2. 读取 academic_5_metadata.jsonl 中的样本元数据
3. 将向量和元数据写入本地 Zvec Collection
4. 构建 HNSW 向量索引和标量倒排索引
5. 将数据库持久化到 data/vector_db/academic_5_zvec/

运行方式：
    python scripts/08_build_zvec_index.py
"""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import zvec


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VECTOR_DB_DIR = PROJECT_ROOT / "data" / "vector_db"

INPUT_NPY_PATH = (
    VECTOR_DB_DIR
    / "academic_5_embeddings.npy"
)

INPUT_META_PATH = (
    VECTOR_DB_DIR
    / "academic_5_metadata.jsonl"
)

OUTPUT_ZDB_DIR = (
    VECTOR_DB_DIR
    / "academic_5_zvec"
)

COLLECTION_PATH = (
    OUTPUT_ZDB_DIR
    / "academic_collection"
)


# 每批最多写入 1000 条
BATCH_SIZE = 1000


# ============================================================
# 2. 辅助函数
# ============================================================

def safe_string(
    value: Any,
    default: str = "",
) -> str:
    """
    将任意值安全转换成字符串。

    对 list、dict 使用 JSON 字符串保存，
    避免直接 str() 后格式不统一。
    """
    if value is None:
        return default

    if isinstance(value, (list, dict)):
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    value = str(value).strip()

    if not value:
        return default

    return value


def get_first_available(
    metadata: dict,
    keys: list[str],
    default: Any = "",
) -> Any:
    """
    按顺序查找第一个存在且非空的字段。

    例如兼容：
    dataset_name / dataset
    image / image_path
    """
    for key in keys:
        if key not in metadata:
            continue

        value = metadata[key]

        if value is None:
            continue

        if isinstance(value, str) and not value.strip():
            continue

        return value

    return default


def load_metadata(
    metadata_path: Path,
) -> list[dict]:
    """
    逐行加载 JSONL 元数据。
    """
    metadata_list: list[dict] = []

    with metadata_path.open(
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
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"元数据第 {line_number} 行不是合法 JSON：{exc}"
                ) from exc

            if not isinstance(item, dict):
                raise ValueError(
                    f"元数据第 {line_number} 行不是 JSON 对象。"
                )

            metadata_list.append(item)

    return metadata_list


def validate_embeddings(
    embeddings: np.ndarray,
) -> None:
    """
    检查向量矩阵是否合法。
    """
    if embeddings.ndim != 2:
        raise ValueError(
            "向量矩阵必须是二维矩阵，"
            f"当前 shape={embeddings.shape}"
        )

    if embeddings.shape[0] == 0:
        raise ValueError(
            "向量矩阵中没有任何样本。"
        )

    if embeddings.shape[1] == 0:
        raise ValueError(
            "向量维度不能为 0。"
        )

    # 检查 NaN 和正负无穷
    valid_rows = np.isfinite(
        embeddings
    ).all(axis=1)

    invalid_indices = np.where(
        ~valid_rows
    )[0]

    if len(invalid_indices) > 0:
        raise ValueError(
            "向量中包含 NaN 或 Inf。"
            f"异常向量数量：{len(invalid_indices)}；"
            f"前几个位置：{invalid_indices[:10].tolist()}"
        )

    # Cosine 相似度不适合全零向量
    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    zero_indices = np.where(
        norms == 0
    )[0]

    if len(zero_indices) > 0:
        raise ValueError(
            "发现全零向量，无法正常计算 Cosine 相似度。"
            f"全零向量数量：{len(zero_indices)}；"
            f"前几个位置：{zero_indices[:10].tolist()}"
        )


def build_document(
    row_index: int,
    vector: np.ndarray,
    metadata: dict,
) -> zvec.Doc:
    """
    将一条向量和对应元数据组装成 Zvec Doc。
    """

    raw_id = safe_string(
        get_first_available(
            metadata,
            ["id", "sample_id"],
            default=f"doc_{row_index}",
        ),
        default=f"doc_{row_index}",
    )

    dataset_name = safe_string(
        get_first_available(
            metadata,
            ["dataset_name", "dataset"],
            default="Unknown",
        ),
        default="Unknown",
    )

    task_type = safe_string(
        get_first_available(
            metadata,
            ["task_type", "question_type"],
            default="Unknown",
        ),
        default="Unknown",
    )

    difficulty = safe_string(
        metadata.get(
            "difficulty",
            "unknown",
        ),
        default="unknown",
    )

    question = safe_string(
        metadata.get(
            "question",
            "",
        )
    )

    answer = safe_string(
        metadata.get(
            "answer",
            "",
        )
    )

    image_path = safe_string(
        get_first_available(
            metadata,
            ["image", "image_path"],
            default="",
        )
    )

    domain = safe_string(
        metadata.get(
            "domain",
            "unknown",
        ),
        default="unknown",
    )

    image_type = safe_string(
        get_first_available(
            metadata,
            ["image_type", "context"],
            default="unknown",
        ),
        default="unknown",
    )

    modality = safe_string(
        metadata.get(
            "modality",
            "image+text",
        ),
        default="image+text",
    )

    ability = safe_string(
        get_first_available(
            metadata,
            ["ability", "skills"],
            default=[],
        ),
        default="[]",
    )

    text_for_embedding = safe_string(
        metadata.get(
            "text_for_embedding",
            question,
        )
    )

    # 使用连续数字字符串作为 Zvec 主键，
    # 避免不同数据集的原始 ID 发生重复。
    document_id = str(row_index)

    return zvec.Doc(
        id=document_id,
        vectors={
            "embedding": vector.tolist(),
        },
        fields={
            "original_id": raw_id,
            "dataset_name": dataset_name,
            "task_type": task_type,
            "difficulty": difficulty,
            "question": question,
            "answer": answer,
            "image_path": image_path,
            "domain": domain,
            "image_type": image_type,
            "modality": modality,
            "ability": ability,
            "text_for_embedding": text_for_embedding,
        },
    )


# ============================================================
# 3. Schema
# ============================================================

def create_schema(
    embedding_dimension: int,
) -> zvec.CollectionSchema:
    """
    定义 Zvec Collection 的数据结构。
    """
    return zvec.CollectionSchema(
        name="academic_collection",

        # 普通字段
        fields=[
            # 仅用于展示，不建立索引
            zvec.FieldSchema(
                name="original_id",
                data_type=zvec.DataType.STRING,
            ),

            # 后续经常用于过滤，建立倒排索引
            zvec.FieldSchema(
                name="dataset_name",
                data_type=zvec.DataType.STRING,
                index_param=zvec.InvertIndexParam(),
            ),

            zvec.FieldSchema(
                name="task_type",
                data_type=zvec.DataType.STRING,
                index_param=zvec.InvertIndexParam(),
            ),

            zvec.FieldSchema(
                name="difficulty",
                data_type=zvec.DataType.STRING,
                index_param=zvec.InvertIndexParam(),
            ),

            # 问题和答案主要用于展示，不建立倒排索引
            zvec.FieldSchema(
                name="question",
                data_type=zvec.DataType.STRING,
            ),

            zvec.FieldSchema(
                name="answer",
                data_type=zvec.DataType.STRING,
            ),

            zvec.FieldSchema(
                name="image_path",
                data_type=zvec.DataType.STRING,
            ),

            # 领域、图像类型、模态通常会作为过滤条件
            zvec.FieldSchema(
                name="domain",
                data_type=zvec.DataType.STRING,
                index_param=zvec.InvertIndexParam(),
            ),

            zvec.FieldSchema(
                name="image_type",
                data_type=zvec.DataType.STRING,
                index_param=zvec.InvertIndexParam(),
            ),

            zvec.FieldSchema(
                name="modality",
                data_type=zvec.DataType.STRING,
                index_param=zvec.InvertIndexParam(),
            ),

            # ability 目前按 JSON 字符串保存
            zvec.FieldSchema(
                name="ability",
                data_type=zvec.DataType.STRING,
            ),

            zvec.FieldSchema(
                name="text_for_embedding",
                data_type=zvec.DataType.STRING,
            ),
        ],

        # 向量字段
        vectors=[
            zvec.VectorSchema(
                name="embedding",
                data_type=zvec.DataType.VECTOR_FP32,
                dimension=embedding_dimension,
                index_param=zvec.HnswIndexParam(
                    metric_type=zvec.MetricType.COSINE,
                ),
            ),
        ],
    )


# ============================================================
# 4. 构建数据库
# ============================================================

def build_zvec_index() -> None:
    """
    构建完整 Zvec 向量数据库。
    """

    print("=" * 70)
    print("开始构建 Academic-5 Zvec 向量数据库")
    print("=" * 70)

    # --------------------------------------------------------
    # 检查输入文件
    # --------------------------------------------------------

    if not INPUT_NPY_PATH.exists():
        raise FileNotFoundError(
            f"找不到向量文件：{INPUT_NPY_PATH}"
        )

    if not INPUT_META_PATH.exists():
        raise FileNotFoundError(
            f"找不到元数据文件：{INPUT_META_PATH}"
        )

    # --------------------------------------------------------
    # 加载向量
    # --------------------------------------------------------

    print(f"\n[1/7] 加载向量矩阵：")
    print(f"      {INPUT_NPY_PATH}")

    embeddings = np.load(
        INPUT_NPY_PATH,
        allow_pickle=False,
    )

    # Zvec Schema 使用 VECTOR_FP32
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    validate_embeddings(
        embeddings
    )

    num_records = embeddings.shape[0]
    embedding_dimension = embeddings.shape[1]

    print(f"      向量 shape：{embeddings.shape}")
    print(f"      向量 dtype：{embeddings.dtype}")
    print(f"      样本数量：{num_records}")
    print(f"      向量维度：{embedding_dimension}")

    # --------------------------------------------------------
    # 加载元数据
    # --------------------------------------------------------

    print(f"\n[2/7] 加载元数据：")
    print(f"      {INPUT_META_PATH}")

    metadata_list = load_metadata(
        INPUT_META_PATH
    )

    print(
        f"      元数据数量：{len(metadata_list)}"
    )

    # --------------------------------------------------------
    # 检查数量对应
    # --------------------------------------------------------

    print("\n[3/7] 检查向量与元数据对应关系")

    if len(metadata_list) != num_records:
        raise ValueError(
            "向量数量与元数据数量不一致："
            f"向量={num_records}，"
            f"元数据={len(metadata_list)}。"
            "请检查 embedding 和 metadata 是否由同一批、"
            "相同顺序的样本生成。"
        )

    print(
        "      检查通过：向量数量与元数据数量一致。"
    )

    # --------------------------------------------------------
    # 创建 Schema
    # --------------------------------------------------------

    print("\n[4/7] 创建 Zvec Collection Schema")

    schema = create_schema(
        embedding_dimension
    )

    print(
        "      向量索引：HNSW"
    )
    print(
        "      相似度度量：COSINE"
    )
    print(
        "      过滤索引字段：dataset_name、task_type、"
        "difficulty、domain、image_type、modality"
    )

    # --------------------------------------------------------
    # 删除旧数据库
    # --------------------------------------------------------

    print("\n[5/7] 准备数据库目录")

    OUTPUT_ZDB_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if COLLECTION_PATH.exists():
        print(
            f"      发现旧数据库，正在删除：{COLLECTION_PATH}"
        )
        shutil.rmtree(
            COLLECTION_PATH
        )

    print(
        f"      创建数据库：{COLLECTION_PATH}"
    )

    collection = zvec.create_and_open(
        path=str(COLLECTION_PATH),
        schema=schema,
    )

    # --------------------------------------------------------
    # 分批构造和插入
    # --------------------------------------------------------

    print("\n[6/7] 分批写入文档")

    total_batches = (
        num_records + BATCH_SIZE - 1
    ) // BATCH_SIZE

    for batch_number, start_index in enumerate(
        range(
            0,
            num_records,
            BATCH_SIZE,
        ),
        start=1,
    ):
        end_index = min(
            start_index + BATCH_SIZE,
            num_records,
        )

        batch_docs = []

        for row_index in range(
            start_index,
            end_index,
        ):
            doc = build_document(
                row_index=row_index,
                vector=embeddings[row_index],
                metadata=metadata_list[row_index],
            )

            batch_docs.append(doc)

        collection.insert(
            batch_docs
        )

        print(
            f"      批次 {batch_number}/{total_batches}："
            f"已写入第 {start_index + 1} "
            f"到第 {end_index} 条"
        )

        # 释放当前批次对象
        del batch_docs

    # --------------------------------------------------------
    # Optimize
    # --------------------------------------------------------

    print("\n[7/7] 构建并优化 HNSW 向量索引")
    print(
        "      正在执行 collection.optimize()，"
        "数据量较大时可能需要一些时间……"
    )

    collection.optimize()

    print(
        "      HNSW 索引优化完成。"
    )

    # --------------------------------------------------------
    # 输出统计信息
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("Zvec 数据库构建完成")
    print("=" * 70)

    print(
        f"数据库路径：{COLLECTION_PATH}"
    )
    print(
        f"写入样本数：{num_records}"
    )
    print(
        f"向量维度：{embedding_dimension}"
    )

    print("\nCollection Schema：")
    print(collection.schema)

    print("\nCollection Stats：")
    print(collection.stats)

    print("\n后续查询时请打开：")
    print(
        f'collection = zvec.open('
        f'path="{COLLECTION_PATH}")'
    )


# ============================================================
# 5. 程序入口
# ============================================================

if __name__ == "__main__":
    try:
        build_zvec_index()

    except KeyboardInterrupt:
        print(
            "\n程序被用户中断。"
        )

    except Exception as exc:
        print("\n" + "=" * 70)
        print("Zvec 数据库构建失败")
        print("=" * 70)
        print(
            f"错误类型：{type(exc).__name__}"
        )
        print(
            f"错误信息：{exc}"
        )
        raise