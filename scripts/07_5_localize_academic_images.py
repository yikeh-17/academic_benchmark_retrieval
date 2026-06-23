"""
07_5_localize_academic_images.py

作用：
1. 将 Academic-5 的 2500 张图片统一复制到当前项目中
2. 按数据集分别存放
3. 将 JSONL 中的绝对路径或外部路径改为项目相对路径
4. 同步更新：
   - academic_5_metadata.jsonl
   - academic_5_merged.jsonl
   - academic_5_embeddings_raw.jsonl
5. 自动创建 .bak 备份

运行：
    python scripts/07_5_localize_academic_images.py

仅检查、不实际修改：
    python scripts/07_5_localize_academic_images.py --dry-run
"""

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


# ============================================================
# 1. 路径配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = (
    PROJECT_ROOT
    / "data/vector_db/academic_5_metadata.jsonl"
)

MERGED_PATH = (
    PROJECT_ROOT
    / "data/processed/academic_5_merged.jsonl"
)

RAW_EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data/vector_db/academic_5_embeddings_raw.jsonl"
)

TARGET_IMAGE_ROOT = (
    PROJECT_ROOT
    / "data/raw/academic_5_images"
)


# ============================================================
# 2. 参数
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy Academic-5 images into the current project "
            "and convert image paths to relative paths."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Only inspect and print planned changes. "
            "Do not copy files or modify JSONL."
        ),
    )

    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Overwrite images that already exist.",
    )

    return parser.parse_args()


# ============================================================
# 3. 辅助函数
# ============================================================

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON 格式错误：{path} 第 {line_number} 行：{error}"
                ) from error

            if not isinstance(item, dict):
                raise ValueError(
                    f"{path} 第 {line_number} 行不是 JSON 对象。"
                )

            records.append(item)

    return records


def write_jsonl_atomic(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """
    先写临时文件，再替换原文件，避免中途失败破坏 JSONL。
    """
    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp_path.open("w", encoding="utf-8") as file:
        for item in records:
            file.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )

    temp_path.replace(path)


def create_backup(path: Path) -> Path | None:
    """
    为原始 JSONL 创建一次备份。
    如果 .bak 已存在，则不覆盖。
    """
    if not path.exists():
        return None

    backup_path = path.with_suffix(
        path.suffix + ".bak"
    )

    if not backup_path.exists():
        shutil.copy2(path, backup_path)

    return backup_path


def get_image_value(
    item: dict[str, Any],
) -> str | None:
    """
    兼容 image 和 image_path 两种字段。
    """
    value = item.get("image")

    if not value:
        value = item.get("image_path")

    if value is None:
        return None

    value = str(value).strip()

    return value or None


def resolve_source_path(
    image_value: str,
) -> Path:
    """
    将 JSON 中的路径转换为真实绝对路径。

    兼容：
    - 绝对路径
    - 相对于项目根目录的路径
    """
    path = Path(image_value).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (
        PROJECT_ROOT
        / path
    ).resolve()


def normalize_dataset_name(
    value: Any,
) -> str:
    """
    将数据集名称转换成安全的小写目录名。
    """
    if value is None:
        return "unknown"

    text = str(value).strip().lower()

    aliases = {
        "qcaleval": "qcaleval",
        "scienceqa": "scienceqa",
        "chartqa": "chartqa",
        "plotqa": "plotqa",
        "mathvista": "mathvista",
    }

    compact = re.sub(
        r"[^a-z0-9]+",
        "",
        text,
    )

    if compact in aliases:
        return aliases[compact]

    safe = re.sub(
        r"[^a-z0-9_-]+",
        "_",
        text,
    ).strip("_")

    return safe or "unknown"


def safe_filename_component(
    value: Any,
    default: str,
) -> str:
    """
    将样本 ID 转换成安全文件名。
    """
    text = str(
        value if value is not None else default
    ).strip()

    safe = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        text,
    ).strip("._")

    return safe or default


def get_dataset_name(
    item: dict[str, Any],
) -> str:
    return normalize_dataset_name(
        item.get("dataset_name")
        or item.get("dataset")
    )


def choose_extension(
    source_path: Path,
) -> str:
    extension = source_path.suffix.lower()

    valid_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".gif",
    }

    if extension in valid_extensions:
        return extension

    return ".png"


def to_project_relative_path(
    absolute_path: Path,
) -> str:
    """
    返回统一使用正斜杠的项目相对路径。
    """
    return absolute_path.relative_to(
        PROJECT_ROOT
    ).as_posix()


def set_image_path(
    item: dict[str, Any],
    relative_path: str,
) -> None:
    """
    同时统一 image 和 image_path。

    保留 image 作为主要字段，便于 07 脚本读取；
    同时写 image_path，便于后续检索和展示。
    """
    item["image"] = relative_path
    item["image_path"] = relative_path


# ============================================================
# 4. 复制图片并创建路径映射
# ============================================================

def build_image_mapping(
    metadata_records: list[dict[str, Any]],
    dry_run: bool,
    overwrite_images: bool,
) -> tuple[
    dict[str, str],
    list[dict[str, str]],
]:
    """
    返回：
    1. sample_id -> 新相对路径
    2. 缺失或失败记录
    """

    id_to_new_path: dict[str, str] = {}
    failed_records: list[dict[str, str]] = []

    copied_count = 0
    existing_count = 0

    for index, item in enumerate(
        metadata_records
    ):
        sample_id = str(
            item.get("id", f"doc_{index}")
        ).strip()

        image_value = get_image_value(item)

        if not image_value:
            failed_records.append(
                {
                    "id": sample_id,
                    "error": "missing_image_field",
                }
            )
            continue

        source_path = resolve_source_path(
            image_value
        )

        if not source_path.is_file():
            failed_records.append(
                {
                    "id": sample_id,
                    "error": "source_image_not_found",
                    "source_path": str(source_path),
                }
            )
            continue

        dataset_folder = get_dataset_name(
            item
        )

        safe_id = safe_filename_component(
            sample_id,
            default=f"doc_{index}",
        )

        extension = choose_extension(
            source_path
        )

        target_path = (
            TARGET_IMAGE_ROOT
            / dataset_folder
            / f"{safe_id}{extension}"
        )

        relative_path = (
            to_project_relative_path(
                target_path
            )
        )

        # 防止不同记录的 ID 冲突
        if (
            target_path.exists()
            and not overwrite_images
        ):
            existing_count += 1
        else:
            if not dry_run:
                target_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                shutil.copy2(
                    source_path,
                    target_path,
                )

            copied_count += 1

        id_to_new_path[sample_id] = (
            relative_path
        )

        if (
            index < 5
            or (index + 1) % 500 == 0
        ):
            print(
                f"[{index + 1}/{len(metadata_records)}] "
                f"{source_path} -> {relative_path}"
            )

    print("\n图片处理统计：")
    print("  可映射样本数：", len(id_to_new_path))
    print("  新复制/计划复制：", copied_count)
    print("  已存在并跳过：", existing_count)
    print("  缺失或失败：", len(failed_records))

    return id_to_new_path, failed_records


# ============================================================
# 5. 更新 JSONL
# ============================================================

def update_records_with_mapping(
    records: list[dict[str, Any]],
    id_to_new_path: dict[str, str],
) -> tuple[
    list[dict[str, Any]],
    int,
    int,
]:
    updated_count = 0
    unmatched_count = 0

    for index, item in enumerate(records):
        sample_id = str(
            item.get("id", f"doc_{index}")
        ).strip()

        new_path = id_to_new_path.get(
            sample_id
        )

        if new_path is None:
            unmatched_count += 1
            continue

        set_image_path(
            item,
            new_path,
        )

        updated_count += 1

    return (
        records,
        updated_count,
        unmatched_count,
    )


def update_jsonl_file(
    path: Path,
    id_to_new_path: dict[str, str],
    dry_run: bool,
) -> None:
    if not path.exists():
        print(
            f"跳过不存在的文件：{path}"
        )
        return

    records = load_jsonl(path)

    (
        updated_records,
        updated_count,
        unmatched_count,
    ) = update_records_with_mapping(
        records,
        id_to_new_path,
    )

    print(f"\n更新文件：{path}")
    print("  总记录数：", len(records))
    print("  已更新路径：", updated_count)
    print("  未匹配记录：", unmatched_count)

    if dry_run:
        print("  Dry-run：未实际写入")
        return

    backup_path = create_backup(path)

    if backup_path:
        print(
            f"  原文件备份：{backup_path}"
        )

    write_jsonl_atomic(
        path,
        updated_records,
    )

    print("  写入完成")


# ============================================================
# 6. 最终验证
# ============================================================

def verify_metadata_images() -> None:
    records = load_jsonl(
        METADATA_PATH
    )

    found = 0
    missing: list[
        tuple[str, str]
    ] = []

    for index, item in enumerate(records):
        sample_id = str(
            item.get("id", f"doc_{index}")
        )

        image_value = get_image_value(
            item
        )

        if not image_value:
            missing.append(
                (
                    sample_id,
                    "NO_IMAGE_PATH",
                )
            )
            continue

        image_path = resolve_source_path(
            image_value
        )

        if image_path.is_file():
            found += 1
        else:
            missing.append(
                (
                    sample_id,
                    str(image_path),
                )
            )

    print("\n" + "=" * 70)
    print("最终图片验证")
    print("=" * 70)
    print("Metadata 总记录：", len(records))
    print("图片存在：", found)
    print("图片缺失：", len(missing))

    if missing:
        print("\n前 10 条缺失记录：")

        for sample_id, path in missing[:10]:
            print(
                f"  {sample_id}: {path}"
            )


# ============================================================
# 7. 主程序
# ============================================================

def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("Academic-5 图片本地化")
    print("=" * 70)
    print("项目根目录：", PROJECT_ROOT)
    print("目标图片目录：", TARGET_IMAGE_ROOT)
    print("Dry-run：", args.dry_run)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到 metadata 文件：{METADATA_PATH}"
        )

    metadata_records = load_jsonl(
        METADATA_PATH
    )

    print(
        "\nMetadata 记录数：",
        len(metadata_records),
    )

    (
        id_to_new_path,
        failed_records,
    ) = build_image_mapping(
        metadata_records=metadata_records,
        dry_run=args.dry_run,
        overwrite_images=args.overwrite_images,
    )

    # 更新三个关键 JSONL
    for jsonl_path in [
        METADATA_PATH,
        MERGED_PATH,
        RAW_EMBEDDINGS_PATH,
    ]:
        update_jsonl_file(
            path=jsonl_path,
            id_to_new_path=id_to_new_path,
            dry_run=args.dry_run,
        )

    if failed_records:
        failed_path = (
            PROJECT_ROOT
            / "data/vector_db"
            / "academic_5_image_localization_failed.jsonl"
        )

        print(
            f"\n存在 {len(failed_records)} 条失败记录。"
        )

        if not args.dry_run:
            write_jsonl_atomic(
                failed_path,
                failed_records,
            )

            print(
                f"失败记录保存到：{failed_path}"
            )

    if not args.dry_run:
        verify_metadata_images()

    print("\n完成。")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n操作被用户中断。")

    except Exception as error:
        print("\n迁移失败：")
        print(
            f"{type(error).__name__}: {error}"
        )
        raise
