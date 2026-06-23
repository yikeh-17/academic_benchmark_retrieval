# Academic Benchmark Retrieval

这个仓库包含 Academic-5 基准检索流程的预处理、嵌入构建、向量索引构建和检索脚本。

## 项目结构

- `data/raw/` - 原始数据集和图像资源
- `data/processed/` - 处理后可用于嵌入和检索的 JSONL 索引
- `data/vector_db/` - 向量嵌入、元数据以及 zvec 索引输出
- `scripts/` - 数据准备、融合、嵌入构建、索引构建、检索和推荐脚本

## 依赖环境

推荐使用 Python 3.11/3.12。

至少需要以下包：

- `numpy`
- `requests`
- `tqdm`
- `datasets`
- `zvec`

如果你使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy requests tqdm datasets zvec
```

> 当前仓库中没有 `requirements.txt` / `pyproject.toml`，建议后续补充。

## 运行顺序

### 1. 数据预处理

每个准备脚本会将原始数据转换成统一的 JSONL 索引格式，并且会尝试定位本地图片路径。

- `python scripts/01_prepare_qcaleval.py`
- `python scripts/02_prepare_scienceqa.py`
- `python scripts/03_prepare_chartqa.py`
- `python scripts/04_prepare_plotqa.py`
- `python scripts/05_prepare_mathvista.py`

注意：这些脚本中的图片根路径通常是硬编码的本地路径，例如 `SOURCE_IMAGE_ROOT`、`SCIENCEQA_IMAGE_ROOT`、`CHARTQA_IMAGE_ROOT`、`PROJECT_ROOT / "data/raw/plotqa/images"` 和 `PROJECT_ROOT / "data/raw/mathvista/images"`。如果你的数据位置不同，需要手动修改相应脚本中的路径。

### 2. 合并数据集

将各个数据集索引合并为统一数据集：

```bash
python scripts/06_merge_academic_5.py
```

输出文件：

- `data/processed/academic_5_merged.jsonl`

### 3. 构建 QWEN-VL 多模态嵌入

使用 `DASHSCOPE_API_KEY` 调用多模态嵌入接口。

```bash
export DASHSCOPE_API_KEY="<your_api_key>"
python scripts/07_build_qwenvl_embeddings.py
```

可选参数：

- `--limit N` 只处理前 N 条待计算记录
- `--reset` 删除已有输出并重新开始

输出文件：

- `data/vector_db/academic_5_embeddings_raw.jsonl`
- `data/vector_db/academic_5_embeddings.npy`
- `data/vector_db/academic_5_metadata.jsonl`
- `data/vector_db/academic_5_embedding_failed.jsonl`

### 4. 构建 zvec 索引

```bash
python scripts/08_build_zvec_index.py
```

该脚本读取：

- `data/vector_db/academic_5_embeddings.npy`
- `data/vector_db/academic_5_metadata.jsonl`

并构建本地 zvec collection：

- `data/vector_db/academic_5_zvec/academic_collection`

### 5. 检索相似样本

```bash
export DASHSCOPE_API_KEY="<your_api_key>"
python scripts/09_search_zvec.py --query "scientific plots, oscillation counting and trend analysis" --topk 10
```

可选参数：

- `--image /path/to/query_image.png` 进行多模态查询
- `--dataset` 过滤特定数据集名称
- `--domain` 过滤领域
- `--task-type` 过滤任务类型

### 6. 聚合推荐

从检索结果汇总推荐数据集和代表样本：

```bash
python scripts/10_aggregate_recommendation.py --query "..." --topk 50
```

可选参数：

- `--top-datasets` 结果中推荐的数据集数量
- `--examples-per-dataset` 每个推荐数据集显示的样本数
- `--output` 保存 JSON 输出

## 重要说明

- `DASHSCOPE_API_KEY` 是两个脚本的必需环境变量：
  - `scripts/07_build_qwenvl_embeddings.py`
  - `scripts/09_search_zvec.py`
- `scripts/04_prepare_plotqa.py` 依赖 Hugging Face 的 `achang/plot_qa` 数据集下载。
- 多个预处理脚本使用硬编码的本地图片目录，若数据路径不同需要修改脚本。
- 当前仓库中没有测试文件，建议补充测试或 `requirements.txt`。

## 已确认状态

- 所有 `scripts/*.py` 已通过 `python -m py_compile` 语法检查。
- 当前环境已确认可导入 `numpy`、`requests`、`tqdm`、`datasets`。

## 后续建议

1. 添加 `requirements.txt` 或 `pyproject.toml`。
2. 创建一个简单的 `README.md`（已补充）。
3. 如果希望进一步自动化，可以将路径参数化，而不是依赖硬编码目录。
4. 如果需要，可进一步补充测试脚本。