# Academic Benchmark Retrieval

A sample-level multimodal benchmark retrieval and dataset recommendation system built on five academic visual reasoning datasets.

This project standardizes heterogeneous multimodal benchmark data into a unified representation, generates fused image-text embeddings with Qwen3-VL Embedding, builds a local Zvec vector index, retrieves representative samples based on model capability descriptions, and aggregates retrieval results into dataset-level recommendations.

The current implementation completes the full pipeline from raw benchmark data to searchable vector database and recommendation output.

---

## Overview

Modern multimodal models are often evaluated on a wide range of benchmarks, but selecting an appropriate benchmark remains difficult because datasets differ substantially in domain, task type, visual structure, reasoning requirements, and answer format.

This project addresses that problem by building a sample-level retrieval system over multiple academic benchmarks.

Instead of treating each benchmark as a single label, the system represents every sample through:

- its image;
- natural-language question;
- task type;
- domain;
- reasoning ability;
- answer format;
- structured embedding text.

A user can describe the capability they want to evaluate, optionally provide a query image, and retrieve the most relevant benchmark samples. The retrieved samples are then aggregated to identify the datasets most closely aligned with the target capability.

---

## Key Capabilities

The system currently supports:

- unified preprocessing of heterogeneous multimodal benchmarks;
- sample-level indexing across five datasets;
- fused image-text embedding generation;
- resumable API-based embedding construction;
- local vector indexing with Zvec;
- text-only and image-text retrieval;
- metadata-based filtering;
- dataset-level recommendation aggregation;
- representative sample selection;
- failed-request logging and recovery;
- end-to-end reproducible execution through ordered scripts.

---

## Academic-5 Benchmark Collection

The current database contains five multimodal academic benchmarks:

| Dataset | Primary capability | Samples |
|---|---|---:|
| QCalEval | Quantum calibration chart understanding and scientific reasoning | 500 |
| ScienceQA | Science visual question answering and knowledge reasoning | 500 |
| ChartQA | Chart understanding and numerical reasoning | 500 |
| PlotQA | Scientific plot interpretation and data reasoning | 500 |
| MathVista | Mathematical visual reasoning and multimodal problem solving | 500 |
| **Total** |  | **2,500** |

The collection is intentionally balanced at 500 samples per dataset to support controlled retrieval and dataset-level comparison.

Each sample is indexed independently rather than assigning only one embedding to an entire dataset. This enables more fine-grained matching between a query and the specific tasks, domains, visual patterns, and reasoning abilities present in the benchmark.

---

## System Architecture

```text
Raw benchmark datasets
        ↓
Dataset-specific preprocessing
        ↓
Unified sample-level JSONL records
        ↓
Academic-5 merged collection
        ↓
Qwen3-VL image-text embeddings
        ↓
1024-dimensional vector matrix
        ↓
Zvec local vector index
        ↓
Top-k sample retrieval
        ↓
Dataset-level aggregation
        ↓
Recommended benchmarks and representative samples
```

The system separates data preparation, embedding generation, indexing, retrieval, and recommendation into independent stages. This makes it possible to replace individual components, such as the embedding model or vector database, without rebuilding the entire pipeline architecture.

---

## Current Results

The completed system contains:

```text
Datasets:               5
Samples per dataset:    500
Total indexed samples:  2,500
Embedding model:        qwen3-vl-embedding
Embedding dimension:    1,024
Embedding dtype:        float32
Failed embeddings:      0
Vector index:           Zvec
```

Final embedding matrix:

```text
Shape: (2500, 1024)
Dtype: float32
NaN values: 0
Inf values: 0
```

The embedding rows and metadata rows are aligned one-to-one:

```text
Embedding records: 2500
Metadata records:  2500
Failed records:    0
```

The full pipeline has been completed and verified, including:

- preprocessing;
- dataset merging;
- multimodal embedding generation;
- Zvec index construction;
- similarity retrieval;
- dataset recommendation aggregation.

---

## Repository Structure

```text
academic_benchmark_retrieval/
├── data/
│   ├── raw/
│   │   └── Raw datasets and image files
│   ├── processed/
│   │   └── Unified sample-level JSONL files
│   └── vector_db/
│       ├── Embedding matrix
│       ├── Embedding metadata
│       ├── Failure logs
│       └── Zvec collection
├── scripts/
│   ├── 01_prepare_qcaleval.py
│   ├── 02_prepare_scienceqa.py
│   ├── 03_prepare_chartqa.py
│   ├── 04_prepare_plotqa.py
│   ├── 05_prepare_mathvista.py
│   ├── 06_merge_academic_5.py
│   ├── 07_build_qwenvl_embeddings.py
│   ├── 08_build_zvec_index.py
│   ├── 09_search_zvec.py
│   └── 10_aggregate_recommendation.py
├── .gitignore
├── requirements.txt
└── README.md
```

Large datasets, images, generated embeddings, local vector collections, virtual environments, and API credentials are excluded from version control.

---

## Unified Sample Representation

All datasets are converted into a shared sample-level schema.

A typical record contains:

```json
{
  "id": "sample_id",
  "dataset_name": "DatasetName",
  "image": "/absolute/path/to/image.png",
  "question": "Question text",
  "answer": "Ground-truth answer",
  "task_type": "task category",
  "domain": "domain category",
  "ability": [
    "visual understanding",
    "numerical reasoning"
  ],
  "text_for_embedding": "Structured text used together with the image for embedding"
}
```

The schema preserves dataset-specific information while exposing a common set of retrieval fields.

The `text_for_embedding` field summarizes the sample's task, question, reasoning requirement, domain, and answer format. It is combined with the image through the multimodal embedding interface.

---

## Environment

Recommended environment:

- Python 3.11 or 3.12
- Linux server
- DashScope API access
- sufficient disk space for raw images and generated vector files

A local GPU is not required because the current implementation calls the embedding model through the DashScope API.

Create and activate a virtual environment:

```bash
cd ~/academic_benchmark_retrieval

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Core dependencies include:

- `numpy`
- `requests`
- `tqdm`
- `datasets`
- `Pillow`
- `zvec`

---

## Pipeline

### 1. Prepare the five datasets

Run the preprocessing scripts in order:

```bash
python scripts/01_prepare_qcaleval.py
python scripts/02_prepare_scienceqa.py
python scripts/03_prepare_chartqa.py
python scripts/04_prepare_plotqa.py
python scripts/05_prepare_mathvista.py
```

Each script:

- reads one source dataset;
- samples or selects 500 records;
- resolves the corresponding image file;
- normalizes the record structure;
- generates `text_for_embedding`;
- writes a sample-level JSONL index.

The output files are stored under:

```text
data/processed/
```

Some preprocessing scripts contain dataset-specific local paths. These paths must be updated when the raw data is stored in a different location.

---

### 2. Merge Academic-5

Merge the five processed datasets:

```bash
python scripts/06_merge_academic_5.py
```

Output:

```text
data/processed/academic_5_merged.jsonl
```

Verify the total number of records:

```bash
wc -l data/processed/academic_5_merged.jsonl
```

Expected result:

```text
2500 data/processed/academic_5_merged.jsonl
```

---

### 3. Generate fused multimodal embeddings

Set the DashScope API key:

```bash
export DASHSCOPE_API_KEY="<your_api_key>"
```

The API key must not be written directly into the source code or committed to GitHub.

Run a single-record smoke test:

```bash
python scripts/07_build_qwenvl_embeddings.py --limit 1
```

Continue with additional records:

```bash
python scripts/07_build_qwenvl_embeddings.py --limit 10
```

Run the complete remaining collection:

```bash
python scripts/07_build_qwenvl_embeddings.py
```

The script sends the following inputs to the embedding model:

```text
image + text_for_embedding
```

Current embedding configuration:

```text
Model: qwen3-vl-embedding
Fusion mode: enabled
Dimension: 1024
```

The script supports resumable execution. IDs already present in the successful raw embedding file are skipped automatically.

Available options:

```text
--limit N
    Process only the first N pending records.

--reset
    Delete all existing embedding outputs and restart from the beginning.
```

Use `--reset` carefully because it causes all successful API calls to be repeated.

Outputs:

```text
data/vector_db/academic_5_embeddings_raw.jsonl
data/vector_db/academic_5_embeddings.npy
data/vector_db/academic_5_metadata.jsonl
data/vector_db/academic_5_embedding_failed.jsonl
```

The script also checks:

- image path validity;
- embedding dimension consistency;
- invalid JSONL records;
- NaN values;
- infinite values;
- API error messages;
- interrupted runs.

---

### 4. Validate embedding outputs

Check the record counts:

```bash
wc -l data/vector_db/academic_5_embeddings_raw.jsonl
wc -l data/vector_db/academic_5_metadata.jsonl
wc -l data/vector_db/academic_5_embedding_failed.jsonl
```

Expected output:

```text
2500 data/vector_db/academic_5_embeddings_raw.jsonl
2500 data/vector_db/academic_5_metadata.jsonl
0 data/vector_db/academic_5_embedding_failed.jsonl
```

Validate the NumPy matrix:

```bash
python - <<'PY'
import numpy as np

x = np.load("data/vector_db/academic_5_embeddings.npy")

print("Shape:", x.shape)
print("Dtype:", x.dtype)
print("NaN:", np.isnan(x).sum())
print("Inf:", np.isinf(x).sum())
PY
```

Expected output:

```text
Shape: (2500, 1024)
Dtype: float32
NaN: 0
Inf: 0
```

---

### 5. Build the Zvec index

Run:

```bash
python scripts/08_build_zvec_index.py
```

The script reads:

```text
data/vector_db/academic_5_embeddings.npy
data/vector_db/academic_5_metadata.jsonl
```

and builds the local Zvec collection under:

```text
data/vector_db/academic_5_zvec/
```

The default collection is:

```text
academic_collection
```

Each indexed document contains:

- the 1024-dimensional vector;
- dataset identity;
- sample ID;
- question;
- image path;
- task metadata;
- domain metadata;
- reasoning ability metadata.

---

### 6. Retrieve similar samples

Text query example:

```bash
export DASHSCOPE_API_KEY="<your_api_key>"

python scripts/09_search_zvec.py \
  --query "scientific plots, oscillation counting and trend analysis" \
  --topk 10
```

Image-text query example:

```bash
python scripts/09_search_zvec.py \
  --query "analyze the scientific behavior shown in this figure" \
  --image /path/to/query_image.png \
  --topk 10
```

Supported filters:

```text
--dataset NAME
--domain NAME
--task-type NAME
```

The query is embedded using the same model and embedding dimension as the indexed samples, ensuring that queries and stored vectors remain in the same representation space.

---

### 7. Aggregate dataset recommendations

Run:

```bash
python scripts/10_aggregate_recommendation.py \
  --query "scientific chart understanding and numerical reasoning" \
  --topk 50
```

The aggregation stage converts sample-level retrieval results into dataset-level recommendations.

It summarizes:

- how frequently each dataset appears;
- the similarity scores of retrieved samples;
- the relevance of each dataset to the query;
- representative examples from the top-ranked datasets.

Available options:

```text
--top-datasets N
--examples-per-dataset N
--output PATH
```

This allows the system to answer not only:

```text
Which individual samples are relevant?
```

but also:

```text
Which benchmark datasets are most appropriate for evaluating this capability?
```

---

## Example Use Cases

The system can be used to identify benchmarks for capabilities such as:

- scientific chart understanding;
- numerical reasoning;
- mathematical visual reasoning;
- trend analysis;
- plot interpretation;
- experimental diagnosis;
- knowledge-based science QA;
- calibration figure understanding;
- multi-choice visual reasoning;
- image-grounded problem solving.

A query can describe a model, a target task, a desired capability, or a new evaluation requirement.

---

## Engineering Features

### Resumable embedding generation

The embedding pipeline records successful samples incrementally. If execution is interrupted, the script resumes from unfinished IDs rather than repeating completed API calls.

### Explicit failure tracking

Failed samples are written to a separate JSONL file together with the sample ID, image path, and API error message.

### Metadata alignment

The vector matrix and metadata file are generated from the same successful raw embedding stream, preserving row-level correspondence.

### Dataset-level aggregation

The system does not stop at nearest-neighbor search. It converts sample retrieval results into a benchmark recommendation, which is more directly useful for evaluation planning.

### Modular pipeline

Each stage is implemented as an independent script, making the system easier to inspect, debug, replace, and extend.

---

## Data and Security Notes

### API credentials

The following scripts require `DASHSCOPE_API_KEY`:

```text
scripts/07_build_qwenvl_embeddings.py
scripts/09_search_zvec.py
scripts/10_aggregate_recommendation.py
```

Set the key through an environment variable:

```bash
export DASHSCOPE_API_KEY="<your_api_key>"
```

Do not commit:

- `.env`;
- API keys;
- access tokens;
- credentials embedded in shell history;
- screenshots containing complete credentials.

### Large files

The following directories are normally excluded from Git:

```text
.venv/
data/raw/
data/vector_db/
```

Raw benchmark data, images, generated embeddings, and local vector indexes should be transferred separately through approved storage.

### Dataset licenses

This repository contains the processing and retrieval code. The original datasets remain subject to their own licenses, access conditions, and redistribution restrictions.

---

## Completed Status

The current version has completed the complete Academic-5 retrieval pipeline:

- [x] QCalEval preprocessing
- [x] ScienceQA preprocessing
- [x] ChartQA preprocessing
- [x] PlotQA preprocessing
- [x] MathVista preprocessing
- [x] unified sample-level schema
- [x] balanced 2,500-sample merged collection
- [x] image-text fused embedding generation
- [x] 2,500 successful 1024-dimensional embeddings
- [x] embedding validation
- [x] metadata alignment
- [x] Zvec collection construction
- [x] text retrieval
- [x] multimodal retrieval
- [x] metadata filtering
- [x] dataset-level aggregation
- [x] representative sample recommendation
- [x] resumable API execution
- [x] detailed error logging

---

## Future Work

Potential extensions include:

- configuration-driven dataset paths;
- automatic data validation before embedding;
- retrieval evaluation with Recall@K and ranking metrics;
- comparison of alternative multimodal embedding models;
- hybrid sparse-dense retrieval;
- reranking;
- additional academic and domain-specific benchmarks;
- automated benchmark coverage analysis;
- a REST API;
- a lightweight Web interface;
- continuous integration and unit tests.

---

## License

The source code license can be added according to the intended release policy.

The original datasets are not relicensed by this repository and remain governed by their respective licenses and terms of use.