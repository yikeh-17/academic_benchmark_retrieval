# EvalAgent: Intelligent Multimodal Benchmark Retrieval and Evaluation Planning

EvalAgent is an intelligent benchmark recommendation system for multimodal model evaluation.

The system converts natural-language evaluation requirements into structured capability profiles, retrieves relevant samples from multiple academic benchmarks using multimodal embeddings and Zvec vector search, and produces a ranked evaluation set with dataset, task, ability, domain, and difficulty coverage analysis.

The project currently integrates five multimodal academic benchmarks:

- QCalEval
- ScienceQA
- ChartQA
- PlotQA
- MathVista

A total of 2,500 benchmark samples are indexed, with 500 samples selected from each dataset.

---

## 1. Project Overview

Evaluating multimodal models often requires manually selecting datasets, identifying relevant capabilities, and assembling representative test samples.

EvalAgent automates this process.

Given a natural-language requirement such as:

```text
I want to evaluate a multimodal model for understanding quantum
experiment plots, with emphasis on trend analysis, parameter
extraction, and scientific reasoning. Prefer medium and hard samples,
with a budget of 20 examples.
```

EvalAgent automatically:

1. Parses the evaluation requirement.
2. Extracts target domains, tasks, abilities, difficulty preferences, and evaluation budget.
3. Builds an embedding-oriented retrieval query.
4. Generates a 1,024-dimensional multimodal query embedding.
5. Searches the Academic-5 Zvec vector database.
6. Reranks candidates using semantic similarity and metadata matching.
7. Selects the final evaluation samples.
8. Produces dataset allocation and capability coverage statistics.
9. Saves the evaluation recommendation as structured JSON.

---

## 2. System Architecture

```text
Natural-Language Evaluation Requirement
                    |
                    v
        Requirement Parsing Agent
                    |
                    v
        Structured Evaluation Profile
                    |
                    v
        Retrieval Query Construction
                    |
                    v
       Qwen3-VL Embedding Generation
                    |
                    v
          Zvec Vector Retrieval
                    |
                    v
       Metadata-Aware Candidate Reranking
                    |
                    v
     Recommended Evaluation Sample Set
                    |
                    v
 Dataset / Task / Ability / Difficulty Report
```

---

## 3. Main Features

### Natural-Language Requirement Parsing

EvalAgent converts user requirements into a structured profile containing:

```json
{
  "model_name": null,
  "model_description": "",
  "target_domains": [],
  "target_tasks": [],
  "target_abilities": [],
  "preferred_difficulties": [],
  "preferred_datasets": [],
  "evaluation_budget": 20
}
```

The current implementation uses deterministic label mapping to provide stable and interpretable outputs.

### Multimodal Embedding Retrieval

The system uses:

```text
qwen3-vl-embedding
```

to generate 1,024-dimensional embeddings for benchmark samples and user queries.

The embedding input can contain:

- Text only
- Image only
- Text and image

### Zvec Vector Database

All benchmark embeddings are stored in a Zvec collection for efficient nearest-neighbor retrieval.

Current vector database:

```text
Number of samples: 2,500
Embedding dimension: 1,024
Vector database: Zvec
Collection: academic_collection
```

### Metadata-Aware Reranking

After vector retrieval, EvalAgent reranks candidates using:

- Semantic similarity
- Dataset preference
- Domain matching
- Task matching
- Ability matching
- Difficulty preference

The final score combines vector similarity with structured requirement matching.

### Evaluation Coverage Analysis

The final output includes:

- Recommended datasets
- Dataset proportions
- Average similarity
- Best similarity
- Domain coverage
- Task coverage
- Ability coverage
- Difficulty distribution
- Per-sample recommendation reasons

---

## 4. Integrated Benchmarks

| Dataset | Samples | Primary Focus |
|---|---:|---|
| QCalEval | 500 | Quantum experiment plots and calibration reasoning |
| ScienceQA | 500 | Science education and visual question answering |
| ChartQA | 500 | Chart understanding and numerical question answering |
| PlotQA | 500 | Scientific plot understanding and data extraction |
| MathVista | 500 | Mathematical and visual reasoning |
| **Total** | **2,500** | Multimodal academic evaluation |

---

## 5. Repository Structure

```text
academic_benchmark_retrieval/
├── configs/
│   └── evalagent/
│       ├── label_mapping.json
│       └── profile_schema.json
│
├── data/
│   ├── processed/
│   │   └── academic_5_merged.jsonl
│   │
│   ├── vector_db/
│   │   ├── academic_5_embeddings.npy
│   │   ├── academic_5_metadata.jsonl
│   │   └── academic_5_zvec/
│   │
│   └── evaluation_runs/
│       └── .gitkeep
│
├── reports/
│   ├── metadata_label_inventory.json
│   └── metadata_label_inventory.txt
│
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
│   │
│   └── evalagent/
│       ├── 00_export_metadata_labels.py
│       ├── 00_validate_label_mapping.py
│       ├── 01_parse_requirement.py
│       ├── 02_build_retrieval_query.py
│       └── 03_run_evalagent.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 6. Installation

### Clone the repository

```bash
git clone https://github.com/yikeh-17/academic_benchmark_retrieval.git
cd academic_benchmark_retrieval
```

### Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
numpy
requests
tqdm
zvec
```

---

## 7. API Configuration

The project uses the Alibaba Cloud DashScope API for Qwen3-VL embedding generation.

Set the API key as an environment variable:

```bash
export DASHSCOPE_API_KEY="your_api_key"
```

Confirm that the key is available:

```bash
if [ -n "$DASHSCOPE_API_KEY" ]; then
  echo "DASHSCOPE_API_KEY is set"
else
  echo "DASHSCOPE_API_KEY is NOT set"
fi
```

Do not store the API key directly in source code.

---

## 8. Data Preparation Pipeline

The five datasets are processed independently before being merged.

```bash
python scripts/01_prepare_qcaleval.py
python scripts/02_prepare_scienceqa.py
python scripts/03_prepare_chartqa.py
python scripts/04_prepare_plotqa.py
python scripts/05_prepare_mathvista.py
```

Merge all processed samples:

```bash
python scripts/06_merge_academic_5.py
```

Expected output:

```text
data/processed/academic_5_merged.jsonl
```

Expected number of records:

```text
2,500
```

Check the number of samples:

```bash
wc -l data/processed/academic_5_merged.jsonl
```

---

## 9. Build Multimodal Embeddings

Generate embeddings using Qwen3-VL:

```bash
python scripts/07_build_qwenvl_embeddings.py
```

Expected outputs:

```text
data/vector_db/academic_5_embeddings.npy
data/vector_db/academic_5_metadata.jsonl
```

Expected embedding shape:

```text
(2500, 1024)
```

Check the embedding matrix:

```bash
python - <<'PY'
import numpy as np

vectors = np.load(
    "data/vector_db/academic_5_embeddings.npy"
)

print("Shape:", vectors.shape)
print("Dtype:", vectors.dtype)
PY
```

---

## 10. Build the Zvec Index

Build the Academic-5 Zvec collection:

```bash
python scripts/08_build_zvec_index.py
```

Expected collection path:

```text
data/vector_db/academic_5_zvec/academic_collection
```

---

## 11. Basic Vector Search

The original retrieval system can be used directly.

### Text-only search

```bash
python scripts/09_search_zvec.py \
  --query "scientific plots, oscillation counting and trend analysis" \
  --topk 10
```

### Text-and-image search

```bash
python scripts/09_search_zvec.py \
  --query "find benchmarks requiring understanding of this plot" \
  --image /path/to/query_image.png \
  --topk 10
```

### Metadata filtering

```bash
python scripts/09_search_zvec.py \
  --query "quantum experiment trend analysis" \
  --dataset QCalEval \
  --domain quantum_physics \
  --task-type trend_analysis \
  --topk 10
```

---

## 12. Run EvalAgent

Run the complete evaluation recommendation workflow:

```bash
python scripts/evalagent/03_run_evalagent.py \
  --request-id demo_quantum \
  --requirement "我想评测一个用于量子实验图理解的多模态模型，重点关注趋势分析、参数提取和科学推理，希望使用中等到困难样本，预算20条。"
```

The workflow performs:

```text
Requirement parsing
→ Retrieval query construction
→ Query embedding generation
→ Zvec candidate retrieval
→ Metadata-aware reranking
→ Evaluation sample selection
→ Coverage analysis
→ JSON report generation
```

The result is saved to:

```text
data/evaluation_runs/demo_quantum_recommendation.json
```

---

## 13. Example Parsed Requirement

Input:

```text
我想评测一个用于量子实验图理解的多模态模型，
重点关注趋势分析、参数提取和科学推理，
希望使用中等到困难样本，预算20条。
```

Structured profile:

```json
{
  "model_name": null,
  "target_domains": [
    "quantum_physics"
  ],
  "target_tasks": [
    "scientific_reasoning",
    "trend_analysis",
    "parameter_extraction"
  ],
  "target_abilities": [
    "scientific_reasoning",
    "trend_analysis",
    "parameter_extraction"
  ],
  "preferred_difficulties": [
    "medium",
    "hard"
  ],
  "preferred_datasets": [],
  "evaluation_budget": 20
}
```

---

## 14. Example Recommendation Result

For a quantum experiment evaluation request, EvalAgent selected:

```text
Recommended samples: 20
Primary dataset: QCalEval
Primary domain: quantum_physics
```

Task distribution:

```text
scientific_reasoning: 12
trend_analysis: 7
parameter_extraction: 1
```

Difficulty distribution after metadata correction:

```text
medium: 17
hard: 2
easy: 1
```

The result demonstrates that the system can identify domain-specific samples while respecting task and difficulty preferences.

---

## 15. Output Format

The generated recommendation file contains:

```json
{
  "evalagent_version": "phase1_v1",
  "request_id": "demo_quantum",
  "raw_requirement": "",
  "query_profile": {},
  "retrieval_query": "",
  "retrieval": {
    "embedding_model": "qwen3-vl-embedding",
    "embedding_dimension": 1024,
    "candidate_k": 60,
    "returned_candidates": 60,
    "selected_samples": 20
  },
  "recommended_datasets": {},
  "coverage_summary": {
    "datasets": {},
    "domains": {},
    "tasks": {},
    "abilities": {},
    "difficulties": {}
  },
  "recommended_samples": [],
  "recommendation_summary": ""
}
```

Each recommended sample contains:

```json
{
  "rank": 1,
  "sample_id": "",
  "dataset_name": "",
  "question": "",
  "image_path": "",
  "task_type": "",
  "domain": "",
  "difficulty": "",
  "ability": [],
  "distance": 0.0,
  "similarity_score": 0.0,
  "metadata_match_score": 0.0,
  "final_score": 0.0,
  "recommendation_reason": ""
}
```

---

## 16. Phase 1 Status

The current implementation supports:

- [x] Five-dataset benchmark integration
- [x] Unified sample format
- [x] Multimodal embedding generation
- [x] Zvec vector indexing
- [x] Text and image retrieval
- [x] Natural-language requirement parsing
- [x] Structured evaluation profiles
- [x] Metadata-aware reranking
- [x] Difficulty preference handling
- [x] Dataset recommendation statistics
- [x] Capability coverage analysis
- [x] JSON evaluation recommendation reports

---

## 17. Current Limitations

The current Phase 1 system has several limitations:

1. Requirement parsing is rule-based rather than LLM-based.
2. Difficulty is treated as a preference rather than a strict constraint.
3. Sample selection does not yet guarantee balanced task allocation.
4. Some datasets contain incomplete metadata.
5. EvalAgent recommends evaluation samples but does not yet execute the target model.
6. Automatic scoring and model capability diagnosis are not yet implemented.

---

## 18. Planned Development

Future work will extend EvalAgent from benchmark recommendation to automated model evaluation.

Planned components include:

### Coverage-Aware Sample Selection

Allocate evaluation samples across target tasks and abilities instead of relying only on ranking scores.

### Hard and Soft Constraints

Distinguish between:

```text
Prefer medium and hard samples
```

and:

```text
Use only medium and hard samples
```

### Target Model Execution

Automatically call a user-provided multimodal model API on selected benchmark samples.

### Automatic Evaluation

Support:

- Exact-match scoring
- Multiple-choice accuracy
- Numerical tolerance
- Semantic matching
- LLM-as-a-judge evaluation

### Model Capability Profiling

Generate summaries of:

- Strong capabilities
- Weak capabilities
- Dataset-level performance
- Task-level performance
- Failure patterns
- Representative error cases

### Automated Evaluation Reports

Generate complete Markdown or web-based reports containing:

- Evaluation objectives
- Sample composition
- Dataset distribution
- Task and ability coverage
- Model performance
- Failure analysis
- Improvement recommendations

---

## 19. Project Positioning

The original system provides multimodal benchmark retrieval.

EvalAgent extends this retrieval foundation into an evaluation planning agent by adding:

```text
Natural-language understanding
+ Structured requirement extraction
+ Metadata-aware decision logic
+ Evaluation coverage analysis
+ Explainable recommendation output
```

The long-term objective is to build an end-to-end system for:

```text
Evaluation requirement
→ Benchmark planning
→ Model execution
→ Automatic scoring
→ Capability diagnosis
→ Evaluation report
```

---

## 20. License

This repository is intended for academic research and experimental development.

The original licenses and usage conditions of QCalEval, ScienceQA, ChartQA, PlotQA, MathVista, Qwen, and Zvec should be followed when using the corresponding datasets, models, and software.