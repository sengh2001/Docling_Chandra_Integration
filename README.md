# Parallel Corpora for Multilingual Textbooks

Tools for creating aligned parallel corpora from multilingual textbook PDFs and performing cross-lingual retrieval. Supports English, Hindi, and Tamil textbooks.

## Installation

```bash
pip install -r requirements.txt
```

**System dependency:** [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) must be installed on your system.

**ML models (downloaded automatically on first run):**

| Model | Used by | Purpose |
|-------|---------|---------|
| `intfloat/multilingual-e5-base` | Workflow 2 (alignment) | Bi-encoder for candidate retrieval |
| `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Workflow 2 (alignment) | Cross-encoder for reranking |
| `krutrim-ai-labs/vyakyarth` | Workflow 3 (retrieval) | Multilingual sentence embeddings |

To pre-download models before running the pipeline:

```python
from sentence_transformers import SentenceTransformer, CrossEncoder

SentenceTransformer("intfloat/multilingual-e5-base")
CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
SentenceTransformer("krutrim-ai-labs/vyakyarth")
```

## Quick Start: Full Parallel Corpora Pipeline

To produce aligned parallel corpora from scratch, run **Workflows 2 → 3 → 4** in sequence:

```bash
# Step 1: Scan PDFs, merge OCR outputs, and align English ↔ Hindi
python src/create_parallel_corpora.py

# Step 2: Pre-compute sentence embeddings for retrieval
python dev_scripts/process_embeddings.py

# Step 3: Export parallel text blocks as JSONL
python dev_scripts/create_parallel_text_blocks_for_textbooks.py
```

Pre-computed results for **NCERT Class 6 Science** are available in `parallel_corpora/Class_6-Science/` — you can skip straight to Workflow 3.

## Workflows

| # | Workflow | Description | Required for parallel corpora? |
|---|----------|-------------|-------------------------------|
| 1 | [PDF Extraction & Annotation](docs/01_pdf_extraction_and_annotation.md) | Dual-scan PDFs, merge, and visualise OCR bounding boxes | No — for testing/tuning only |
| 2 | [Creating Parallel Corpora](docs/02_creating_parallel_corpora.md) | Scan → merge → align PDFs into parallel documents | **Yes** |
| 3 | [Cross-Lingual Retrieval](docs/03_cross_lingual_retrieval.md) | Semantic search across the multilingual corpus with parallel content retrieval | **Yes** |
| 4 | [Creating Parallel Text Blocks](docs/04_creating_parallel_text_blocks.md) | Export aligned text blocks as JSONL for downstream use | **Yes** |

## Directory Overview

| Directory | Description |
|-----------|-------------|
| `books/` | Input PDF textbooks organised by language |
| `outputs_split/` | Intermediate OCR outputs from Workflow 2 Step 1 |
| `parallel_corpora/` | Final aligned documents and match dictionaries |
| `outputs_embeddings/` | Cached sentence embeddings for retrieval |
| `src/` | Core library code |
| `src/processing/` | PDF processing and OCR pipeline |
| `src/merging/` | Document merge helpers and filters |
| `src/retreival/` | Embedding generation and similarity search |
| `dev_scripts/` | Standalone utility scripts |
| `notebooks/` | Exploratory and test notebooks |
