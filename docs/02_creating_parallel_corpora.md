# Workflow 2: Creating Parallel Corpora

This workflow processes PDF textbooks through three steps to create aligned parallel documents across English, Hindi, and Tamil.

**Pre-computed results for NCERT Class 6 Science** are available in `parallel_corpora/Class_6-Science/`. You can skip directly to [Workflow 3](03_cross_lingual_retrieval.md).

## Required Directory Structure

Place your PDF textbooks in the `books/` directory:

```
books/
├── Class_6-Science-English/
│   ├── Chapter_1.pdf
│   └── ...
├── Class_6-Science-Hindi/
│   ├── Chapter_1.pdf
│   └── ...
└── Class_6-Science-Tamil/
    ├── Chapter_1.pdf
    └── ...
```

- Each language needs its own subdirectory named `Class_6-Science-{Language}`
- Chapter PDFs must be named `Chapter_{N}.pdf` (N = 1–12)

## Pipeline Steps

The workflow is implemented in `src/create_parallel_corpora.py` and executes three steps:

| Step | Description | Output | Code |
|------|-------------|--------|------|
| **1. PDF Scanning** | Extracts document structure using Docling OCR + Tesseract. Runs two scans per chapter: full original and text-only (for improved accuracy) | `outputs_split/{Language}/Chapter_{N}/parsed_original.json`, `parsed_text.json` | `src/processing/docling_pipeline.py`, `src/processing/pdf_utils.py` |
| **2. Document Merging** | Merges the two OCR scans into a single improved document per language | `parallel_corpora/Class_6-Science/Chapter_{N}/merged_{language}.json` | `src/merge_document_scans.py`, `src/merging/` |
| **3. Hindi-English Alignment** | Aligns English ↔ Hindi text blocks using a multilingual bi-encoder for candidate retrieval and a cross-encoder for reranking | `parallel_corpora/Class_6-Science/Chapter_{N}/eng_to_hin.json`, `hin_to_eng.json` | `src/matching.py` |

Each step automatically skips chapters that have already been processed.

## Running

```bash
python src/create_parallel_corpora.py
```

## How Alignment Works

1. **Chunking** — Long text blocks are split into fixed-size character chunks (400 chars) so embeddings can handle them
2. **Bi-encoder retrieval** — Each source chunk is embedded with `intfloat/multilingual-e5-base` and the top-5 similar target chunks are retrieved via cosine similarity
3. **Cross-encoder reranking** — Candidate pairs are scored by `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` to select the best match
4. **Directional alignment** — The process runs in both directions (English→Hindi, Hindi→English) to produce two mapping dictionaries

## Configuration

| Parameter | Location | Description |
|-----------|----------|-------------|
| `LANGUAGES` | `src/create_parallel_corpora.py` | Language name → code mapping |
| `CHAPTERS` | `src/create_parallel_corpora.py` | Chapter numbers to process (default 1–12) |
| `TOP_K` | `src/matching.py` | Candidates retrieved by bi-encoder before reranking (default 5) |
| `CHUNK_SIZE` | `src/matching.py` | Character chunk size for long blocks (default 400) |
| `BI_ENCODER_MODEL` | `src/matching.py` | Bi-encoder model name |
| `CROSS_ENCODER_MODEL` | `src/matching.py` | Cross-encoder model name |
| `min_charspan` | `src/merge_document_scans.py` | Minimum characters for a text block to be kept during merging |

## Key Files

| File | Role |
|------|------|
| `src/create_parallel_corpora.py` | Main orchestrator — runs all 3 steps |
| `src/processing/docling_pipeline.py` | Docling OCR pipeline |
| `src/processing/pdf_utils.py` | Text-layer extraction, watermark removal |
| `src/merge_document_scans.py` | Merges two scans into one improved document |
| `src/merging/merge_helpers.py` | Low-level merge logic (text items, tables, captions) |
| `src/merging/filters.py` | Pre-merge cleanup (headers, footers, short items) |
| `src/matching.py` | Bi-encoder + cross-encoder alignment |
