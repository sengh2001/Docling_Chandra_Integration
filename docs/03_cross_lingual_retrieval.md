# Workflow 3: Cross-Lingual Retrieval

> **Prerequisite:** [Workflow 2](02_creating_parallel_corpora.md) must be completed first to generate the parallel corpora and alignment files.

This workflow enables semantic search across the multilingual corpus and retrieves parallel content in all three languages.

For a complete working example, see `aligned_corpora.ipynb`.

## Key Functions

### 1. `retrieve_top_k_matches`

Performs semantic search to find the most relevant document items for a query.

```python
from src.retrieve_best_match import retrieve_top_k_matches

queries = {
    1: {
        "query": "high calorie food",
        "language": "eng"  # "eng", "hi", or "ta"
    }
}

top_k_matches = retrieve_top_k_matches(queries, k=3)
```

Embeddings are cached to `outputs_embeddings/` and computed on first run using the `krutrim-ai-labs/vyakyarth` model.

### 2. `get_parallel_data_for_matches`

Retrieves the aligned parallel content (main text, before/after context) for each match across all three languages.

```python
from src.alignment import get_parallel_data_for_matches, format_match_as_text

parallel_docs = {}
parallel_data, parallel_docs = get_parallel_data_for_matches(
    top_k_matches, 
    initialized_docs=parallel_docs
)
```

### 3. `format_match_as_text`

Formats the retrieved content as a readable string.

```python
for query_id, matches in parallel_data.items():
    for match_id, match_data in matches["top_k_matches"].items():
        print(f"Similarity: {match_data['similarity']:.3f}")
        for language in ["eng", "hi", "ta"]:
            print(f"{language}: {format_match_as_text(match_data, language)}")
```

## Output Structure

Each match returns parallel data in this format:

```python
{
    "eng": {"main": "...", "before": "...", "after": "..."},
    "hi":  {"main": "...", "before": "...", "after": "..."},
    "ta":  {"main": "...", "before": "...", "after": "..."}
}
```

- **main**: The matched text block
- **before**: The preceding sibling text block (context)
- **after**: The succeeding sibling text block (context)

## Alignment Strategy

| Source → Target | Method |
|-----------------|--------|
| English → Hindi | Precomputed match dictionary (`eng_to_hin.json`) |
| Hindi → English | Precomputed match dictionary (`hin_to_eng.json`) |
| English ↔ Tamil | Bounding box alignment (same page position) |
| Hindi → Tamil | Routed through English (Hindi→Eng→Tamil) |
| Tamil → Hindi | Routed through English (Tamil→Eng→Hindi) |

## Configuration

| Parameter | Location | Description |
|-----------|----------|-------------|
| `k` | `retrieve_top_k_matches()` | Number of top results per query |
| `embeddings_dir` | `retrieve_top_k_matches()` | Directory for cached embeddings |
| Embedding model | `src/retrieve_best_match.py` | `krutrim-ai-labs/vyakyarth` (multilingual sentence transformer) |

## Pre-computing Embeddings

To pre-compute embeddings for all languages (instead of on-the-fly):

```bash
python dev_scripts/process_embeddings.py
```

## Key Files

| File | Role |
|------|------|
| `src/retrieve_best_match.py` | Semantic search with embedding caching |
| `src/alignment.py` | `ParallelAlignedDocument` class and parallel data retrieval |
| `src/retreival/preprocess.py` | Embedding computation from document JSONs |
| `src/retreival/rets.py` | Top-k similarity search |
| `aligned_corpora.ipynb` | Interactive example notebook |
