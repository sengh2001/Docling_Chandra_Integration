# Workflow 4: Creating Parallel Text Blocks

> **Prerequisite:** [Workflow 2](02_creating_parallel_corpora.md) and [Workflow 3](03_cross_lingual_retrieval.md) must be completed first.

This workflow iterates through every English text block in the parallel corpora and retrieves the corresponding Hindi and Tamil matches, producing JSONL files suitable for downstream tasks (training data, evaluation, etc.).

## What It Does

For each chapter:
1. Loads the `ParallelAlignedDocument` for that chapter
2. Iterates through all English text blocks
3. Retrieves the aligned Hindi and Tamil content (via dictionary lookup and bounding box matching)
4. Saves results as a JSONL file per chapter

## Running

```bash
python dev_scripts/create_parallel_text_blocks_for_textbooks.py
```

## Output Format

Each line in the output JSONL is a JSON object:

```json
{
    "id": "#/texts/42",
    "english": "Main: ... | Before: ... | After: ...",
    "hindi": "Main: ... | Before: ... | After: ...",
    "tamil": "Main: ... | Before: ... | After: ..."
}
```

Output files are saved to `parallel_corpora/Class_6-Science/Chapter_{N}/parallel_text_blocks.jsonl`.

## Configuration

| Parameter | Location | Description |
|-----------|----------|-------------|
| `CHAPTERS` | `dev_scripts/create_parallel_text_blocks_for_textbooks.py` | Chapter numbers to process (default 1–12) |
| `PARALLEL_CORPORA_DIR` | `dev_scripts/create_parallel_text_blocks_for_textbooks.py` | Base path for parallel corpora |

## Key Files

| File | Role |
|------|------|
| `dev_scripts/create_parallel_text_blocks_for_textbooks.py` | Main script |
| `src/alignment.py` | `ParallelAlignedDocument` class and `format_match_as_text` |
