# Workflow 1: PDF Extraction & Annotation

> **Purpose:** Test and tune the Docling OCR pipeline on individual PDFs. This workflow is **not required** for creating parallel corpora — use it when you need to inspect or adjust extraction quality.

## What It Does

1. Runs a dual OCR scan on each chapter (full original + text-only) to produce two parsed documents
2. Merges the two scans into one improved document
3. Draws bounding boxes on the original PDF for visual inspection of the merged result

This mirrors what [Workflow 2](02_creating_parallel_corpora.md) does in Steps 1–2, but with annotated PDF output so you can visually verify quality.

## Steps

### Step 1: Dual Scan

Run `extract_structures_split.py` to extract both the original and text-only versions of each chapter:

```bash
python dev_scripts/extract_structures_split.py
```

Configure which languages and chapters to process by editing the constants at the top of the file:

```python
CHAPS = [12]                      # chapter numbers to process
LANGUAGES = {"Hindi": "hin"}      # language name → Tesseract code
```

**Output:** For each chapter, two JSON files in `outputs_split/`:
- `parsed_original.json` — OCR of the full PDF (with images, watermarks)
- `parsed_text.json` — OCR of the text-only version (images/watermarks stripped)

### Step 2: Merge & Annotate

Open `notebooks/test_merge.ipynb` to merge the two scans and produce annotated PDFs:

```python
from src.merge_document_scans import merge_documents
from src.bbox_draw import draw_bboxes_on_pdf
from docling_core.types.doc.document import DoclingDocument

full_doc = DoclingDocument.load_from_json("outputs_split/.../parsed_original.json")
text_doc = DoclingDocument.load_from_json("outputs_split/.../parsed_text.json")

merged_doc = merge_documents(full_doc, text_doc, verbose=False)
merged_doc.save_as_json("merged.json")

draw_bboxes_on_pdf(merged_doc, "books/.../Chapter_1.pdf", "merged_annotated.pdf")
```

The notebook includes a loop that processes all chapters and languages in one run.

**Output:** `merged_annotated.pdf` with colour-coded bounding boxes (**red** = text, **blue** = pictures, **cyan** = tables).

## Configuration

| Parameter | Location | Description |
|-----------|----------|-------------|
| `LANGUAGES` | `dev_scripts/extract_structures_split.py` | Language name → Tesseract code mapping |
| `CHAPS` | `dev_scripts/extract_structures_split.py` | Chapter numbers to process |
| `NUM_PROCESSES` | `dev_scripts/extract_structures_split.py` | Parallel workers (default 2) |
| `force_full_page_ocr` | `src/processing/docling_pipeline.py` | Force OCR even if layout detection succeeds |
| `min_charspan` | `src/merge_document_scans.py` | Minimum characters for a text block to be kept |

## Key Files

| File | Role |
|------|------|
| `dev_scripts/extract_structures_split.py` | Dual-scan extraction (original + text-only) |
| `notebooks/test_merge.ipynb` | Merge scans and produce annotated PDFs |
| `src/processing/docling_pipeline.py` | Docling OCR pipeline and converter factory |
| `src/processing/pdf_utils.py` | Text-layer extraction and watermark removal |
| `src/merge_document_scans.py` | Document merging logic |
| `src/bbox_draw.py` | PDF bounding box visualisation |
