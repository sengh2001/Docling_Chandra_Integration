# Deep Dive: Docling OCR Technical Architecture

Based on a detailed review of the Python source code (`src/` directory), here is a comprehensive breakdown of the project's technical architecture and how the workflows function at a code level.

## 1. Core Architectural Data Structure
The entire application centers around a single data standard: **`DoclingDocument`**. This is a structured document representation format from the `docling-core` library. 
Every stage of the pipeline inputs and outputs `DoclingDocument` objects (usually serialized to JSON). 

Within a `DoclingDocument`, text blocks, tables, images, and lists are conceptually organized as a **tree of `NodeItem` objects**. Every node has a unique `self_ref` (or [cref](file:///Users/arshsingh/WORK/DoclingOCR/docling_ocr_extraction/src/alignment.py#508-514)) string (e.g., `#/texts/42`), a `parent`, `children`, and most importantly, `prov` (provenance) data which contains its **bounding box ([bbox](file:///Users/arshsingh/WORK/DoclingOCR/docling_ocr_extraction/src/alignment.py#558-563))** coordinates on the physical PDF page.

## 2. Pipeline Modules & Data Flow

The project is structured into three main technical domains mapping to the workflows.

### A. Extraction & OCR Phase (`src/processing/`)
- **[docling_pipeline.py](file:///Users/arshsingh/WORK/DoclingOCR/docling_ocr_extraction/src/processing/docling_pipeline.py) (The Engine):** Wraps `docling.document_converter.DocumentConverter`. It configures a `PdfPipelineOptions` strictly to use the `TesseractCliOcrOptions`.
- **`pdf_utils.py` (The Pre-processor):** Handles raw PDF manipulation (using `PyMuPDF`/`fitz`). It handles removing watermarks and extracting text-only layers to prepare the dual scans.
- **Workflow:** Raw PDF ➔ `pdf_utils.py` ➔ [docling_pipeline.py](file:///Users/arshsingh/WORK/DoclingOCR/docling_ocr_extraction/src/processing/docling_pipeline.py) ➔ `DoclingDocument` JSON.

### B. Merging Phase ([src/merge_document_scans.py](file:///Users/arshsingh/WORK/DoclingOCR/docling_ocr_extraction/src/merge_document_scans.py) & `src/merging/`)
- **Coordinator:** [merge_documents(baseline_doc, text_only_doc)](file:///Users/arshsingh/WORK/DoclingOCR/docling_ocr_extraction/src/merge_document_scans.py#27-52)
- **How it works:** 
  1. It deep-copies the `baseline_doc` (the full PDF scan).
  2. Runs `filters.py` to aggressively remove headers, footers, and text errantly grouped inside pictures.
  3. Uses `merge_helpers.py` to iterate over all `NodeItems` in the `text_only_doc`. It does spatial bounding box comparisons (using Intersection over Union - IoU) to update or insert missing text blocks into the baseline copy.
- **Workflow:** `baseline.json` + `text_only.json` ➔ [merge_documents()](file:///Users/arshsingh/WORK/DoclingOCR/docling_ocr_extraction/src/merge_document_scans.py#27-52) ➔ `merged_{lang}.json`.

### C. Alignment Phase ([src/matching.py](file:///Users/arshsingh/WORK/DoclingOCR/docling_ocr_extraction/src/matching.py))
This is the machine-learning core for aligning English and Hindi.
- **Chunking Strategy:** It aggressively iterates over English and Hindi text blocks. Because transformer models have token limits, it breaks block texts into sequential 400-character chunks ([chunk_text()](file:///Users/arshsingh/WORK/DoclingOCR/docling_ocr_extraction/src/matching.py#49-64)).
- **Phase 1 (Bi-Encoder Retrieval):** Uses `SentenceTransformer` (`intfloat/multilingual-e5-base`). It converts all text chunks into vectors. For an English chunk, it runs cosine similarity (`util.semantic_search`) against the entire Hindi corpus to find the `TOP_K=5` matching Hindi candidates.
- **Phase 2 (Cross-Encoder Reranking):** Uses `CrossEncoder` (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`). It takes the English source alongside the 5 retrieved Hindi candidates and pairs them together deeply through the model to score and select the single best matching [id](file:///Users/arshsingh/WORK/DoclingOCR/docling_ocr_extraction/src/matching.py#102-135) via `argmax`.
- **Workflow:** English Blocks + Hindi Blocks ➔ Bi-Encoder ➔ Cross-Encoder ➔ `eng_to_hin.json` & `hin_to_eng.json` mappings.

### D. Multi-Language Retrieval Engine ([src/alignment.py](file:///Users/arshsingh/WORK/DoclingOCR/docling_ocr_extraction/src/alignment.py))
This module introduces the [ParallelAlignedDocument](file:///Users/arshsingh/WORK/DoclingOCR/docling_ocr_extraction/src/alignment.py#34-628) class, which holds three different `DoclingDocument` objects in memory simultaneously (English, Hindi, Tamil) alongside the match dictionaries.
- **Spatial vs Semantic Matching:**
  - English <-> Hindi requests are instantly resolved by simply looking up the [cref](file:///Users/arshsingh/WORK/DoclingOCR/docling_ocr_extraction/src/alignment.py#508-514) id in the generated match dictionary (`eng_to_hin.json`).
  - Tamil does not use the ML models. Instead, Tamil uses **Bounding Box Anchoring** ([_find_aligned_item](file:///Users/arshsingh/WORK/DoclingOCR/docling_ocr_extraction/src/alignment.py#534-557)). To find what Tamil text matches an English text, the system looks at the English text's coordinates on the page and scans the Tamil document for text intersecting that exact same geometry (using `intersection_over_union` or spatial distance formula on the `tcx/tcy` centroids).
- **Context Construction ([_get_context_siblings](file:///Users/arshsingh/WORK/DoclingOCR/docling_ocr_extraction/src/alignment.py#459-503)):** When rendering matching sections, the software uses the `NodeItem.parent` attribute to traverse the DOM tree, locate the target item's index inside its parent's children array, and pull the `-1` and `+1` sibling elements to generate the "before" and "after" surrounding context.

## 3. Storage and Caching
To maintain fast retrieval times across massive books:
- Document outputs are structurally localized (`outputs_split/Class_6-Science-{Lang}/Chapter_{N}/`).
- Search Embeddings (`src/retreival/preprocess.py`) use the Krutrim Vyakyarth model to calculate sentence embeddings of all text strings and dumps them out into flat NumPY (`.npz`) binaries for instantaneous RAM loading without model inference lag during searches.
