# 🚀 Docling-Chandra Plus: Unified OCR Pipeline

This package unifies **Chandra OCR 2** results with the **Docling** document schema. It automates coordinate scaling, structural conversion, and visual verification into a single, "plug and play" workflow.

---

## 🏗️ Quick Start for New Users

### 1. Clone the Repository
```bash
git clone https://github.com/sengh2001/Docling_Chandra_Integration.git
cd Docling_Chandra_Integration
```

### 2. Install Dependencies
```bash
pip install docling-core pymupdf beautifulsoup4 sentence-transformers torch
```

---

## 🛠️ Project Anatomy
- **`docling_chandra_plus/`**: The main Python package containing the Adapter, Visualizer, and Aligner.
- **`src/`**: Legacy scripts and core Docling processing logic.
- **`parallel_corpora/`**: Dedicated storage for aligned English-Regional textbooks.

---

## 🚀 Usage (CLI)

To process a single PDF and its corresponding Chandra OCR JSON, run the following command from the root directory:

```bash
python -m docling_chandra_plus.core <input_pdf> <chandra_json> --out <output_dir>
```

### Example:
```bash
python -m docling_chandra_plus.core \
    Chandra_OCR/OCR/pdfss/hindi_sample.pdf \
    Chandra_OCR/OCR/chandra_output/hindi_result.json \
    --out my_results
```

---

## 🐍 Usage (Python API)

You can easily integrate the pipeline into your own scripts:

```python
from docling_chandra_plus.core import ChandraPipeline

# Initialize the pipeline
pipeline = ChandraPipeline(output_dir="my_batch_outputs")

# Process a document
# This produces the JSON, Readable JSON, and Annotated PDF in one call
pipeline.process_doc(
    pdf_path="sample.pdf",
    chandra_json_path="sample_result.json",
    doc_name="Chapter_1_Hindi"
)
```

---

## 📑 Generated Artifacts

For every run, the pipeline generates three files:

1.  **`[name]_docling.json`**: The standard, ASCII-safe DoclingDocument JSON. This is ready for indexing or alignment.
2.  **`[name]_docling_readable.json`**: A human-readable JSON that preserves Hindi/Regional characters (UTF-8). Perfect for manual verification.
3.  **`[name]_annotated.pdf`**: A verification PDF with red bounding boxes overlaying the original document. If boxes align with text, your scaling is perfect.

---

## 🧠 Key Features
-   **Auto-Scaling**: Automatically maps Chandra's 1000-grid coordinates to the physical point-size (72 DPI) of your PDF.
-   **Structural Parsing**: Reconstructs hierarchical tables, headers, and lists from Chandra's HTML tags.
-   **Verification Layer**: Built-in `ChandraVisualizer` for immediate visual feedback on OCR accuracy.
