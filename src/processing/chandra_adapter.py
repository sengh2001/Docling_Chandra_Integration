"""
chandra_adapter.py — Chandra OCR 2 to DoclingDocument Translation Layer
========================================================================

Reads a Chandra OCR `_result.json` file and converts it into a fully
valid DoclingDocument (same schema as produced by Docling's Tesseract
pipeline).  The output JSON can be dropped directly into the
`parallel_corpora/` or `outputs_split/` directories so that Step 3
(Hindi-English alignment) and the retrieval workflows work unchanged.

Key design decisions
---------------------
* Spatial bounding boxes (data-bbox) are preserved 1-to-1 as
  ProvenanceItem records, so Tamil bounding-box anchoring in
  alignment.py continues to work.
* Page-Header / Page-Footer items are marked content_layer="furniture"
  (same as Docling does) so the alignment models skip them.
* Tables are parsed from the HTML <table> tag embedded inside each
  <div data-label="Table">. num_rows / num_cols are derived correctly.
* Column-header cells (<th>) are tagged with column_header=True.

Usage
------
    python src/processing/chandra_adapter.py \\
        <input_chandra_result.json> \\
        <output_docling.json> \\
        [optional_pdf_filename]
"""

import json
import sys
import fitz
from pathlib import Path
from bs4 import BeautifulSoup, Tag

from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.document import (
    ContentLayer,
    DoclingDocument,
    ProvenanceItem,
    TableCell,
    TableData,
)

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

_CHANDRA_TO_DOCLING: dict[str, DocItemLabel] = {
    "Page-Header":    DocItemLabel.PAGE_HEADER,
    "Page-Footer":    DocItemLabel.PAGE_FOOTER,
    "Section-Header": DocItemLabel.SECTION_HEADER,
    "Text":           DocItemLabel.TEXT,
    "Caption":        DocItemLabel.CAPTION,
    "Table":          DocItemLabel.TABLE,
    "Image":          DocItemLabel.PICTURE,
    "Equation":       DocItemLabel.FORMULA,
    "List":           DocItemLabel.LIST_ITEM,
}

# Labels that should live in the "furniture" content layer (not body text)
_FURNITURE_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}


def _get_label(label_str: str) -> DocItemLabel:
    return _CHANDRA_TO_DOCLING.get(label_str, DocItemLabel.TEXT)


# ---------------------------------------------------------------------------
# BoundingBox parsing
# ---------------------------------------------------------------------------

def _parse_bbox(bbox_str: str) -> BoundingBox:
    """Parse Chandra's 'l t r b' data-bbox string into a Docling BoundingBox."""
    parts = bbox_str.strip().split()
    if len(parts) != 4:
        raise ValueError(f"Unexpected bbox format: '{bbox_str}'")
    l, t, r, b = map(float, parts)
    return BoundingBox(l=l, t=t, r=r, b=b, coord_origin=CoordOrigin.TOPLEFT)


# ---------------------------------------------------------------------------
# Table parsing helper
# ---------------------------------------------------------------------------

def _build_table_data(table_tag: Tag) -> TableData:
    """Parse an HTML <table> tag into a Docling TableData object."""
    rows = table_tag.find_all("tr")
    num_rows = len(rows)
    num_cols = 0
    cells: list[TableCell] = []

    for r_idx, row in enumerate(rows):
        col_tags = row.find_all(["td", "th"])
        if len(col_tags) > num_cols:
            num_cols = len(col_tags)
        for c_idx, cell in enumerate(col_tags):
            row_span = int(cell.get("rowspan", 1))
            col_span = int(cell.get("colspan", 1))
            is_header = cell.name == "th"
            tc = TableCell(
                text=cell.get_text(separator=" ", strip=True),
                start_row_offset_idx=r_idx,
                end_row_offset_idx=r_idx + row_span - 1,
                start_col_offset_idx=c_idx,
                end_col_offset_idx=c_idx + col_span - 1,
                column_header=is_header,
                row_header=False,
            )
            cells.append(tc)

    return TableData(table_cells=cells, num_rows=num_rows, num_cols=num_cols)


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def convert_chandra_to_docling(
    chandra_json_path: str,
    output_docling_json_path: str,
    pdf_name: str = None,
    pdf_path: str = None,
) -> DoclingDocument:
    """
    Convert a Chandra OCR `_result.json` into a DoclingDocument JSON.

    Parameters
    ----------
    chandra_json_path : str
        Path to Chandra's `_result.json` output.
    output_docling_json_path : str
        Destination path for the DoclingDocument JSON.
    pdf_name : str
        Display name set in the DoclingDocument origin.

    Returns
    -------
    DoclingDocument
    """
    with open(chandra_json_path, "r", encoding="utf-8") as f:
        chandra_data = json.load(f)

    if not pdf_name:
        pdf_name = Path(pdf_path).name if pdf_path else Path(chandra_json_path).stem

    doc = DoclingDocument(name=pdf_name)
    
    # Open PDF if provided to get physical dimensions
    pdf_doc = fitz.open(pdf_path) if pdf_path else None
    
    # Chandra uses a fixed normalized grid for coordinates
    CHANDRA_GRID = 1000.0

    for page_data in chandra_data:
        page_num: int = page_data["page"]
        raw_html: str = page_data.get("raw", "")

        if not raw_html.strip():
            continue

        soup = BeautifulSoup(raw_html, "html.parser")
        divs = soup.find_all("div", recursive=False)

        # ------------------------------------------------------------------
        # Determine Page Dimensions and Scaling
        # ------------------------------------------------------------------
        if pdf_doc and page_num <= pdf_doc.page_count:
            pdf_page = pdf_doc.load_page(page_num - 1)
            pdf_w = pdf_page.rect.width
            pdf_h = pdf_page.rect.height
        else:
            # Fallback to A4 points if no PDF provided
            pdf_w, pdf_h = 595.0, 841.0
            
        doc.add_page(page_no=page_num, size=Size(width=pdf_w, height=pdf_h))
        
        # Scaling factors from 1000-grid to PDF points
        x_scale = pdf_w / CHANDRA_GRID
        y_scale = pdf_h / CHANDRA_GRID

        for div in divs:
            bbox_str: str = div.get("data-bbox", "")
            label_str: str = div.get("data-label", "Text")

            if not bbox_str:
                continue

            try:
                # Parse raw 1000-grid coords
                parts = list(map(float, bbox_str.strip().split()))
                if len(parts) != 4:
                    raise ValueError(f"Invalid bbox: {bbox_str}")
                
                # Scale to PDF points
                l, t, r, b = parts
                bbox = BoundingBox(
                    l=l * x_scale,
                    t=t * y_scale,
                    r=r * x_scale,
                    b=b * y_scale,
                    coord_origin=CoordOrigin.TOPLEFT
                )
            except Exception as exc:
                print(f"[WARN] page {page_num}: {exc} — skipping element")
                continue

            doc_label = _get_label(label_str)
            text = div.get_text(separator=" ", strip=True)
            char_len = len(text)

            prov = ProvenanceItem(
                page_no=page_num,
                bbox=bbox,
                charspan=(0, char_len),
            )

            # Furniture layer for headers / footers
            content_layer = (
                ContentLayer.FURNITURE
                if doc_label in _FURNITURE_LABELS
                else ContentLayer.BODY
            )

            # ------------------------------------------------------------------
            # TABLE
            # ------------------------------------------------------------------
            if doc_label == DocItemLabel.TABLE:
                table_tag = div.find("table")
                if not table_tag:
                    # No <table> inside — treat as plain text
                    doc.add_text(
                        label=DocItemLabel.TEXT,
                        text=text,
                        prov=prov,
                        content_layer=content_layer,
                    )
                    continue

                table_data = _build_table_data(table_tag)
                doc.add_table(
                    data=table_data,
                    label=DocItemLabel.TABLE,
                    prov=prov,
                    content_layer=content_layer,
                )

            # ------------------------------------------------------------------
            # PICTURE / IMAGE
            # ------------------------------------------------------------------
            elif doc_label == DocItemLabel.PICTURE:
                doc.add_picture(prov=prov, content_layer=content_layer)

            # ------------------------------------------------------------------
            # ALL TEXT-LIKE ELEMENTS (Text, Section-Header, Page-Header, etc.)
            # ------------------------------------------------------------------
            else:
                doc.add_text(
                    label=doc_label,
                    text=text,
                    prov=prov,
                    content_layer=content_layer,
                )

    # ─── Save standard ASCII-safe JSON ───────────────────
    doc.save_as_json(output_docling_json_path)
    
    # ─── Save Human-Readable JSON (Hindi/chars) ─────────
    readable_path = Path(output_docling_json_path).with_suffix("").as_posix() + "_readable.json"
    with open(readable_path, "w", encoding="utf-8") as f:
        json.dump(doc.export_to_dict(), f, ensure_ascii=False, indent=2)

    print(
        f"[OK] Converted '{chandra_json_path}' -> '{output_docling_json_path}'\n"
        f"     (Also created readable version: '{Path(readable_path).name}')\n"
        f"     Counts: {len(doc.texts)} texts, {len(doc.tables)} tables, {len(doc.pictures)} pictures"
    )
    return doc


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python chandra_adapter.py "
            "<input_chandra_result.json> <output_docling.json> [pdf_name]"
        )
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    
    # Optional arguments
    pdf_name = None
    pdf_path = None
    
    if len(sys.argv) > 3:
        # Check if 3rd arg is a PDF path or a name
        arg3 = sys.argv[3]
        if arg3.lower().endswith(".pdf"):
            pdf_path = arg3
        else:
            pdf_name = arg3
            
    if len(sys.argv) > 4:
        pdf_path = sys.argv[4]

    try:
        convert_chandra_to_docling(
            chandra_json_path=in_path, 
            output_docling_json_path=out_path, 
            pdf_name=pdf_name,
            pdf_path=pdf_path
        )
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
