import json
import fitz
from pathlib import Path
from bs4 import BeautifulSoup, Tag
from typing import Dict, List, Optional, Tuple

from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.labels import DocItemLabel
from docling_core.types.doc.document import (
    ContentLayer,
    DoclingDocument,
    ProvenanceItem,
    TableCell,
    TableData,
)

class ChandraAdapter:
    """
    Adapter for converting Chandra OCR 2 output to DoclingDocument.
    Handles coordinate scaling from the 1000-grid to physical PDF points.
    """
    
    CHANDRA_GRID = 1000.0
    
    LABEL_MAPPING: Dict[str, DocItemLabel] = {
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
    
    FURNITURE_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}

    def __init__(self, pdf_path: Optional[str] = None):
        """
        Initialize the adapter.
        
        Args:
            pdf_path: Optional path to the source PDF to use for physical coordinate scaling.
        """
        self.pdf_path = pdf_path
        self.doc: Optional[DoclingDocument] = None
        self.pdf_doc = fitz.open(pdf_path) if pdf_path else None

    def _get_docling_label(self, label_str: str) -> DocItemLabel:
        """Map Chandra string label to DocItemLabel."""
        return self.LABEL_MAPPING.get(label_str, DocItemLabel.TEXT)

    def _build_table_data(self, table_tag: Tag) -> TableData:
        """Parse an HTML <table> tag into a Docling TableData object."""
        rows = table_tag.find_all("tr")
        num_rows = len(rows)
        num_cols = 0
        cells: List[TableCell] = []

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

    def convert(self, chandra_json_path: str, document_name: Optional[str] = None) -> DoclingDocument:
        """
        Main conversion entry point.
        
        Args:
            chandra_json_path: Path to the Chandra OCR result JSON.
            document_name: Optional name for the output document.
            
        Returns:
            A populated DoclingDocument.
        """
        with open(chandra_json_path, "r", encoding="utf-8") as f:
            chandra_data = json.load(f)

        name = document_name or Path(chandra_json_path).stem
        self.doc = DoclingDocument(name=name)

        for page_data in chandra_data:
            page_num = page_data["page"]
            raw_html = page_data.get("raw", "")
            if not raw_html.strip():
                continue

            # Determine Physical Dimensions
            if self.pdf_doc and page_num <= self.pdf_doc.page_count:
                pdf_page = self.pdf_doc.load_page(page_num - 1)
                pdf_w = pdf_page.rect.width
                pdf_h = pdf_page.rect.height
            else:
                pdf_w, pdf_h = 595.0, 841.0 # Fallback A4

            self.doc.add_page(page_no=page_num, size=Size(width=pdf_w, height=pdf_h))
            
            x_scale = pdf_w / self.CHANDRA_GRID
            y_scale = pdf_h / self.CHANDRA_GRID

            # Parse Page Content
            soup = BeautifulSoup(raw_html, "html.parser")
            elements = soup.find_all("div", recursive=False)

            for el in elements:
                bbox_str = el.get("data-bbox", "")
                label_str = el.get("data-label", "Text")
                if not bbox_str:
                    continue

                try:
                    l, t, r, b = map(float, bbox_str.strip().split())
                    bbox = BoundingBox(
                        l=l * x_scale,
                        t=t * y_scale,
                        r=r * x_scale,
                        b=b * y_scale,
                        coord_origin=CoordOrigin.TOPLEFT
                    )
                except Exception:
                    continue

                doc_label = self._get_docling_label(label_str)
                text = el.get_text(separator=" ", strip=True)
                prov = ProvenanceItem(
                    page_no=page_num,
                    bbox=bbox,
                    charspan=(0, len(text)),
                )
                content_layer = (
                    ContentLayer.FURNITURE if doc_label in self.FURNITURE_LABELS else ContentLayer.BODY
                )

                if doc_label == DocItemLabel.TABLE:
                    table_tag = el.find("table")
                    if table_tag:
                        table_data = self._build_table_data(table_tag)
                        self.doc.add_table(data=table_data, label=DocItemLabel.TABLE, prov=prov, content_layer=content_layer)
                        continue
                
                if doc_label == DocItemLabel.PICTURE:
                    self.doc.add_picture(prov=prov, content_layer=content_layer)
                else:
                    self.doc.add_text(label=doc_label, text=text, prov=prov, content_layer=content_layer)

        return self.doc

    def save(self, output_path: str, ensure_ascii: bool = True):
        """Save the document to JSON."""
        if not self.doc:
            raise RuntimeError("No document has been converted yet.")
        
        # Standard Docling save
        self.doc.save_as_json(output_path)
        
        # Human-readable save (if not ascii)
        if not ensure_ascii:
            readable_path = Path(output_path).with_suffix("").as_posix() + "_readable.json"
            with open(readable_path, "w", encoding="utf-8") as f:
                json.dump(self.doc.export_to_dict(), f, ensure_ascii=False, indent=2)

    def close(self):
        """Clean up resources."""
        if self.pdf_doc:
            self.pdf_doc.close()
