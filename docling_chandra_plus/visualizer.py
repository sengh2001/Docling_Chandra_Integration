import fitz
from typing import Union, List, Tuple
from docling_core.types.doc.document import DoclingDocument
from pathlib import Path

class ChandraVisualizer:
    """
    Visualizer for drawing DoclingDocument bounding boxes onto PDFs.
    Used for verifying Chandra OCR coordinate scaling.
    """

    COLORS = {
        "text": (1, 0, 0),    # Red
        "picture": (0, 0, 1), # Blue
        "table": (0, 1, 1),   # Cyan
    }

    def __init__(self, label_size: float = 10.0):
        self.label_size = label_size

    def annotate_document(
        self, 
        doc: DoclingDocument, 
        input_pdf_path: str, 
        output_pdf_path: str
    ):
        """
        Draw all bounding boxes from a DoclingDocument onto a PDF.
        """
        pdf_doc = fitz.open(input_pdf_path)

        # Draw regions by type
        self._draw_group(doc.texts, self.COLORS["text"], pdf_doc, doc)
        self._draw_group(doc.pictures, self.COLORS["picture"], pdf_doc, doc)
        self._draw_group(doc.tables, self.COLORS["table"], pdf_doc, doc)

        pdf_doc.save(output_pdf_path)
        pdf_doc.close()
        return output_pdf_path

    def _draw_group(self, items, color: Tuple[float, float, float], pdf_doc: fitz.Document, doc: DoclingDocument):
        """Internal helper to draw a group of specific doc items."""
        for item in items:
            if not item.prov:
                continue
                
            prov = item.prov[0]
            bbox = prov.bbox
            page_idx = prov.page_no - 1

            if page_idx < 0 or page_idx >= pdf_doc.page_count:
                continue

            page = pdf_doc.load_page(page_idx)
            
            # Coordinate scaling (Handling TOPLEFT vs BOTTOMLEFT)
            # Our Chandra adapter uses TOPLEFT normalized to PDF points.
            page_rect = page.rect
            
            x_scale, y_scale = 1.0, 1.0
            if prov.page_no in doc.pages:
                stored_size = doc.pages[prov.page_no].size
                if stored_size and stored_size.width > 0:
                    x_scale = page_rect.width / stored_size.width
                    y_scale = page_rect.height / stored_size.height

            if "TOPLEFT" in str(bbox.coord_origin):
                x0, y0 = bbox.l * x_scale, bbox.t * y_scale
                x1, y1 = bbox.r * x_scale, bbox.b * y_scale
            else:
                # Flip Y for BOTTOMLEFT (Standard PDF origin)
                x0, x1 = bbox.l, bbox.r
                y0 = page_rect.height - bbox.t
                y1 = page_rect.height - bbox.b

            rect = fitz.Rect(x0, y0, x1, y1)
            page.draw_rect(rect, color=color, width=1.0)
            
            # Add Label
            label_text = str(item.label)
            page.insert_text(
                fitz.Point(x0 + 1, y0 + self.label_size),
                label_text,
                fontsize=self.label_size,
                color=color,
                overlay=True
            )
