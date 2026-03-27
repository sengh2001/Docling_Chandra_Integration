import fitz
from typing import Union
from docling_core.types.doc.document import DoclingDocument
import json
import sys


def draw_bboxes_on_pdf(
    doc: DoclingDocument,
    original_pdf: Union[str, bytes],  # local path or URL
    output_pdf: str = "annotated_output.pdf",
    label_size: float = 11,
):
    """
    Draw bounding boxes and labels on the original PDF.
    :param doc: DoclingDocument loaded by docling
    :param original_pdf: Path or URL to original PDF
    :param output_pdf: Where to save the annotated PDF
    :param box_color: RGB tuple for bbox
    :param label_color: RGB tuple for label text
    :param label_size: Font size
    """
    pdf_doc = fitz.open(original_pdf)

    # ─── Draw each text box ────────────────────────────────
    annotate_regions(doc.texts, (1, 0, 0), (1, 0, 0), label_size, pdf_doc, doc)
    annotate_regions(doc.pictures, (0, 0, 1), (0, 0, 1), label_size, pdf_doc, doc)
    annotate_regions(doc.tables, (0, 1, 1), (0, 1, 1), label_size, pdf_doc, doc)

    # ─── Save annotated PDF ───────────────────────────────
    pdf_doc.save(output_pdf)
    pdf_doc.close()

    print(f"Annotated PDF saved to: {output_pdf}")


def annotate_regions(doc_regions, box_color, label_color, label_size, pdf_doc, doc: DoclingDocument):
    """
    Takes doc regions given as bounding boxes and adds a box and label.
    Handles both TOPLEFT (Chandra/pixel) and BOTTOMLEFT (standard PDF/Docling)
    coordinate origins correctly.
    """
    for ti, t in enumerate(doc_regions):
        # bounding box from Docling
        bbox = t.prov[0].bbox
        page_index = t.prov[0].page_no - 1  # 0-based for PyMuPDF

        # ensure page exists
        if page_index < 0 or page_index >= pdf_doc.page_count:
            continue

        page = pdf_doc.load_page(page_index)
        page_rect = page.rect
        page_height = page_rect.height
        page_width = page_rect.width

        l = bbox.l
        r = bbox.r
        b = bbox.b
        t_ = bbox.t

        # Detect coordinate origin
        coord_origin = str(bbox.coord_origin)

        if "TOPLEFT" in coord_origin:
            # Our updated Chandra adapter now outputs coordinates directly
            # in PDF points (72 DPI). Thus, the scale factor is 1.0 if the
            # JSON and PDF are in sync.
            x_scale = 1.0
            y_scale = 1.0
            
            if t.prov[0].page_no in doc.pages:
                stored_size = doc.pages[t.prov[0].page_no].size
                if stored_size and stored_size.width > 0:
                    # If for some reason the PDF point sizes differ, this handles it
                    x_scale = page_width / stored_size.width
                    y_scale = page_height / stored_size.height

            x0 = l * x_scale
            x1 = r * x_scale
            y0 = t_ * y_scale   # already top-to-bottom
            y1 = b * y_scale
        else:
            # Standard BOTTOMLEFT origin (native Docling/Tesseract output).
            # PyMuPDF uses top-left, so we flip the Y axis.
            x0 = l
            x1 = r
            y0 = page_height - t_
            y1 = page_height - b

        rect = fitz.Rect(x0, y0, x1, y1)

        # draw rectangle
        page.draw_rect(
            rect,
            color=box_color,
            width=1.2,
        )

        # label text
        label = str(t.label)

        # position the text inside the box top-left corner
        text_point = fitz.Point(x0 + 1, y0 + label_size + 1)

        page.insert_text(
            text_point,
            label,
            fontsize=label_size,
            color=label_color,
            overlay=True,
        )


def main(json_file: str, pdf_file: str, out_file: str):
    doc = DoclingDocument.load_from_json(json_file)
    draw_bboxes_on_pdf(doc, pdf_file, out_file)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python bbox_draw.py <json_file> <pdf_file> <out_file>")
        sys.exit(1)

    json_file = sys.argv[1]
    pdf_file = sys.argv[2]
    out_file = sys.argv[3]
    main(json_file, pdf_file, out_file)
