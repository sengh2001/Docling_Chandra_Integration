"""
Docling OCR Pipeline: PDF-to-structured-document conversion
============================================================

Wraps the Docling library to run layout detection + Tesseract CLI OCR
on PDF files, producing structured DoclingDocument output in HTML and/or
JSON formats.

Used by src/create_parallel_corpora.py and dev_scripts/extract_structures_split.py.
"""

from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


def docling_ocr_pipeline(
    input_pdf: str,
    converter: DocumentConverter | None = None,
    output_html: str | None = None,
    output_json: str | None = None,
    lang: str | None = None,
):
    """
    Run Docling OCR on a PDF and optionally save as HTML and/or JSON.

    Args:
        input_pdf: Path to the input PDF file.
        converter: Pre-configured DocumentConverter. If None, one is created
            using the lang parameter.
        output_html: If provided, save HTML export to this path.
        output_json: If provided, save JSON export to this path.
        lang: Tesseract language code (e.g. "eng", "hin"). Required if
            converter is not provided.

    Returns:
        DoclingDocument with the parsed document structure.

    Raises:
        ValueError: If neither converter nor lang is provided.
    """
    if converter is None:
        if lang is None:
            raise ValueError("Either one of converter or lang is needed")
        converter = create_converter(lang)

    # Run conversion
    doc = converter.convert(Path(input_pdf)).document

    # Write html_doc to file
    if output_html is not None:
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(doc.export_to_html())

    if output_json is not None:
        doc.save_as_json(output_json)

    print(f"Done: output saved to {output_html}, json to {output_json}")
    return doc


def create_converter(lang: str, psm: int | None = None) -> DocumentConverter:
    """
    Create a Docling DocumentConverter configured for Tesseract OCR.

    Args:
        lang: Tesseract language code (e.g. "eng", "hin", "tam").
        psm: Tesseract page segmentation mode. None uses the default.

    Returns:
        Configured DocumentConverter instance.
    """
    tesseract_options = TesseractCliOcrOptions(
        lang=[lang],
        force_full_page_ocr=True,
        psm=psm,
    )

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        ocr_options=tesseract_options,
    )

    # Create Docling converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    return converter
