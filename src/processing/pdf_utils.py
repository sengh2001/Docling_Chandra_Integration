"""
PDF Utilities: Text-layer extraction and watermark removal
==========================================================

Provides functions to manipulate PDF files using PyMuPDF (fitz):
- Extract the text layer while removing images and vector graphics.
- Remove grayscale watermark images from PDFs.

These are used to create the "text-only" scan for dual-scan OCR processing.
"""

import fitz


def extract_save_text_layer(
    input_pdf: str, output_pdf: str | None = None, verbose: bool = False
):
    """
    Strip all non-text content from a PDF, preserving text with original positioning.

    Removes images and vector graphics/drawings, keeping only the text layer.
    This produces a cleaner input for OCR, avoiding interference from watermarks,
    figures, and decorative elements.

    Args:
        input_pdf: Path to the source PDF.
        output_pdf: If provided, save the text-only PDF to this path.
        verbose: If True, print the output path after saving.
    """
    doc = fitz.open(input_pdf)

    for page in doc:
        # Remove images
        for img in page.get_images():
            page.delete_image(img[0])

        # Remove drawings/vector graphics
        page.clean_contents()

        # This removes all non-text content while preserving text positioning
        page.wrap_contents()

    if output_pdf is not None:
        doc.save(output_pdf, garbage=4, deflate=True)
        if verbose:
            print(f"Created: {output_pdf}")


def remove_watermark(
    input_pdf: str, output_pdf: str | None = None, verbose: bool = False
):
    """
    Remove grayscale watermark images from a PDF.

    Targets images using the DeviceGray colour space, which typically
    corresponds to watermarks in textbook PDFs. Non-grayscale images
    (colour photos, diagrams) are preserved.

    Args:
        input_pdf: Path to the source PDF.
        output_pdf: If provided, save the cleaned PDF to this path.
        verbose: If True, print the output path after saving.
    """
    doc = fitz.open(input_pdf)

    for page in doc:
        # Remove images
        for img in page.get_images():
            if img[5] == "DeviceGray":
                page.delete_image(img[0])

        # Remove drawings/vector graphics
        page.clean_contents()

        # This removes all non-text content while preserving text positioning
        page.wrap_contents()

    doc.save(output_pdf, garbage=4, deflate=True)
    if verbose:
        print(f"Created: {output_pdf}")
