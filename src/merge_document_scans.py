"""
Merge Document Scans: Coordinator for dual-scan OCR merging
============================================================

Orchestrates the merging of two OCR scans (full original + text-only)
into a single improved DoclingDocument. Applies cleanup filters first,
then runs text, table, and caption merge operations.

This is the main entry point called by src/create_parallel_corpora.py Step 2.
"""

from docling_core.types.doc import DoclingDocument
from src.merging.filters import (
    remove_headers_and_footers,
    remove_short_text_items,
    remove_text_from_pictures,
)
from src.merging.merge_helpers import (
    merge_captions,
    merge_tables,
    merge_text_items,
    insert_text_items,
)
import copy


def merge_documents(
    baseline_doc: DoclingDocument,
    text_only_doc: DoclingDocument,
    min_charspan: int = 5,
    verbose: bool = False,
) -> DoclingDocument:
    """
    Coordinator function to merge data from text_only_doc into baseline_doc.
    Returns a new DoclingDocument with the changes.
    """
    # Create Deep Copy (to avoid modifying originals)
    new_doc = copy.deepcopy(baseline_doc)

    # 1. Cleanups
    remove_text_from_pictures(new_doc)
    remove_headers_and_footers(new_doc)
    remove_short_text_items(new_doc, min_charspan=min_charspan)

    # 2. Merging Logic
    merge_text_items(new_doc, text_only_doc, min_charspan=min_charspan)
    insert_text_items(new_doc, text_only_doc, min_charspan=min_charspan)
    merge_tables(new_doc, text_only_doc)
    merge_captions(new_doc, text_only_doc, verbose)

    return new_doc


if __name__ == "__main__":
    json_paths: list[dict[str, str]] = []
    # Define paths as a list of dictionaries with keys "baseline", "text_only" and "output"

    for document_paths in json_paths:
        baseline_doc = DoclingDocument.load(document_paths["baseline"])
        text_only_doc = DoclingDocument.load(document_paths["text_only"])
        new_doc = merge_documents(baseline_doc, text_only_doc)
        new_doc.save(document_paths["output"])
        print("Merge completed and saved to ", document_paths["output"])
