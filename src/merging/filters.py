"""
Pre-merge cleanup filters for DoclingDocuments.

Removes unwanted elements (headers, footers, short text, text inside pictures)
before merging. Used by merge_document_scans.merge_documents() as Step 1.
"""

from docling_core.types.doc import (
    DocItemLabel,
    ContentLayer,
)


def remove_text_from_pictures(doc):
    """Remove all non-caption text items that are children of picture elements."""
    items_to_remove = []
    for item, level in doc.iterate_items(traverse_pictures=True):
        if (item.label != DocItemLabel.CAPTION) and ("picture" in item.parent.cref):
            items_to_remove.append(item)

    doc.delete_items(node_items=items_to_remove)


def remove_headers_and_footers(doc):
    """Remove all page headers and footers from a DoclingDocument."""
    items_to_remove = []
    for item, level in doc.iterate_items(
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
    ):
        if item.label in [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]:
            items_to_remove.append(item)

    if items_to_remove:
        doc.delete_items(node_items=items_to_remove)


def remove_short_text_items(doc, min_charspan):
    """Remove text items with charspan less than min_charspan."""
    items_to_remove = []

    for item, level in doc.iterate_items(
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
    ):
        # Ensure we only touch TEXT items
        if item.label != DocItemLabel.TEXT:
            continue

        # Check if item has prov with charspan
        if hasattr(item, "prov") and item.prov:
            for prov in item.prov:
                charspan = prov.charspan
                # charspan is a tuple like (start, end)
                span_length = charspan[1] - charspan[0]
                if span_length < min_charspan:
                    items_to_remove.append(item)
                    break  # Only need to add once

    if items_to_remove:
        doc.delete_items(node_items=items_to_remove)
