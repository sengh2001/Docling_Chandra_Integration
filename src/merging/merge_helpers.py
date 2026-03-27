"""
Low-level merge operations for dual-scan OCR documents.

Merges text items, tables, and captions from a text-only OCR scan into a
baseline (full) OCR scan using bounding box overlap (IoU) and spatial proximity.
Used by merge_document_scans.merge_documents().
"""

from docling_core.types.doc import (
    DoclingDocument,
    DocItemLabel,
    TextItem,
    TableItem,
    NodeItem,
)


def _get_bbox_for_item(item: TextItem):
    """Helper to get the first bounding box from provenance."""
    if hasattr(item, "prov") and item.prov and len(item.prov) > 0:
        return item.prov[0].bbox
    return None


def _group_by_page(
    doc: DoclingDocument, target_labels: set, transverse_pictures: bool = False
) -> dict[int, NodeItem]:
    """Group document items by page number, filtering by label type."""
    doc_by_page = {}
    for item, _ in doc.iterate_items(traverse_pictures=transverse_pictures):
        if item.label in target_labels and item.prov:
            page_no = item.prov[0].page_no
            if page_no not in doc_by_page:
                doc_by_page[page_no] = []
            doc_by_page[page_no].append(item)
    return doc_by_page


def _get_siblings(doc: DoclingDocument, item: TextItem):
    """
    Get (prev_sibling, next_sibling) items for a given item in the doc.
    Returns (None, None) if not found or no siblings.
    """
    if not item.parent:
        return None, None

    # Parent ref
    parent_ref = item.parent
    try:
        parent = parent_ref.resolve(doc)
    except Exception:
        return None, None

    # Find item index in parent.children using cref for equality
    # children are RefItems
    my_cref = item.self_ref
    children = parent.children

    idx = -1
    for i, child_ref in enumerate(children):
        if child_ref.cref == my_cref:
            idx = i
            break

    if idx == -1:
        return None, None

    prev_sib = None
    next_sib = None

    if idx > 0:
        prev_sib = children[idx - 1].resolve(doc)

    if idx < len(children) - 1:
        next_sib = children[idx + 1].resolve(doc)

    return prev_sib, next_sib


def _find_best_match(t_item, candidate_items):
    """
    Find best matching item from candidates based on IoU (priority) then Distance.
    Returns dict with match details or None.
    """
    if not t_item or not candidate_items:
        return None

    t_bbox = _get_bbox_for_item(t_item)
    if not t_bbox:
        return None

    best_match = None
    best_iou = -1.0

    closest_match = None
    min_dist_sq = float("inf")

    # Center of t_item
    # BoundingBox has l, t, r, b. Origin implies coordinates.
    # Assuming origin is consistent (TOPLEFT or BOTTOMLEFT).
    tcx = (t_bbox.l + t_bbox.r) / 2.0
    tcy = (t_bbox.t + t_bbox.b) / 2.0

    for b_item in candidate_items:
        b_bbox = _get_bbox_for_item(b_item)
        if not b_bbox:
            continue

        # Skip items with different coordinate origins

        if t_bbox.coord_origin != b_bbox.coord_origin:
            continue  # Can't compare different origins easily without conversion

        iou = t_bbox.intersection_over_union(b_bbox)
        if iou > 0:
            if iou > best_iou:
                best_iou = iou
                best_match = b_item

        # Only track distance as fallback when no IoU overlap found
        if best_iou <= 0:
            bcx = (b_bbox.l + b_bbox.r) / 2.0
            bcy = (b_bbox.t + b_bbox.b) / 2.0
            dist_sq = (tcx - bcx) ** 2 + (tcy - bcy) ** 2

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_match = b_item

    if best_match is not None:
        return {"item": best_match, "score": best_iou, "type": "iou"}
    elif closest_match:
        # Score is negative distance so higher is better
        return {"item": closest_match, "score": -min_dist_sq, "type": "dist"}

    return None


def _has_valid_charspan(item, min_charspan):
    """Check if item meets minimum charspan requirement."""
    if not hasattr(item, "prov") or not item.prov:
        return True  # No provenance — assume valid

    for prov in item.prov:
        charspan = prov.charspan
        span_length = charspan[1] - charspan[0]
        if span_length < min_charspan:
            # Matches remove_short_text_items: reject if ANY prov is short
            return False
    return True


# Doesn't handle complex N:M overlaps
def merge_text_items(
    baseline_doc: DoclingDocument, text_only_doc: DoclingDocument, min_charspan: int = 5
):
    """
    Merge text items from text_only_doc into baseline_doc.

    Logic:
    1. Iterate through pages.
    2. Overlap detection (IoU) between baseline items and text-only items.
    3. Case 1 (1:1): High overlap -> Replace text. Imperfect overlap -> Keep larger box & text.
    4. Case 2 (N:1): Text-only item covers multiple baseline items -> Replace N baseline with 1 text-only.
    5. Case 3 (1:N): Baseline covers multiple text-only -> Keep baseline (do nothing).
    """

    # Collect all operations first, then apply (avoids mutation during iteration)

    # Identify items to modify
    items_to_replace_text: list[tuple[TextItem, str]] = []  # (baseline_item, new_text)
    items_to_replace_full: list[
        tuple[TextItem, TextItem]
    ] = []  # (baseline_item, new_item_replacement)
    items_to_merge_n_to_1: list[
        tuple[list[TextItem], TextItem]
    ] = []  # ([baseline_items], new_item)

    # We only care about TEXT type items as per request
    target_labels = {DocItemLabel.TEXT}  # Can add others if needed

    # Group items by page for efficiency
    # Assume provenance[0].page_no is valid

    # We collect Text Items for replacement logic

    baseline_items_by_page: dict[int, list[TextItem]] = _group_by_page(
        baseline_doc, target_labels
    )
    text_items_by_page: dict[int, list[TextItem]] = _group_by_page(
        text_only_doc, target_labels
    )
    # Iterate pages present in both
    common_pages = set(baseline_items_by_page.keys()) & set(text_items_by_page.keys())

    for page in common_pages:
        b_items = baseline_items_by_page[page]
        t_items = text_items_by_page[page]

        # Build mapping of T-item -> overlapping B-items
        # and B-item -> overlapping T-items

        # This is O(N*M), simple enough for typical page item counts
        t_to_b_overlaps: dict[id, list[TextItem]] = {id(t): [] for t in t_items}
        b_to_t_overlaps: dict[id, list[TextItem]] = {id(b): [] for b in b_items}

        # Mapping back from id to object for convenience
        t_map = {id(t): t for t in t_items}
        # b_map = {id(b): b for b in b_items}

        for t in t_items:
            t_bbox = _get_bbox_for_item(t)
            if not t_bbox:
                continue

            for b in b_items:
                b_bbox = _get_bbox_for_item(b)
                if not b_bbox:
                    continue

                # Check weak overlap first (quick filter)
                if t_bbox.overlaps(b_bbox):

                    inter_area = t_bbox.intersection_area_with(b_bbox)
                    if inter_area > 0:
                        t_to_b_overlaps[id(t)].append(b)
                        b_to_t_overlaps[id(b)].append(t)

        # Process overlaps
        processed_b_ids = set()

        for t_id, overlapping_b_items in t_to_b_overlaps.items():
            t_item = t_map[t_id]

            if not overlapping_b_items:
                continue  # No matching baseline item found

            # Case: 1 Text Item overlaps N Baseline Items
            if len(overlapping_b_items) > 1:
                # N:1 case — text-only item covers multiple baseline items
                # Filter out B items that also overlap other T items (ambiguous)

                valid_b_items = []
                for b in overlapping_b_items:
                    # Ensure this B item primarily belongs to this T item
                    # i.e., it doesn't overlap significantly with other T items?
                    if len(b_to_t_overlaps[id(b)]) == 1:
                        valid_b_items.append(b)
                    else:
                        # Ambiguous / Complex overlap
                        pass

                if len(valid_b_items) > 1:
                    # Valid N:1 candidates
                    items_to_merge_n_to_1.append((valid_b_items, t_item))
                    for b in valid_b_items:
                        processed_b_ids.add(id(b))

            elif len(overlapping_b_items) == 1:
                # 1:1 Match (or 1:N from B perspective)
                b_item = overlapping_b_items[0]

                if id(b_item) in processed_b_ids:
                    continue

                # Check if B overlaps other T items
                overlapping_t_for_b = b_to_t_overlaps[id(b_item)]

                if len(overlapping_t_for_b) == 1:
                    # Truly 1:1

                    t_bbox = _get_bbox_for_item(t_item)
                    b_bbox = _get_bbox_for_item(b_item)

                    iou = t_bbox.intersection_over_union(b_bbox)

                    if iou > 0.9:
                        # "Almost complete overlap... replace baseline text with text from text only scan"
                        items_to_replace_text.append((b_item, t_item.text))
                    else:
                        # "Not a perfect match"
                        # Case 1: "One of the scans found a smaller box" -> "use... results of the larger box"

                        t_area = t_bbox.area()
                        b_area = b_bbox.area()

                        if t_area > b_area:
                            # Text Scan is larger -> Replace Baseline item with Text Scan item
                            items_to_replace_full.append((b_item, t_item))
                        else:
                            # Baseline is larger -> Keep Baseline (do nothing)
                            pass

                    processed_b_ids.add(id(b_item))

                else:
                    # B overlaps multiple T items (1 Baseline : N Text)
                    # Case 2 part 2: "If the larger box is from the full scan (baseline), no changes need to be made."
                    # Since B overlaps multiple Ts, B is likely larger (or they are disjoint Ts inside B).
                    # We do nothing.
                    pass

    # Apply changes


    # 1. Text Replacements (Safest)
    for b_item, new_text in items_to_replace_text:
        b_item.text = new_text

    # 2. Full Replacements (1:1)
    for b_item, new_item in items_to_replace_full:
        # Use b_item as base to preserve metadata, but update text and prov from new_item
        replacement = b_item.model_copy()
        replacement.text = new_item.text
        replacement.prov = new_item.prov

        baseline_doc.replace_item(new_item=replacement, old_item=b_item)

    # 3. N:1 Merges
    # "replace the multiple items in the full scan with the larger item."
    # "the ref needs to be add the children of the parent in the appropriate place"

    for b_items_group, new_item in items_to_merge_n_to_1:
        # Insert before the first item in the group, then delete all old items

        if not b_items_group:
            continue

        # Check if they have same parent
        parents = {b.parent.cref for b in b_items_group}
        if len(parents) > 1:
            # Items span multiple parents — too complex, skip
            print(f"Skipping N:1 merge for items spanning multiple parents: {parents}")
            continue

        # Use first item as anchor for insertion
        target_sibling = b_items_group[0]

        replacement = new_item.model_copy()

        # Insert new item
        baseline_doc.insert_item_before_sibling(
            new_item=replacement, sibling=target_sibling
        )

        # Delete old items
        baseline_doc.delete_items(node_items=b_items_group)


def insert_text_items(
    baseline_doc: DoclingDocument, text_only_doc: DoclingDocument, min_charspan: int = 5
):
    """
    Insert text items and section headers from text_only_doc that do not overlap with any text items
    or section headers in baseline_doc.
    Uses sibling matching to find insertion point.
    """
    # We care about TEXT and SECTION_HEADER items for insertion
    target_labels = {DocItemLabel.TEXT, DocItemLabel.SECTION_HEADER}

    # 1. Collect Baseline items in target_labels for overlap check
    # Note: SECTION_HEADER is also in 'special_labels' below to ensure we don't overwrite them
    # even if they are in target_labels.
    baseline_items_by_page: dict[int, list[DoclingDocument.NodeItem]] = _group_by_page(
        baseline_doc, target_labels
    )

    # 2. Collect All Baseline items (anchors) for sibling matching
    baseline_anchors_by_page: dict[int, list[DoclingDocument.NodeItem]] = {}
    for item, _ in baseline_doc.iterate_items():
        if item.prov and _get_bbox_for_item(item):
            page_no = item.prov[0].page_no
            if page_no not in baseline_anchors_by_page:
                baseline_anchors_by_page[page_no] = []
            baseline_anchors_by_page[page_no].append(item)

    # 3. Collect Text-only items
    text_items_by_page: dict[int, list[TextItem]] = _group_by_page(
        text_only_doc, target_labels
    )
    # 4. Collect Baseline Special Items (Headers, Captions) to avoid overlap
    # N.B. SECTION_HEADER is included here so that we treat them as "blockers"
    # even though we also insert them. This prevents replacing a baseline header with a new text item.
    special_labels = {
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.CAPTION,
        DocItemLabel.LIST_ITEM,
    }
    baseline_special_items_by_page: dict[int, list[DoclingDocument.NodeItem]] = (
        _group_by_page(baseline_doc, special_labels, True)
    )

    # list of insertions to perform: (mode, anchor_item, new_item)
    items_to_insert: list[
        tuple[str, DoclingDocument.NodeItem, DoclingDocument.NodeItem]
    ] = []

    # Iterate pages present in text_only_doc
    for page in text_items_by_page:
        t_items = text_items_by_page[page]
        b_items = baseline_items_by_page.get(page, [])
        anchors = baseline_anchors_by_page.get(page, [])
        special_items = baseline_special_items_by_page.get(page, [])

        if not anchors:
            continue  # Can't insert if no anchors on page?

        for t_item in t_items:
            # Check charspan if strictly TEXT item
            if t_item.label == DocItemLabel.TEXT and min_charspan > 0:
                if not _has_valid_charspan(t_item, min_charspan):
                    continue

            t_bbox = _get_bbox_for_item(t_item)
            if not t_bbox:
                continue

            # Check overlap with ANY baseline SPECIAL item (skip if overlap)
            has_special_overlap = False
            for s_item in special_items:
                s_bbox = _get_bbox_for_item(s_item)
                if not s_bbox:
                    continue
                if t_bbox.overlaps(s_bbox):
                    if t_bbox.coord_origin == s_bbox.coord_origin:
                        if t_bbox.intersection_area_with(s_bbox) > 0:
                            has_special_overlap = True
                            break

            if has_special_overlap:
                continue

            # Check overlap with ANY baseline TEXT item
            has_overlap = False
            for b in b_items:
                b_bbox = _get_bbox_for_item(b)
                if not b_bbox:
                    continue
                # Basic overlap check
                if t_bbox.overlaps(b_bbox):

                    if t_bbox.coord_origin == b_bbox.coord_origin:
                        if t_bbox.intersection_area_with(b_bbox) > 0:
                            has_overlap = True
                            break

            if has_overlap:
                continue

            # No overlap found -> Insert

            # 1. Get Siblings in text_only_doc
            prev_sib, next_sib = _get_siblings(text_only_doc, t_item)

            # 2. Find matches for siblings in baseline anchors
            match_prev = _find_best_match(prev_sib, anchors)
            match_next = _find_best_match(next_sib, anchors)

            # 3. Decide insertion
            winner_ref = None
            mode = None

            def compare_matches(m1, m2):
                if not m1 and not m2:
                    return 0
                if m1 and not m2:
                    return 1
                if not m1 and m2:
                    return -1

                if m1["type"] == "iou" and m2["type"] == "dist":
                    return 1
                if m1["type"] == "dist" and m2["type"] == "iou":
                    return -1

                if m1["score"] > m2["score"]:
                    return 1
                if m1["score"] < m2["score"]:
                    return -1
                return 0

            res = compare_matches(match_prev, match_next)

            if res >= 0 and match_prev:
                winner_ref = match_prev["item"]
                mode = "after"
            elif res < 0 and match_next:
                winner_ref = match_next["item"]
                mode = "before"

            if winner_ref:
                items_to_insert.append((mode, winner_ref, t_item))

    # Perform insertions
    for mode, anchor_item, new_item in items_to_insert:
        replacement = new_item.model_copy()
        try:
            if mode == "after":
                baseline_doc.insert_item_after_sibling(
                    new_item=replacement, sibling=anchor_item
                )
            elif mode == "before":
                baseline_doc.insert_item_before_sibling(
                    new_item=replacement, sibling=anchor_item
                )
        except Exception:
            pass


def merge_tables(baseline_doc: DoclingDocument, text_only_doc: DoclingDocument):
    """
    Merge tables from text_only_doc into baseline_doc.

    Cases:
    1. Overlap: Replace baseline table with text-only table (keep baseline metadata).
    2. No Overlap: Insert text-only table using sibling matchmaking.
    """
    target_labels = {DocItemLabel.TABLE}

    # 1. Collect Baseline Tables
    baseline_tables_by_page: dict[int, list[TableItem]] = _group_by_page(
        baseline_doc, target_labels
    )
    # 2. Collect Baseline Anchors (for insertion)
    baseline_anchors_by_page: dict[int, list[DoclingDocument.NodeItem]] = {}
    for item, _ in baseline_doc.iterate_items():
        if item.prov and _get_bbox_for_item(item):
            page_no = item.prov[0].page_no
            if page_no not in baseline_anchors_by_page:
                baseline_anchors_by_page[page_no] = []
            baseline_anchors_by_page[page_no].append(item)

    # 3. Collect Text-only Tables
    text_tables_by_page: dict[int, list[TableItem]] = _group_by_page(
        text_only_doc, target_labels
    )

    items_to_replace: list[tuple[TableItem, TableItem]] = []  # (baseline, text_only)
    items_to_insert: list[
        tuple[str, DoclingDocument.NodeItem, TableItem]
    ] = []  # (mode, anchor, new_item)

    # Process pages present in text_only_doc
    for page in text_tables_by_page:
        t_tables = text_tables_by_page.get(page, [])
        b_tables = baseline_tables_by_page.get(page, [])
        anchors = baseline_anchors_by_page.get(page, [])

        for t_table in t_tables:
            t_bbox = _get_bbox_for_item(t_table)
            if not t_bbox:
                continue

            # Check for overlap
            best_overlap_b_table = None
            max_iou = 0.0

            for b_table in b_tables:
                b_bbox = _get_bbox_for_item(b_table)
                if not b_bbox:
                    continue

                if t_bbox.coord_origin == b_bbox.coord_origin:
                    iou = t_bbox.intersection_over_union(b_bbox)
                    if iou > 0:
                        if iou > max_iou:
                            max_iou = iou
                            best_overlap_b_table = b_table

            if best_overlap_b_table:
                # Case 1: Overlap found -> Replace
                items_to_replace.append((best_overlap_b_table, t_table))
            else:
                # Case 2: No Overlap -> Insert
                if not anchors:
                    continue

                prev_sib, next_sib = _get_siblings(text_only_doc, t_table)
                match_prev = _find_best_match(prev_sib, anchors)
                match_next = _find_best_match(next_sib, anchors)

                winner_ref = None
                mode = None

                # Compare matches logic (same as insert_text_items)
                def compare_matches(m1, m2):
                    if not m1 and not m2:
                        return 0
                    if m1 and not m2:
                        return 1
                    if not m1 and m2:
                        return -1
                    if m1["type"] == "iou" and m2["type"] == "dist":
                        return 1
                    if m1["type"] == "dist" and m2["type"] == "iou":
                        return -1
                    if m1["score"] > m2["score"]:
                        return 1
                    if m1["score"] < m2["score"]:
                        return -1
                    return 0

                res = compare_matches(match_prev, match_next)

                if res >= 0 and match_prev:
                    winner_ref = match_prev["item"]
                    mode = "after"
                elif res < 0 and match_next:
                    winner_ref = match_next["item"]
                    mode = "before"

                if winner_ref:
                    items_to_insert.append((mode, winner_ref, t_table))

    # Apply Replacements
    for b_table, t_table in items_to_replace:
        # Update baseline table with text-only data and prov
        replacement = b_table.model_copy()

        # TableItem usually has 'data' field. Use getattr/setattr or direct access if typed.
        if hasattr(t_table, "data"):
            replacement.data = t_table.data
        if hasattr(t_table, "prov"):
            replacement.prov = t_table.prov

        # We replace the item in the doc
        baseline_doc.replace_item(new_item=replacement, old_item=b_table)

    # Apply Insertions
    for mode, anchor_item, new_item in items_to_insert:
        replacement = new_item.model_copy()
        try:
            replacement.children = []
            if mode == "after":
                baseline_doc.insert_item_after_sibling(
                    new_item=replacement, sibling=anchor_item
                )
            elif mode == "before":
                baseline_doc.insert_item_before_sibling(
                    new_item=replacement, sibling=anchor_item
                )
        except Exception:
            pass


def merge_captions(
    baseline_doc: DoclingDocument, text_only_doc: DoclingDocument, verbose: bool = False
):
    """
    Merge captions from text_only_doc into baseline_doc.
    Moves text-only captions to their corresponding parents in baseline_doc.
    Handles overlap with existing text (mislabeled) or captions (skip).
    """
    target_labels = {DocItemLabel.CAPTION}

    # 1. Collect Baseline Parents (Pictures and Tables) for matching
    # Map page -> list of (item, bbox)
    parent_labels = {DocItemLabel.PICTURE, DocItemLabel.TABLE}

    baseline_parents_by_page: dict[int, list[DoclingDocument.NodeItem]] = (
        _group_by_page(baseline_doc, parent_labels)
    )

    # 2. Collect Baseline Captions (to check for existence)
    baseline_captions_by_page: dict[int, list[DoclingDocument.NodeItem]] = (
        _group_by_page(baseline_doc, {DocItemLabel.CAPTION})
    )

    # 3. Collect Baseline Text (to check for mislabeled text)
    baseline_text_by_page: dict[int, list[DoclingDocument.NodeItem]] = _group_by_page(
        text_only_doc, {DocItemLabel.TEXT}
    )

    # iterate text-only captions
    for t_caption, _ in text_only_doc.iterate_items():
        if t_caption.label not in target_labels:
            continue

        t_bbox = _get_bbox_for_item(t_caption)
        if not t_bbox:
            continue
        if not t_caption.prov:
            continue
        page_no = t_caption.prov[0].page_no

        # Check if caption already exists in baseline at this location
        existing_captions = baseline_captions_by_page.get(page_no, [])
        already_exists = False
        for b_cap in existing_captions:
            b_cap_bbox = _get_bbox_for_item(b_cap)
            if b_cap_bbox and t_bbox.coord_origin == b_cap_bbox.coord_origin:
                if (
                    t_bbox.intersection_over_union(b_cap_bbox) > 0.5
                ):  # Sufficient overlap
                    already_exists = True
                    break
        if already_exists:
            continue

        # Find matching baseline parent
        if not t_caption.parent:
            continue
        try:
            t_parent = t_caption.parent.resolve(text_only_doc)
        except Exception:
            continue

        t_parent_bbox = _get_bbox_for_item(t_parent)
        if not t_parent_bbox:
            continue

        # Search for this parent in baseline
        potential_parents = baseline_parents_by_page.get(page_no, [])
        matching_b_parent = None
        max_iou = 0.0

        for b_parent in potential_parents:
            # Type must match (Picture <-> Picture, Table <-> Table)
            if b_parent.label != t_parent.label:
                continue

            b_parent_bbox = _get_bbox_for_item(b_parent)
            if not b_parent_bbox:
                continue

            if t_parent_bbox.coord_origin == b_parent_bbox.coord_origin:
                iou = t_parent_bbox.intersection_over_union(b_parent_bbox)
                if iou > 0.5 and iou > max_iou:
                    max_iou = iou
                    matching_b_parent = b_parent

        if matching_b_parent:
            # We found the parent in baseline.
            # 1. Check for overlapping baseline text (Mislabeled) -> Delete it
            existing_text = baseline_text_by_page.get(page_no, [])
            items_to_delete = []
            for b_text in existing_text:
                b_text_bbox = _get_bbox_for_item(b_text)
                if b_text_bbox and t_bbox.coord_origin == b_text_bbox.coord_origin:
                    # Overlap?
                    if (
                        t_bbox.intersection_over_union(b_text_bbox) > 0.1
                    ):  # Some significant overlap
                        items_to_delete.append(b_text)

            if items_to_delete:
                baseline_doc.delete_items(node_items=items_to_delete)

            # 2. Add caption to baseline parent
            new_caption = t_caption.model_copy()
            # Append child item handles adding to children list and setting parent ref
            try:

                baseline_doc.append_child_item(
                    child=new_caption, parent=matching_b_parent
                )
            except Exception:
                pass
