"""
ParallelAlignedDocument: Multi-language document alignment
===========================================================

This module manages aligned versions of a document in three languages:
English, Hindi, and Tamil.

Alignment Strategy:
- English ↔ Hindi: Uses precomputed match dictionaries (semantic matching)
- English ↔ Tamil: Uses bounding box matching (spatial alignment)
- Hindi ↔ Tamil: Never matched directly; always routes through English

Context Collection:
- Once a main match is found (via dict or bbox), context (before/after) is
  computed the same way for all languages using sibling traversal.
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from docling_core.types.doc import (
    DoclingDocument,
    DocItemLabel,
    NodeItem,
)

from src.retreival.preprocess import embed_glob_texts
from src.retreival.rets import top_k_similar_refs


class ParallelAlignedDocument:
    """
    Manages aligned versions of a document in multiple languages (English, Hindi, Tamil).
    Provides functionality to retrieve parallel content and context based on bounding box alignment.
    """

    def __init__(
        self,
        data_en: Dict[str, Any],
        data_hi: Dict[str, Any],
        data_ta: Dict[str, Any],
        eng_to_hi_matches: Optional[Dict[str, str]] = None,
        hi_to_eng_matches: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the ParallelAlignedDocument with JSON data for each language.
        
        Args:
            data_en: JSON data for English document.
            data_hi: JSON data for Hindi document.
            data_ta: JSON data for Tamil document.
            eng_to_hi_matches: Precomputed English cref → Hindi cref mapping.
            hi_to_eng_matches: Precomputed Hindi cref → English cref mapping.
        """
        # Load DoclingDocument objects from the provided dictionaries
        self.docs = {
            "eng": DoclingDocument.model_validate(data_en),
            "hi": DoclingDocument.model_validate(data_hi),
            "ta": DoclingDocument.model_validate(data_ta),
        }
        
        # Store precomputed match dictionaries for English-Hindi alignment
        self._eng_to_hi = eng_to_hi_matches or {}
        self._hi_to_eng = hi_to_eng_matches or {}
        
        # Pre-compute page groupings to speed up bbox-based retrieval
        # map: lang -> page_no -> list[item]
        self._page_cache: Dict[str, Dict[int, List[NodeItem]]] = {}
        for lang, doc in self.docs.items():
            self._page_cache[lang] = self._group_items_by_page(doc)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def retrieve_relevant_docitem(
        self,
        query: str,
        language: str = "eng",
        embeddings_dir: str = "outputs_embeddings",
        docs_glob_template: str = "./outputs_split/Class_6-Science-{lang}/*/merged.json",
    ) -> Tuple[str, str]:
        """
        Retrieve the most relevant document item for a given query using RAG.
        
        This function uses precomputed embeddings if available, otherwise computes
        and saves them for future use.
        
        Args:
            query: The search query text.
            language: The target language to search in ("eng", "hi", or "ta").
            embeddings_dir: Directory where embeddings are stored/cached.
            docs_glob_template: Glob pattern template for document files.
                                Use {lang} as placeholder for language name.
            
        Returns:
            Tuple[str, str]: (language, doc_item_ref) - The language code and 
                             the self_ref of the most relevant document item.
        """
        # Map internal language codes to directory names
        lang_dir_map = {"eng": "English", "hi": "Hindi", "ta": "Tamil"}
        
        if language not in lang_dir_map:
            raise ValueError(f"Unknown language: {language}. Must be one of {list(lang_dir_map.keys())}")
        
        lang_dir_name = lang_dir_map[language]
        
        # Load embeddings and refs (compute if not available)
        embeddings, refs_dict = self._load_or_compute_embeddings(
            language=language,
            lang_dir_name=lang_dir_name,
            embeddings_dir=embeddings_dir,
            docs_glob_template=docs_glob_template,
        )
        
        # Load the sentence transformer model (lazily cached)
        model = self._get_embedding_model()
        
        # Find top-1 most similar document item
        top_results = top_k_similar_refs(embeddings, query, model, refs_dict, k=5)
        
        if not top_results:
            raise ValueError(f"No matching document items found for query: {query}")
        
        # Return the language and self_ref of the best match
        best_match = top_results[0]
        return language, best_match["self_ref"]

    def _get_embedding_model(self) -> SentenceTransformer:
        """
        Get the sentence transformer model, loading it lazily on first access.
        
        The model is cached as a class attribute to avoid reloading on each call.
        
        Returns:
            SentenceTransformer: The loaded embedding model.
        """
        # Use class-level caching to avoid reloading the model
        if not hasattr(ParallelAlignedDocument, "_embedding_model"):
            print("Loading sentence transformer model: krutrim-ai-labs/vyakyarth")
            ParallelAlignedDocument._embedding_model = SentenceTransformer(
                "krutrim-ai-labs/vyakyarth"
            )
        return ParallelAlignedDocument._embedding_model

    def _load_or_compute_embeddings(
        self,
        language: str,
        lang_dir_name: str,
        embeddings_dir: str,
        docs_glob_template: str,
    ) -> Tuple[np.ndarray, dict]:
        """
        Load precomputed embeddings from disk, or compute and save them if not available.
        
        Args:
            language: Internal language code ("eng", "hi", "ta").
            lang_dir_name: Human-readable language name for directory ("English", "Hindi", "Tamil").
            embeddings_dir: Base directory for embedding storage.
            docs_glob_template: Glob template for finding document JSON files.
            
        Returns:
            Tuple[np.ndarray, dict]: (embeddings array, refs dictionary)
        """
        embeddings_path = os.path.join(embeddings_dir, lang_dir_name, "embeddings.npz")
        refs_path = os.path.join(embeddings_dir, lang_dir_name, "refs.json")
        
        # Check if precomputed embeddings exist
        if os.path.exists(embeddings_path) and os.path.exists(refs_path):
            print(f"Loading precomputed embeddings from {embeddings_dir}/{lang_dir_name}/")
            
            # Load embeddings from NPZ file
            embeddings = np.load(embeddings_path)["arr_0"]
            
            # Load refs dictionary from JSON
            with open(refs_path, "r") as f:
                refs_dict = json.load(f)
            
            return embeddings, refs_dict
        
        # Embeddings not found - compute and save them
        print(f"Precomputed embeddings not found for {lang_dir_name}. Computing...")
        
        # Build the glob pattern for this language
        docs_glob = docs_glob_template.format(lang=lang_dir_name)
        save_dir = os.path.join(embeddings_dir, lang_dir_name)
        
        # Get the model and compute embeddings
        model = self._get_embedding_model()
        embeddings, refs_list, file_ranges = embed_glob_texts(
            docs_glob, model, save_to_dir=save_dir
        )
        
        # Build refs_dict in the expected format
        refs_dict = {"refs": refs_list, "file_ranges": file_ranges}
        
        print(f"Embeddings computed and saved to {save_dir}/")
        
        return embeddings, refs_dict

    def retrieve_parallel_data(self, cref: str, language: str) -> Dict[str, Any]:
        """
        Retrieve the item corresponding to 'cref' in the specified 'language', 
        along with its parallel versions in other languages and surrounding context.

        Matching rules:
        - English query: Hindi via dict, Tamil via bbox
        - Hindi query: English via dict, Tamil via bbox on English match
        - Tamil query: English via bbox, Hindi via dict on English match

        Args:
            cref: The coordinate reference (cref) of the requested item.
            language: The language of the document where the cref exists ("eng", "hi", "ta").

        Returns:
            Dict[str, Any]: A dictionary containing aligned data for all languages.
            Format:
            {
                "eng": { "main": ..., "before": ..., "after": ... },
                "hi": { ... },
                "ta": { ... }
            }
        """
        if language not in self.docs:
            raise ValueError(f"Unknown language: {language}")

        source_doc = self.docs[language]
        
        # 1. Resolve the requested item in the source language
        source_item = self._find_item_by_cref(source_doc, cref)
        if not source_item:
            raise ValueError(f"Item with cref '{cref}' not found in {language} document.")

        # 2. Expand to principal item (e.g., ListItem → parent List)
        principal_source_item = self._expand_to_principal(source_item, source_doc)

        # 3. Route to appropriate handler based on source language
        if language == "eng":
            return self._retrieve_from_english(cref, principal_source_item)
        elif language == "hi":
            return self._retrieve_from_hindi(cref, principal_source_item)
        elif language == "ta":
            return self._retrieve_from_tamil(cref, principal_source_item)
        else:
            raise ValueError(f"Unsupported language: {language}")

    # =========================================================================
    # LANGUAGE-SPECIFIC RETRIEVAL HANDLERS
    # =========================================================================

    def _retrieve_from_english(self, eng_cref: str, eng_item: NodeItem) -> Dict[str, Any]:
        """
        Handle retrieval when the source is English.
        
        - Hindi: Use precomputed match dictionary
        - Tamil: Use bounding box matching
        """
        result = {}
        
        # English: the source
        result["eng"] = self._build_context_response(eng_item, self.docs["eng"])
        
        # Hindi: use match dictionary
        result["hi"] = self._get_match_via_dict(
            eng_cref, 
            self._eng_to_hi, 
            self.docs["hi"]
        )
        
        # Tamil: use bounding box matching
        result["ta"] = self._get_match_via_bbox(
            eng_item, 
            self.docs["eng"], 
            self.docs["ta"], 
            "ta"
        )
        
        return result

    def _retrieve_from_hindi(self, hi_cref: str, hi_item: NodeItem) -> Dict[str, Any]:
        """
        Handle retrieval when the source is Hindi.
        
        - English: Use precomputed match dictionary (reverse direction)
        - Tamil: Route through English (get English first, then bbox to Tamil)
        """
        result = {}
        
        # Hindi: the source
        result["hi"] = self._build_context_response(hi_item, self.docs["hi"])
        
        # English: use match dictionary (hi -> eng)
        eng_cref = self._hi_to_eng.get(hi_cref)
        if eng_cref:
            eng_item = self._find_item_by_cref(self.docs["eng"], eng_cref)
            if eng_item:
                eng_item = self._expand_to_principal(eng_item, self.docs["eng"])
                result["eng"] = self._build_context_response(eng_item, self.docs["eng"])
                
                # Tamil: use bbox matching on the English item
                result["ta"] = self._get_match_via_bbox(
                    eng_item, 
                    self.docs["eng"], 
                    self.docs["ta"], 
                    "ta"
                )

                return result
        
        result["eng"] = self._empty_match()
        result["ta"] = self._empty_match()
        
        return result

    def _retrieve_from_tamil(self, ta_cref: str, ta_item: NodeItem) -> Dict[str, Any]:
        """
        Handle retrieval when the source is Tamil.
        
        - English: Use bounding box matching
        - Hindi: Route through English (get English first, then dict to Hindi)
        """
        result = {}
        
        # Tamil: the source
        result["ta"] = self._build_context_response(ta_item, self.docs["ta"])
        
        # English: use bounding box matching
        eng_item = self._find_aligned_item(ta_item, self.docs["eng"], "eng")
        
        if eng_item:
            eng_item = self._expand_to_principal(eng_item, self.docs["eng"])
            result["eng"] = self._build_context_response(eng_item, self.docs["eng"])
            
            # Hindi: use match dictionary on the English match
            eng_cref = eng_item.self_ref
            result["hi"] = self._get_match_via_dict(
                eng_cref, 
                self._eng_to_hi, 
                self.docs["hi"]
            )
        else:
            result["eng"] = self._empty_match()
            result["hi"] = self._empty_match()
        
        return result

    # =========================================================================
    # MATCHING HELPERS
    # =========================================================================

    def _get_match_via_dict(
        self, 
        source_cref: str, 
        match_dict: Dict[str, str], 
        target_doc: DoclingDocument
    ) -> Dict[str, Any]:
        """
        Find a match using the precomputed dictionary.
        
        Args:
            source_cref: The cref to look up in the dictionary.
            match_dict: The precomputed cref -> cref mapping.
            target_doc: The document to resolve the matched cref in.
            
        Returns:
            Context response dict or empty match if not found.
        """
        target_cref = match_dict.get(source_cref)
        if not target_cref:
            return self._empty_match()
        
        target_item = self._find_item_by_cref(target_doc, target_cref)
        if not target_item:
            return self._empty_match()
        
        target_item = self._expand_to_principal(target_item, target_doc)
        return self._build_context_response(target_item, target_doc)

    def _get_match_via_bbox(
        self, 
        source_item: NodeItem, 
        source_doc: DoclingDocument,
        target_doc: DoclingDocument, 
        target_lang: str
    ) -> Dict[str, Any]:
        """
        Find a match using bounding box alignment.
        
        Args:
            source_item: The item to match from.
            source_doc: The document containing the source item.
            target_doc: The document to find a match in.
            target_lang: The language key for the target document.
            
        Returns:
            Context response dict or empty match if not found.
        """
        matched_item = self._find_aligned_item(source_item, target_doc, target_lang)
        
        if matched_item:
            matched_item = self._expand_to_principal(matched_item, target_doc)
            return self._build_context_response(matched_item, target_doc)
        
        return self._empty_match()

    def _empty_match(self) -> Dict[str, Any]:
        """Return an empty match result."""
        return {"main": None, "before": None, "after": None}

    # =========================================================================
    # CONTEXT BUILDING
    # =========================================================================

    def _build_context_response(self, item: NodeItem, doc: DoclingDocument) -> Dict[str, Any]:
        """
        Construct the response dict (main, before, after) for a given item in a doc.
        
        This logic is the same regardless of how the item was matched (dict or bbox).
        """
        
        def _extract_item_text(node: Optional[NodeItem]) -> Optional[str]:
            if node is None:
                return None
            
            if node.label == DocItemLabel.PICTURE:
                return ""
            
            if hasattr(node, "text") and node.text is not None:
                return node.text
            
            if hasattr(node, "children") and node.children:
                child_texts = []
                for child_ref in node.children:
                    try:
                        child = child_ref.resolve(doc)
                        child_text = _extract_item_text(child)
                        if child_text:
                            child_texts.append(child_text)
                    except Exception:
                        continue
                return "\n".join(child_texts) if child_texts else ""
            
            return ""
        
        prev_sib, next_sib = self._get_context_siblings(doc, item)
        
        start_context = self._expand_to_principal(prev_sib, doc) if prev_sib else None
        end_context = self._expand_to_principal(next_sib, doc) if next_sib else None

        return {
            "main": _extract_item_text(item),
            "before": _extract_item_text(start_context),
            "after": _extract_item_text(end_context),
        }

    def _get_context_siblings(
        self, 
        doc: DoclingDocument, 
        item: NodeItem
    ) -> Tuple[Optional[NodeItem], Optional[NodeItem]]:
        """
        Get the preceding and succeeding siblings in the document structure.
        """
        if not item.parent:
            return None, None
        
        try:
            parent = item.parent.resolve(doc)
        except Exception:
            return None, None

        children_refs = parent.children
        my_cref = item.self_ref
        
        idx = -1
        for i, child_ref in enumerate(children_refs):
            if child_ref.cref == my_cref:
                idx = i
                break
        
        if idx == -1:
            return None, None

        prev_sib = None
        next_sib = None

        if idx > 0:
            try:
                prev_sib = children_refs[idx - 1].resolve(doc)
            except Exception:
                pass

        if idx < len(children_refs) - 1:
            try:
                next_sib = children_refs[idx + 1].resolve(doc)
            except Exception:
                pass
                
        return prev_sib, next_sib

    # =========================================================================
    # ITEM RESOLUTION HELPERS
    # =========================================================================

    def _find_item_by_cref(self, doc: DoclingDocument, cref: str) -> Optional[NodeItem]:
        """Helper to find an item by its self_ref/cref."""
        for item, _ in doc.iterate_items():
            if item.self_ref == cref:
                return item
        return None

    def _expand_to_principal(self, item: NodeItem, doc: DoclingDocument) -> NodeItem:
        """
        If the item is a specific sub-element (like a List Item), return its container.
        Otherwise return the item itself.
        """
        if item.label == DocItemLabel.LIST_ITEM:
            if item.parent:
                try:
                    parent = item.parent.resolve(doc)
                    return parent
                except Exception:
                    print(f"Warning: Could not resolve parent for list item {item.self_ref}")
                    return item
        return item

    # =========================================================================
    # BOUNDING BOX MATCHING
    # =========================================================================

    def _find_aligned_item(
        self, 
        source_item: NodeItem, 
        target_doc: DoclingDocument, 
        target_lang: str
    ) -> Optional[NodeItem]:
        """
        Find the item in target_doc that best matches the bounding box of source_item.
        """
        source_bbox = self._get_bbox(source_item)
        if not source_bbox:
            return None

        # Optimization: Only look at items on the same page
        page_no = source_item.prov[0].page_no if source_item.prov else -1
        if page_no == -1:
            return None

        candidates = self._page_cache.get(target_lang, {}).get(page_no, [])
        if not candidates:
            return None

        return self._matches_bbox(source_item, candidates)

    def _get_bbox(self, item: NodeItem):
        """Helper to get bounding box."""
        if hasattr(item, "prov") and item.prov and len(item.prov) > 0:
            return item.prov[0].bbox
        return None

    def _matches_bbox(
        self, 
        t_item: NodeItem, 
        candidate_items: List[NodeItem]
    ) -> Optional[NodeItem]:
        """
        Find best matching item from candidates based on IoU (priority) then Distance.
        """
        t_bbox = self._get_bbox(t_item)
        if not t_bbox:
            return None

        best_match = None
        best_iou = -1.0
        closest_match = None
        min_dist_sq = float("inf")

        tcx = (t_bbox.l + t_bbox.r) / 2.0
        tcy = (t_bbox.t + t_bbox.b) / 2.0

        for b_item in candidate_items:
            b_bbox = self._get_bbox(b_item)
            if not b_bbox:
                continue

            if t_bbox.coord_origin != b_bbox.coord_origin:
                continue

            iou = t_bbox.intersection_over_union(b_bbox)
            if iou > 0:
                if iou > best_iou:
                    best_iou = iou
                    best_match = b_item

            # Track distance if no good overlap found yet
            if best_iou <= 0:
                bcx = (b_bbox.l + b_bbox.r) / 2.0
                bcy = (b_bbox.t + b_bbox.b) / 2.0
                dist_sq = (tcx - bcx) ** 2 + (tcy - bcy) ** 2

                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_match = b_item

        if best_match:
            return best_match
        return closest_match

    # =========================================================================
    # PAGE CACHING
    # =========================================================================

    def _group_items_by_page(self, doc: DoclingDocument) -> Dict[int, List[NodeItem]]:
        """
        Helper to group all items in a doc by page for fast lookup.
        """
        doc_by_page: Dict[int, List[NodeItem]] = {}
        for item, _ in doc.iterate_items():
            if hasattr(item, "prov") and item.prov and len(item.prov) > 0:
                page_no = item.prov[0].page_no
                if page_no not in doc_by_page:
                    doc_by_page[page_no] = []
                doc_by_page[page_no].append(item)
        return doc_by_page


def get_parallel_data_for_matches(
    retrieval_results: Dict[str, Dict[str, Any]],
    parallel_corpora_path: str = "./parallel_corpora/Class_6-Science",
    initialized_docs: Optional[Dict[int, ParallelAlignedDocument]] = {},
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, ParallelAlignedDocument]]:
    """
    Process retrieval results and get parallel data for each match.
    
    Takes the output from retrieve_top_k_matches and uses ParallelAlignedDocument
    to get parallel content (main, before, after) for each match across languages.
    
    Args:
        retrieval_results: Dict of query results from retrieve_top_k_matches,
            keyed by query_id. Each value contains:
            - "query": The query text
            - "language": The language searched
            - "top_k": List of matches with file, self_ref, chapter_number, similarity
        parallel_corpora_path: Base path to parallel corpora directory containing
            merged documents and match files.
        initialized_docs: Dict of pre-initialized ParallelAlignedDocument objects
            keyed by chapter number. Defaults to empty dict.
            
    Returns:
        Tuple of:
        - Dict keyed by query_id, each value containing:
            - "query": The query text
            - "language": The language searched  
            - "top_k_matches": Dict with keys "match_1", "match_2", etc.
              Each match has: similarity, chapter_number, parallel_data
        - Dict of initialized ParallelAlignedDocument objects keyed by chapter number
    """
    
    results = {}
    
    for query_id, result in retrieval_results.items():
        query = result["query"]
        language = result["language"]
        top_k = result["top_k"]
        
        top_k_matches = {}
        
        for i, match in enumerate(top_k):
            chapter_num = match["chapter_number"]
            self_ref = match["self_ref"]
            similarity = match["similarity"]
            
            # Initialize ParallelAlignedDocument for this chapter if not already done
            if chapter_num not in initialized_docs:
                initialized_docs[chapter_num] = _initialize_parallel_doc(
                    chapter_num, parallel_corpora_path
                )
            
            doc = initialized_docs[chapter_num]
            
            # Get parallel data for this match
            try:
                parallel_data = doc.retrieve_parallel_data(self_ref, language)
            except Exception as e:
                print(f"Warning: Could not retrieve parallel data for {self_ref}: {e}")
                parallel_data = {"eng": None, "hi": None, "ta": None}
            
            # Build match result
            top_k_matches[f"match_{i + 1}"] = {
                "similarity": float(similarity),
                "chapter_number": chapter_num,
                "parallel_data": parallel_data,
            }
        
        results[query_id] = {
            "query": query,
            "language": language,
            "top_k_matches": top_k_matches,
        }
    
    return results, initialized_docs


def _initialize_parallel_doc(
    chapter_num: int,
    parallel_corpora_path: str,
) -> ParallelAlignedDocument:
    """
    Initialize a ParallelAlignedDocument for a specific chapter.
    
    Loads English, Hindi, and Tamil merged documents along with precomputed
    English-Hindi match dictionaries from the parallel_corpora directory.
    
    Args:
        chapter_num: The chapter number to load.
        parallel_corpora_path: Base path to parallel corpora (e.g., "./parallel_corpora/Class_6-Science").
        
    Returns:
        Initialized ParallelAlignedDocument for the chapter.
    """
    chapter_dir = os.path.join(parallel_corpora_path, f"Chapter_{chapter_num}")
    
    # Build paths for each language's merged document
    eng_path = os.path.join(chapter_dir, "merged_english.json")
    hi_path = os.path.join(chapter_dir, "merged_hindi.json")
    ta_path = os.path.join(chapter_dir, "merged_tamil.json")
    
    # Build paths for match dictionaries
    eng_to_hin_path = os.path.join(chapter_dir, "eng_to_hin.json")
    hin_to_eng_path = os.path.join(chapter_dir, "hin_to_eng.json")
    
    print(f"Initializing ParallelAlignedDocument for Chapter {chapter_num}...")
    
    # Load document JSONs
    with open(eng_path, "r") as f:
        data_en = json.load(f)
    with open(hi_path, "r") as f:
        data_hi = json.load(f)
    with open(ta_path, "r") as f:
        data_ta = json.load(f)
    
    # Load precomputed match dictionaries if available
    eng_to_hi = {}
    hi_to_eng = {}
    if os.path.exists(eng_to_hin_path):
        with open(eng_to_hin_path, "r") as f:
            eng_to_hi = json.load(f)
    if os.path.exists(hin_to_eng_path):
        with open(hin_to_eng_path, "r") as f:
            hi_to_eng = json.load(f)
    
    return ParallelAlignedDocument(
        data_en=data_en,
        data_hi=data_hi,
        data_ta=data_ta,
        eng_to_hi_matches=eng_to_hi,
        hi_to_eng_matches=hi_to_eng,
    )


def format_match_as_text(
    match_data: Dict[str, Any],
    language: str = "eng",
) -> str:
    """
    Format a single match's parallel data as a complete text string.
    
    Takes one match from top_k_matches (e.g., result["top_k_matches"]["match_1"])
    and formats the before/main/after content as a single readable string.
    
    Args:
        match_data: A single match dict containing "parallel_data" with
            language keys ("eng", "hi", "ta"), each having "main", "before", "after".
        language: Which language's text to format ("eng", "hi", or "ta").
            
    Returns:
        A formatted string with before, main, and after separated by blank lines:
        
        before
        
        main
        
        after
    """
    parallel_data = match_data.get("parallel_data", {})
    lang_data = parallel_data.get(language, {})
    
    # Extract before, main, and after text
    before_text = lang_data.get("before", "") if lang_data else ""
    main_text = lang_data.get("main", "") if lang_data else ""
    after_text = lang_data.get("after", "") if lang_data else ""
    
    # Format as a complete string with blank line separators
    parts = []
    if before_text:
        parts.append(before_text)
    if main_text:
        parts.append(main_text)
    if after_text:
        parts.append(after_text)
    
    return "\n\n".join(parts)
