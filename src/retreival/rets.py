"""
Retrieval: Embedding similarity search and reference resolution
===============================================================

Provides functions to find the most similar document items to a query
using cosine similarity on precomputed embeddings, and to resolve
matching indices back to file paths and document references.

Used by src/retrieve_best_match.py and src/alignment.py.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


def _find_file(intervals, idx):
    """
    Binary search to find which file an embedding index belongs to.

    Args:
        intervals: Sorted list of dicts with 'start', 'end', 'file' keys
            describing the index range for each source file.
        idx: The embedding index to look up.

    Returns:
        File path string, or None if not found.
    """
    lo, hi = 0, len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s, e, f = intervals[mid]["start"], intervals[mid]["end"], intervals[mid]["file"]

        if s <= idx <= e:
            return f
        elif idx < s:
            hi = mid - 1
        else:
            lo = mid + 1
    return None


def get_index_ref(idx: int, refs_dict: dict):
    """
    Resolve an embedding index to its source file, document reference, and chapter number.

    Args:
        idx: The embedding index.
        refs_dict: Dict with 'refs' (list of text item indices) and
            'file_ranges' (list of start/end/file dicts).

    Returns:
        Dict with 'file', 'self_ref', and 'chapter_number' keys.
    """
    intervals = sorted(refs_dict["file_ranges"], key=lambda x: x["start"])

    file_path = _find_file(intervals, idx)
    ref = f"#/texts/{refs_dict['refs'][idx]}"

    # Extract chapter number from file path (e.g., "Chapter_3" -> 3)
    chapter_match = re.search(r"Chapter_(\d+)", file_path) if file_path else None
    chapter_number = int(chapter_match.group(1)) if chapter_match else -1

    return {"file": file_path, "self_ref": ref, "chapter_number": chapter_number}


def top_k_similar(
    embeddings: np.ndarray, target: str, model: SentenceTransformer, k: int = 10
):
    """
    Find the top-k most similar embeddings to a target query string.

    Args:
        embeddings: Pre-computed embedding matrix (num_items × dim).
        target: Query text to encode and compare against.
        model: SentenceTransformer model for encoding the query.
        k: Number of top results to return.

    Returns:
        List of (index, similarity_score) tuples, sorted descending.
    """
    target_embedding = np.array(model.encode([target]))
    similarities = cosine_similarity(target_embedding, embeddings)[0]
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return [(i, similarities[i]) for i in top_k_indices]


def top_k_similar_refs(
    embeddings: np.ndarray,
    target: str,
    model: SentenceTransformer,
    refs_dict: dict,
    k: int = 10,
):
    """
    Find the top-k most similar document items and resolve their references.

    Combines top_k_similar with get_index_ref to return full reference
    information including file path, document cref, and chapter number.

    Args:
        embeddings: Pre-computed embedding matrix.
        target: Query text to search for.
        model: SentenceTransformer model for encoding.
        refs_dict: Reference dictionary from embed_glob_texts.
        k: Number of top results to return.

    Returns:
        List of dicts with 'file', 'self_ref', 'chapter_number', 'similarity'.
    """
    top_k = top_k_similar(embeddings, target, model, k)
    refs = []
    for idx, simi in top_k:
        idx_ref = get_index_ref(idx, refs_dict)
        idx_ref["similarity"] = simi
        refs.append(idx_ref)

    return refs
