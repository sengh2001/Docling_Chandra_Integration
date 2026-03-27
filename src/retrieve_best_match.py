"""
Retrieve Best Match: RAG-based document retrieval
==================================================

This module provides simple functions to retrieve the most relevant document
items for a given query using semantic search with sentence embeddings.

Embeddings are cached to disk for efficiency. If embeddings don't exist,
they are computed and saved automatically.
"""

import os
import json
from typing import Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retreival.preprocess import embed_glob_texts
from src.retreival.rets import top_k_similar_refs



# Module-level cache for the embedding model (loaded once, reused)
_embedding_model: SentenceTransformer = None

# Language code to directory name mapping
LANG_DIR_MAP = {"eng": "English", "hi": "Hindi", "ta": "Tamil"}


def get_embedding_model() -> SentenceTransformer:
    """
    Get the sentence transformer model, loading it lazily on first access.
    
    The model is cached at module level to avoid reloading on each call.
    
    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    global _embedding_model
    
    if _embedding_model is None:
        print("Loading sentence transformer model: krutrim-ai-labs/vyakyarth")
        _embedding_model = SentenceTransformer("krutrim-ai-labs/vyakyarth")
    
    return _embedding_model


def load_or_compute_embeddings(
    language: str,
    embeddings_dir: str = "outputs_embeddings",
    docs_glob_template: str = "./outputs_split/Class_6-Science-{lang}/*/merged.json",
) -> Tuple[np.ndarray, dict]:
    """
    Load precomputed embeddings from disk, or compute and save them if not available.
    
    Args:
        language: Language code ("eng", "hi", or "ta").
        embeddings_dir: Base directory for embedding storage.
        docs_glob_template: Glob template for finding document JSON files.
                            Use {lang} as placeholder for language name.
        
    Returns:
        Tuple[np.ndarray, dict]: (embeddings array, refs dictionary)
        
    Raises:
        ValueError: If language is not one of the supported languages.
    """
    if language not in LANG_DIR_MAP:
        raise ValueError(f"Unknown language: {language}. Must be one of {list(LANG_DIR_MAP.keys())}")
    
    lang_dir_name = LANG_DIR_MAP[language]
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
    model = get_embedding_model()
    embeddings, refs_list, file_ranges = embed_glob_texts(
        docs_glob, model, save_to_dir=save_dir
    )
    
    # Build refs_dict in the expected format
    refs_dict = {"refs": refs_list, "file_ranges": file_ranges}
    
    print(f"Embeddings computed and saved to {save_dir}/")
    
    return embeddings, refs_dict


def retrieve_top_k_matches(
    queries: Dict[str, Dict[str, str]],
    k: int = 5,
    embeddings_dir: str = "outputs_embeddings",
    docs_glob_template: str = "./outputs_split/Class_6-Science-{lang}/*/merged.json",
) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve the top k most relevant document items for a dict of queries.
    
    Uses semantic search with sentence embeddings to find matching documents.
    Embeddings are loaded from cache or computed and saved if not available.
    
    Args:
        queries: Dict of query dicts, keyed by query_id.
            Each query dict should have:
            - "query": The search query text
            - "language": The language to search in ("eng", "hi", or "ta")
        k: Number of top results to return per query.
        embeddings_dir: Directory where embeddings are stored/cached.
        docs_glob_template: Glob pattern template for document files.
        
    Returns:
        Dict keyed by query_id, each value containing:
        - "query": The original query text
        - "language": The language searched
        - "top_k": List of top k matches, each with file, self_ref, chapter_number, similarity
        
    Raises:
        ValueError: If language is unsupported.
    """
    # Get the embedding model (loaded once)
    model = get_embedding_model()
    
    # Cache embeddings by language to avoid reloading
    embeddings_cache: Dict[str, Tuple[Any, Dict]] = {}
    
    results = {}
    for query_id, query_data in queries.items():
        query = query_data["query"]
        language = query_data["language"]
        
        # Load embeddings for this language if not cached
        if language not in embeddings_cache:
            embeddings_cache[language] = load_or_compute_embeddings(
                language=language,
                embeddings_dir=embeddings_dir,
                docs_glob_template=docs_glob_template,
            )
        
        embeddings, refs_dict = embeddings_cache[language]
        
        # Find top k most similar document items for this query
        top_results = top_k_similar_refs(embeddings, query, model, refs_dict, k=k)
        
        results[query_id] = {
            "query": query,
            "language": language,
            "top_k": top_results if top_results else [],
        }
    
    return results
