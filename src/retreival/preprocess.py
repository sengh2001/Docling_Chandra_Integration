"""
Retrieval Preprocessing: Text chunking and embedding generation
===============================================================

Provides functions to chunk DoclingDocument text items and generate
sentence embeddings for semantic search. Supports batch processing
of multiple documents via glob patterns.

Used by dev_scripts/process_embeddings.py and src/retrieve_best_match.py.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from docling_core.types.doc import DoclingDocument, NodeItem
import glob
import os
import json


def chunk_text_with_overlap(
    items: NodeItem, max_seq_len: int, overlap: int
):  # -> tuple[list[str], list[str]]:
    """
    Chunks text from items based on max_seq_len with overlap between chunks.

    Args:
        items: List of objects with 'text' and 'self_ref' attributes
        max_seq_len: Maximum length of each chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        Tuple of (chunked_texts, corresponding_self_refs)
        The refs are the int part of '#/texts/[int]'
    """
    chunked_texts = []
    corresponding_refs = []

    for item in sorted(items, key=lambda x: int(x.self_ref[len("#/texts/") :])):
        text = item.text
        ref = int(item.self_ref[len("#/texts/") :])

        if len(text) <= max_seq_len:
            chunked_texts.append(text)
            corresponding_refs.append(ref)
        else:
            # Split text into chunks with overlap
            start = 0
            while start < len(text):
                end = start + max_seq_len
                chunk = text[start:end]
                chunked_texts.append(chunk)
                corresponding_refs.append(ref)

                # Move start position by (max_seq_len - overlap) for next iteration
                start += max_seq_len - overlap

    return chunked_texts, corresponding_refs


def embed_doc_texts(
    doc: DoclingDocument,
    model: SentenceTransformer,
    max_seq_len: int = 100,
    chunk_overlap: int = 40,
):
    """
    Chunk and embed all text items from a single DoclingDocument.

    Args:
        doc: The DoclingDocument to process.
        model: SentenceTransformer model for encoding.
        max_seq_len: Maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        Tuple of (embeddings array, list of text item ref indices).
    """
    text_only = [itm for itm, _ in doc.iterate_items() if itm.self_ref[2:7] == "texts"]
    chunked_text, chunked_refs = chunk_text_with_overlap(
        text_only, max_seq_len, chunk_overlap
    )
    embeddings = np.array(model.encode(chunked_text))
    return embeddings, chunked_refs


def embed_glob_texts(
    docs_glob: str,
    model: SentenceTransformer,
    max_seq_len: int = 100,
    chunk_overlap: int = 40,
    save_to_dir: str | None = None,
):
    """Generates the embeddings for all the docling json files described by the glob, then stores the generated embeddings along with which file it came from and what the reference for the embedding is.

    Args:
        docs_glob (str): glob for valid docstring jsons
        model (SentenceTransformer):
        max_seq_len (int, optional): Defaults to 100.
        chunk_overlap (int, optional): Defaults to 40.
        save_to_dir (str, optional): If provided, saves the embedding and ref list to the dir
    """
    embedding_list = []
    refs_list = []
    file_ranges = []
    ptr = 0

    file_paths = glob.glob(docs_glob)
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            raise OSError(f"File not found: {file_path}")
        doc = DoclingDocument.load_from_json(file_path)
        embs, refs = embed_doc_texts(doc, model, max_seq_len, chunk_overlap)

        embedding_list.append(embs)
        refs_list.extend(refs)
        file_ranges.append(
            {"start": ptr, "end": ptr + len(refs) - 1, "file": file_path}
        )
        ptr = ptr + len(refs)

    embeddings = np.vstack(embedding_list)

    if save_to_dir is not None:
        os.makedirs(save_to_dir, exist_ok=True)
        # Save embeddings in NPZ format (optimal for numpy arrays)
        embeddings_path = os.path.join(save_to_dir, "embeddings.npz")
        np.savez_compressed(embeddings_path, embeddings)

        # Save refs_list as JSON
        refs_path = os.path.join(save_to_dir, "refs.json")
        with open(refs_path, "w") as f:
            json.dump({"refs": refs_list, "file_ranges": file_ranges}, f)

    return embeddings, refs_list, file_ranges
