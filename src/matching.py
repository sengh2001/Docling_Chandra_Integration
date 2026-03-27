"""
Cross-lingual Textbook Block Alignment (English <-> Hindi)
=========================================================

This script aligns English and Hindi textbook blocks using:

1. A multilingual bi-encoder for fast candidate retrieval
2. A multilingual cross-encoder for accurate reranking

Input:
- english_blocks: Dict[str, str]  (id -> English text)
- hindi_blocks:   Dict[str, str]  (id -> Hindi text)

Output:
- eng_to_hin: Dict[str, str]  (English id -> best Hindi id)
- hin_to_eng: Dict[str, str]  (Hindi id -> best English id)

Requirements:
- pip install sentence-transformers torch

Models (out of the box, no training):
- Bi-encoder:  intfloat/multilingual-e5-base
- Cross-encoder: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
"""

from typing import Dict, List, Tuple
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util


# -----------------------------
# Configuration
# -----------------------------

BI_ENCODER_MODEL = "intfloat/multilingual-e5-base"
CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Number of top candidates retrieved by bi-encoder before reranking
TOP_K = 5

# Chunk size (in characters) for long blocks
CHUNK_SIZE = 400


# -----------------------------
# Helper functions
# -----------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Splits long text into roughly equal-sized chunks.
    This helps embeddings handle long textbook blocks.

    Very simple heuristic: fixed-size character chunks.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    return [
        text[i : i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


def embed_blocks(
    model: SentenceTransformer,
    blocks: Dict[str, str],
) -> Tuple[List[str], torch.Tensor, Dict[str, List[int]]]:
    """
    Embeds blocks by first chunking them, then embedding all chunks.

    Returns:
    - block_ids: list of block ids repeated per chunk
    - embeddings: tensor of shape (num_chunks, dim)
    - block_to_chunk_indices: mapping block_id -> indices in embeddings
    """
    all_chunks = []
    block_ids = []
    block_to_chunk_indices = {}

    for block_id, text in blocks.items():
        chunks = chunk_text(text)
        indices = []

        for chunk in chunks:
            indices.append(len(all_chunks))
            all_chunks.append(chunk)
            block_ids.append(block_id)

        block_to_chunk_indices[block_id] = indices

    embeddings = model.encode(
        all_chunks,
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    return block_ids, embeddings, block_to_chunk_indices


def retrieve_candidates(
    query_indices: List[int],
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    top_k: int,
) -> List[int]:
    """
    Given indices for query chunks, retrieve top-k corpus chunk indices
    using max similarity across chunks.
    """
    scores = []

    for qi in query_indices:
        hits = util.semantic_search(
            query_embeddings[qi].unsqueeze(0),
            corpus_embeddings,
            top_k=top_k,
        )[0]
        scores.extend(hits)

    # Aggregate by corpus index, keep max score
    best_scores = {}
    for hit in scores:
        cid = hit["corpus_id"]
        score = hit["score"]
        best_scores[cid] = max(best_scores.get(cid, 0.0), score)

    # Return top-k corpus indices
    return sorted(
        best_scores.keys(),
        key=lambda i: best_scores[i],
        reverse=True,
    )[:top_k]


def rerank_with_cross_encoder(
    cross_encoder: CrossEncoder,
    query_text: str,
    candidate_texts: List[Tuple[str, str]],
) -> str:
    """
    Uses a cross-encoder to pick the best candidate.

    candidate_texts: list of (candidate_id, candidate_text)
    Returns: best candidate_id
    """
    pairs = [[query_text, text] for _, text in candidate_texts]
    scores = cross_encoder.predict(pairs)

    best_idx = int(scores.argmax())
    return candidate_texts[best_idx][0]


# -----------------------------
# Core alignment logic
# -----------------------------

def align_blocks(
    source_blocks: Dict[str, str],
    target_blocks: Dict[str, str],
    source_embeddings_data: Tuple[List[str], torch.Tensor, Dict[str, List[int]]],
    target_embeddings_data: Tuple[List[str], torch.Tensor, Dict[str, List[int]]],
    cross_encoder: CrossEncoder,
) -> Dict[str, str]:
    """
    Aligns each source block to its best matching target block.

    This is directional (argmax per source block).
    """
    # Unpack pre-computed embeddings
    src_ids, src_embs, src_map = source_embeddings_data
    tgt_ids, tgt_embs, tgt_map = target_embeddings_data

    result = {}

    # 1. Retrieval Phase
    print(f"Retrieving candidates for {len(source_blocks)} blocks...")
    candidates_map = {}

    for src_id, src_text in source_blocks.items():
        # Retrieve candidate target chunks
        candidate_chunk_indices = retrieve_candidates(
            src_map[src_id],
            src_embs,
            tgt_embs,
            TOP_K,
        )

        # Map chunk indices to unique target block ids
        candidate_block_ids = list({
            tgt_ids[idx] for idx in candidate_chunk_indices
        })

        # Prepare candidate texts for reranking
        candidates = [
            (bid, target_blocks[bid]) for bid in candidate_block_ids
        ]
        candidates_map[src_id] = candidates

    # 2. Reranking Phase
    print(f"Reranking candidates for {len(source_blocks)} blocks...")
    result = {}

    for src_id, candidates in candidates_map.items():
        # Cross-encoder reranking
        best_match = rerank_with_cross_encoder(
            cross_encoder,
            source_blocks[src_id],
            candidates,
        )

        result[src_id] = best_match

    return result


# -----------------------------
# Public API function
# -----------------------------

def align_english_hindi_blocks(
    english_blocks: Dict[str, str],
    hindi_blocks: Dict[str, str],
    bi_encoder: SentenceTransformer,
    cross_encoder: CrossEncoder,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Main entry point for aligning English and Hindi blocks.

    Args:
        english_blocks: Dict mapping English block ids to text.
        hindi_blocks: Dict mapping Hindi block ids to text.
        bi_encoder: Pre-loaded SentenceTransformer model for embeddings.
        cross_encoder: Pre-loaded CrossEncoder model for reranking.

    Returns:
        Tuple of (english_to_hindi, hindi_to_english) mappings.
    """

    print("Embedding English blocks...")
    english_embeddings = embed_blocks(bi_encoder, english_blocks)

    print("Embedding Hindi blocks...")
    hindi_embeddings = embed_blocks(bi_encoder, hindi_blocks)

    print("Aligning English -> Hindi...")
    english_to_hindi = align_blocks(
        english_blocks,
        hindi_blocks,
        english_embeddings,
        hindi_embeddings,
        cross_encoder,
    )

    print("Aligning Hindi -> English...")
    hindi_to_english = align_blocks(
        hindi_blocks,
        english_blocks,
        hindi_embeddings,
        english_embeddings,
        cross_encoder,
    )

    return english_to_hindi, hindi_to_english


# -----------------------------
# Example usage (remove in prod)
# -----------------------------
if __name__ == "__main__":
    # Load models once
    print("Loading models...")
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    # Dummy example
    english_blocks = {
        "E1": "Force can change the shape or speed of an object.",
        "E2": "Plants make their own food using sunlight.",
    }

    hindi_blocks = {
        "H1": "बल किसी वस्तु की गति या आकार बदल सकता है।",
        "H2": "पौधे सूर्य के प्रकाश से अपना भोजन बनाते हैं।",
    }

    eng2hin, hin2eng = align_english_hindi_blocks(
        english_blocks,
        hindi_blocks,
        bi_encoder,
        cross_encoder,
    )

    print("\nEnglish -> Hindi")
    print(eng2hin)

    print("\nHindi -> English")
    print(hin2eng)