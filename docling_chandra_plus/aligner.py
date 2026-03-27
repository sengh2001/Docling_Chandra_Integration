import torch
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from docling_core.types.doc.document import DoclingDocument

class ChandraAligner:
    """
    Semantic Aligner for matching English and Hindi blocks in DoclingDocuments.
    Uses Bi-Encoders for retrieval and Cross-Encoders for reranking.
    """

    BI_ENCODER_MODEL = "intfloat/multilingual-e5-base"
    CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    
    TOP_K = 5
    CHUNK_SIZE = 400

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.bi_encoder = None
        self.cross_encoder = None

    def _lazy_load_models(self):
        """Load models only when needed to save memory."""
        if not self.bi_encoder:
            print(f"Loading Bi-Encoder: {self.BI_ENCODER_MODEL}...")
            self.bi_encoder = SentenceTransformer(self.BI_ENCODER_MODEL, device=self.device)
        if not self.cross_encoder:
            print(f"Loading Cross-Encoder: {self.CROSS_ENCODER_MODEL}...")
            self.cross_encoder = CrossEncoder(self.CROSS_ENCODER_MODEL, device=self.device)

    def _chunk_text(self, text: str) -> List[str]:
        text = text.strip()
        if len(text) <= self.CHUNK_SIZE:
            return [text]
        return [text[i : i + self.CHUNK_SIZE] for i in range(0, len(text), self.CHUNK_SIZE)]

    def _embed_blocks(self, blocks: Dict[str, str]):
        all_chunks = []
        block_ids = []
        block_to_chunk_indices = {}

        for block_id, text in blocks.items():
            chunks = self._chunk_text(text)
            indices = []
            for chunk in chunks:
                indices.append(len(all_chunks))
                all_chunks.append(chunk)
                block_ids.append(block_id)
            block_to_chunk_indices[block_id] = indices

        embeddings = self.bi_encoder.encode(all_chunks, convert_to_tensor=True)
        return block_ids, embeddings, block_to_chunk_indices

    def align(self, english_doc: DoclingDocument, hindi_doc: DoclingDocument) -> Dict[str, str]:
        """
        Align English text items to their best matching Hindi counterparts.
        
        Returns:
            Mapping of English self_ref -> Hindi self_ref.
        """
        self._lazy_load_models()

        # Extract text blocks
        eng_blocks = {t.self_ref: t.text for t in english_doc.texts if t.text}
        hin_blocks = {t.self_ref: t.text for t in hindi_doc.texts if t.text}

        print(f"Aligning {len(eng_blocks)} English blocks to {len(hin_blocks)} Hindi blocks...")

        # Embed both
        eng_ids, eng_embs, eng_map = self._embed_blocks(eng_blocks)
        hin_ids, hin_embs, hin_map = self._embed_blocks(hin_blocks)

        results = {}

        for eng_ref, eng_text in eng_blocks.items():
            # 1. Retrieval
            scores = []
            for qi in eng_map[eng_ref]:
                hits = util.semantic_search(eng_embs[qi].unsqueeze(0), hin_embs, top_k=self.TOP_K)[0]
                scores.extend(hits)

            # Aggregate scores
            best_scores = {}
            for hit in scores:
                cid = hit["corpus_id"]
                score = hit["score"]
                best_scores[cid] = max(best_scores.get(cid, 0.0), score)

            candidate_hin_refs = list({hin_ids[idx] for idx in sorted(best_scores.keys(), key=lambda i: best_scores[i], reverse=True)[:self.TOP_K]})

            # 2. Reranking
            if not candidate_hin_refs:
                continue
                
            pairs = [[eng_text, hin_blocks[ref]] for ref in candidate_hin_refs]
            cross_scores = self.cross_encoder.predict(pairs)
            best_idx = int(cross_scores.argmax())
            results[eng_ref] = candidate_hin_refs[best_idx]

        return results
