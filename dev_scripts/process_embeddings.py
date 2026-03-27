"""
Process Embeddings: Pre-compute sentence embeddings for all languages
=====================================================================

Generates and saves sentence embeddings for all merged documents across
English, Hindi, and Tamil. The embeddings are saved to outputs_embeddings/
and used by Workflow 3 (cross-lingual retrieval) for semantic search.

Usage:
    python dev_scripts/process_embeddings.py

Output:
    outputs_embeddings/{Language}/embeddings.npz  — numpy array of embeddings
    outputs_embeddings/{Language}/refs.json        — reference index mapping
"""

import os
import sys

# Add parent directory to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sentence_transformers import SentenceTransformer  # noqa: E402
from src.retreival.preprocess import embed_glob_texts  # noqa: E402

# Load the multilingual embedding model once
model = SentenceTransformer("krutrim-ai-labs/vyakyarth")
print("Model Loaded")

# Process each language's merged documents
for lang in ["English", "Hindi", "Tamil"]:
    files = os.path.join(PROJECT_ROOT, f"outputs_split/Class_6-Science-{lang}/*/merged.json")
    save_to = os.path.join(PROJECT_ROOT, f"outputs_embeddings/{lang}/")

    embed_glob_texts(files, model, save_to_dir=save_to)
    print("Processed language ", lang)
