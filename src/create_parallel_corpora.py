"""
Complete PDF Processing Workflow for Parallel Corpora Creation
===============================================================

This script orchestrates a 3-step workflow:
1. Scan PDFs and save to outputs_split/
2. Merge scanned documents and save to parallel_corpora/
3. Align Hindi-English blocks and save matchings

Each step checks for precomputed results and skips if already done.
All chapters complete each step before moving to the next step.

Run from docling_ocr_extraction/ directory:
    python src/create_parallel_corpora.py
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict

# Add parent directory to path for imports to work when running script directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from docling_core.types.doc import DoclingDocument

# Import from existing modules
from src.processing.docling_pipeline import docling_ocr_pipeline, create_converter
from src.processing.pdf_utils import extract_save_text_layer, remove_watermark
from src.merge_document_scans import merge_documents
from src.matching import (
    align_english_hindi_blocks,
    BI_ENCODER_MODEL,
    CROSS_ENCODER_MODEL,
)
from sentence_transformers import SentenceTransformer, CrossEncoder


# -----------------------------
# Configuration
# -----------------------------

LANGUAGES = {"English": "eng", "Hindi": "hin", "Tamil": "tam"}
CHAPTERS = list(range(1, 13))  # Chapters 1-12

# Paths (relative to docling_ocr_extraction/)
BASE_DIR = Path(__file__).parent.parent
BOOKS_DIR = BASE_DIR / "books"
OUTPUTS_DIR = BASE_DIR / "outputs_split"
PARALLEL_CORPORA_DIR = BASE_DIR / "parallel_corpora" / "Class_6-Science"
SCRATCH_DIR = BASE_DIR / "scratch"


# -----------------------------
# Step 1: PDF Scanning
# -----------------------------

def scan_chapter(lang_name: str, lang_code: str, chap_num: int) -> bool:
    """
    Scan a single chapter's PDF and save parsed_original.json and parsed_text.json.
    
    Returns True if scanning was performed, False if skipped.
    """
    input_pdf = BOOKS_DIR / f"Class_6-Science-{lang_name}" / f"Chapter_{chap_num}.pdf"
    output_dir = OUTPUTS_DIR / f"Class_6-Science-{lang_name}" / f"Chapter_{chap_num}"
    
    original_json = output_dir / "parsed_original.json"
    text_json = output_dir / "parsed_text.json"
    
    # Skip if already computed
    if original_json.exists() and text_json.exists():
        print(f"  [SKIP] {lang_name} Chapter {chap_num} - already scanned")
        return False
    
    # Check if input PDF exists
    if not input_pdf.exists():
        print(f"  [WARN] {lang_name} Chapter {chap_num} - PDF not found: {input_pdf}")
        return False
    
    print(f"  [SCAN] {lang_name} Chapter {chap_num}...")
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create converter for this language
    converter = create_converter(lang_code)
    
    # Scratch file for processing
    pid = os.getpid()
    scratch_pdf = SCRATCH_DIR / f"temp_{pid}.pdf"
    
    try:
        # Process original document (without watermark)
        remove_watermark(str(input_pdf), str(scratch_pdf))
        docling_ocr_pipeline(str(input_pdf), converter, output_json=str(original_json))
        
        # Process text-only version
        extract_save_text_layer(str(input_pdf), str(scratch_pdf))
        docling_ocr_pipeline(str(scratch_pdf), converter, output_json=str(text_json))
        
        print(f"  [DONE] {lang_name} Chapter {chap_num}")
        return True
        
    finally:
        # Cleanup scratch file
        if scratch_pdf.exists():
            scratch_pdf.unlink()


def run_step1_scanning():
    """Run scanning for all chapters and languages, skipping those already done."""
    print("\n" + "=" * 60)
    print("STEP 1: PDF Scanning")
    print("=" * 60)
    
    total_scanned = 0
    total_skipped = 0
    
    for chap_num in CHAPTERS:
        print(f"\n--- Chapter {chap_num} ---")
        for lang_name, lang_code in LANGUAGES.items():
            if scan_chapter(lang_name, lang_code, chap_num):
                total_scanned += 1
            else:
                total_skipped += 1
    
    print(f"\nStep 1 Complete: {total_scanned} scanned, {total_skipped} skipped")


# -----------------------------
# Step 2: Document Merging
# -----------------------------

def merge_chapter(chap_num: int) -> Dict[str, bool]:
    """
    Merge parsed_original.json + parsed_text.json -> merged_{lang}.json
    for all 3 languages. Save to parallel_corpora/Class_6-Science/Chapter_{chap_num}/
    
    Returns dict mapping language -> whether merge was performed.
    """
    chapter_output_dir = PARALLEL_CORPORA_DIR / f"Chapter_{chap_num}"
    results = {}
    
    for lang_name in LANGUAGES:
        merged_path = chapter_output_dir / f"merged_{lang_name.lower()}.json"
        
        # Skip if already computed
        if merged_path.exists():
            print(f"  [SKIP] {lang_name} Chapter {chap_num} - already merged")
            results[lang_name] = False
            continue
        
        # Source files from outputs_split
        source_dir = OUTPUTS_DIR / f"Class_6-Science-{lang_name}" / f"Chapter_{chap_num}"
        original_json = source_dir / "parsed_original.json"
        text_json = source_dir / "parsed_text.json"
        
        # Check if source files exist
        if not original_json.exists() or not text_json.exists():
            print(f"  [WARN] {lang_name} Chapter {chap_num} - source files missing")
            results[lang_name] = False
            continue
        
        print(f"  [MERGE] {lang_name} Chapter {chap_num}...")
        
        # Create output directory
        chapter_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load documents
        baseline_doc = DoclingDocument.load_from_json(original_json)
        text_only_doc = DoclingDocument.load_from_json(text_json)
        
        # Merge
        merged_doc = merge_documents(baseline_doc, text_only_doc)
        
        # Save
        merged_doc.save_as_json(merged_path)
        
        print(f"  [DONE] {lang_name} Chapter {chap_num} -> {merged_path}")
        results[lang_name] = True
    
    return results


def run_step2_merging():
    """Run merging for all chapters, skipping those already done."""
    print("\n" + "=" * 60)
    print("STEP 2: Document Merging")
    print("=" * 60)
    
    total_merged = 0
    total_skipped = 0
    
    for chap_num in CHAPTERS:
        print(f"\n--- Chapter {chap_num} ---")
        results = merge_chapter(chap_num)
        for was_merged in results.values():
            if was_merged:
                total_merged += 1
            else:
                total_skipped += 1
    
    print(f"\nStep 2 Complete: {total_merged} merged, {total_skipped} skipped")


# -----------------------------
# Step 3: Hindi-English Alignment
# -----------------------------

def extract_text_blocks(doc: DoclingDocument) -> Dict[str, str]:
    """
    Extract text blocks from a DoclingDocument.
    Returns dict mapping self_ref -> text.
    """
    blocks = {}
    for text_item in doc.texts:
        if text_item.text and text_item.text.strip():
            blocks[text_item.self_ref] = text_item.text
    return blocks


def align_chapter(
    chap_num: int,
    bi_encoder: SentenceTransformer,
    cross_encoder: CrossEncoder,
) -> bool:
    """
    Align English and Hindi blocks for a chapter.
    Save eng_to_hin.json and hin_to_eng.json to parallel corpora.
    
    Returns True if alignment was performed, False if skipped.
    """
    chapter_dir = PARALLEL_CORPORA_DIR / f"Chapter_{chap_num}"
    
    eng_to_hin_path = chapter_dir / "eng_to_hin.json"
    hin_to_eng_path = chapter_dir / "hin_to_eng.json"
    
    # Skip if already computed
    if eng_to_hin_path.exists() and hin_to_eng_path.exists():
        print(f"  [SKIP] Chapter {chap_num} - already aligned")
        return False
    
    # Load merged documents
    english_merged = chapter_dir / "merged_english.json"
    hindi_merged = chapter_dir / "merged_hindi.json"
    
    if not english_merged.exists() or not hindi_merged.exists():
        print(f"  [WARN] Chapter {chap_num} - merged documents missing")
        return False
    
    print(f"  [ALIGN] Chapter {chap_num}...")
    
    # Load documents
    english_doc = DoclingDocument.load_from_json(english_merged)
    hindi_doc = DoclingDocument.load_from_json(hindi_merged)
    
    # Extract text blocks
    english_blocks = extract_text_blocks(english_doc)
    hindi_blocks = extract_text_blocks(hindi_doc)
    
    print(f"    English blocks: {len(english_blocks)}, Hindi blocks: {len(hindi_blocks)}")
    
    # Run alignment
    eng_to_hin, hin_to_eng = align_english_hindi_blocks(
        english_blocks,
        hindi_blocks,
        bi_encoder,
        cross_encoder,
    )
    
    # Save results
    with open(eng_to_hin_path, "w", encoding="utf-8") as f:
        json.dump(eng_to_hin, f, ensure_ascii=False, indent=2)
    
    with open(hin_to_eng_path, "w", encoding="utf-8") as f:
        json.dump(hin_to_eng, f, ensure_ascii=False, indent=2)
    
    print(f"  [DONE] Chapter {chap_num}")
    print(f"    -> {eng_to_hin_path}")
    print(f"    -> {hin_to_eng_path}")
    
    return True


def run_step3_alignment():
    """Run alignment for all chapters, models loaded once."""
    print("\n" + "=" * 60)
    print("STEP 3: Hindi-English Alignment")
    print("=" * 60)
    
    # Load models once
    print("\nLoading models...")
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    print("Models loaded.\n")
    
    total_aligned = 0
    total_skipped = 0
    
    for chap_num in CHAPTERS:
        print(f"\n--- Chapter {chap_num} ---")
        if align_chapter(chap_num, bi_encoder, cross_encoder):
            total_aligned += 1
        else:
            total_skipped += 1
    
    print(f"\nStep 3 Complete: {total_aligned} aligned, {total_skipped} skipped")


# -----------------------------
# Main Orchestrator
# -----------------------------

def run_full_workflow():
    """
    Execute all 3 steps in sequence.
    Each step completes for ALL chapters before moving to next step.
    """
    print("=" * 60)
    print("PARALLEL CORPORA CREATION WORKFLOW")
    print("=" * 60)
    print(f"Languages: {list(LANGUAGES.keys())}")
    print(f"Chapters: {CHAPTERS}")
    print(f"Output: {PARALLEL_CORPORA_DIR}")
    
    run_step1_scanning()
    run_step2_merging()
    run_step3_alignment()
    
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_full_workflow()
