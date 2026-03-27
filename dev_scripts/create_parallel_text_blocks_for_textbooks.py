#!/usr/bin/env python3
"""
Create Parallel Text Blocks for Textbooks
==========================================

This script processes each chapter in the parallel corpora and creates JSONL files
containing aligned text blocks across English, Hindi, and Tamil languages.

For each text block in the English document, it retrieves the corresponding
Hindi and Tamil matches using the ParallelAlignedDocument class and the
format_match_as_text function.

Output Format (JSONL):
    Each line is a JSON object with:
    - id: The cref (coordinate reference) of the English text block
    - english: Formatted text from format_match_as_text for English
    - hindi: Formatted text from format_match_as_text for Hindi
    - tamil: Formatted text from format_match_as_text for Tamil

Usage:
    python dev_scripts/create_parallel_text_blocks_for_textbooks.py
"""

import sys
import json
from pathlib import Path

# Add the project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from docling_core.types.doc import DoclingDocument  # noqa: E402
from src.alignment import format_match_as_text, _initialize_parallel_doc  # noqa: E402


# -----------------------------
# Configuration
# -----------------------------

# Which chapters to process (1-12 for full NCERT Class 6 Science)
CHAPTERS = list(range(1, 13))

# Paths (relative to docling_ocr_extraction/)
PARALLEL_CORPORA_DIR = PROJECT_ROOT / "parallel_corpora" / "Class_6-Science"


def get_all_text_block_crefs(doc: DoclingDocument) -> list:
    """
    Extract all text block crefs from a DoclingDocument.
    
    Args:
        doc: The DoclingDocument to extract crefs from.
        
    Returns:
        List of cref strings for all text blocks in the document.
    """
    crefs = []
    for item, _ in doc.iterate_items():
        # Only include items that have text content
        if hasattr(item, "text") and item.text:
            crefs.append(item.self_ref)
    return crefs


def process_chapter(chapter_num: int, parallel_corpora_path: str) -> list:
    """
    Process a single chapter and return all parallel text blocks.
    
    Args:
        chapter_num: The chapter number to process.
        parallel_corpora_path: Base path to parallel corpora directory.
        
    Returns:
        List of dictionaries containing parallel text blocks.
    """
    print(f"Processing Chapter {chapter_num}...")
    
    # Initialize the ParallelAlignedDocument for this chapter
    parallel_doc = _initialize_parallel_doc(chapter_num, parallel_corpora_path)
    
    # Get the English document to iterate through its text blocks
    english_doc = parallel_doc.docs["eng"]
    
    # Get all crefs from the English document
    english_crefs = get_all_text_block_crefs(english_doc)
    print(f"  Found {len(english_crefs)} text blocks in English document")
    
    results = []
    
    for cref in english_crefs:
        try:
            # Retrieve parallel data for this English text block
            parallel_data = parallel_doc.retrieve_parallel_data(cref, "eng")
            
            # Build the match_data structure expected by format_match_as_text
            match_data = {"parallel_data": parallel_data}
            
            # Format text for each language
            english_text = format_match_as_text(match_data, "eng")
            hindi_text = format_match_as_text(match_data, "hi")
            tamil_text = format_match_as_text(match_data, "ta")
            
            # Create the output record
            record = {
                "id": cref,
                "english": english_text,
                "hindi": hindi_text,
                "tamil": tamil_text,
            }
            results.append(record)
            
        except Exception as e:
            # Log the error but continue processing other text blocks
            print(f"  Warning: Failed to process cref '{cref}': {e}")
            continue
    
    print(f"  Successfully processed {len(results)} text blocks")
    return results


def save_results_as_jsonl(results: list, output_path: str) -> None:
    """
    Save the results as a JSONL file.
    
    Args:
        results: List of dictionaries to save.
        output_path: Path to the output JSONL file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for record in results:
            # Write each record as a JSON line
            json_line = json.dumps(record, ensure_ascii=False)
            f.write(json_line + "\n")
    
    print(f"  Saved {len(results)} records to {output_path}")


def main():
    """
    Main function to process all chapters and create parallel text block files.
    """
    print("=" * 60)
    print("PARALLEL TEXT BLOCKS CREATION")
    print("=" * 60)
    print(f"Chapters: {CHAPTERS}")
    print(f"Output: {PARALLEL_CORPORA_DIR}")
    print("=" * 60)
    
    # Process each chapter from the configuration
    for chapter_num in CHAPTERS:
        # Process the chapter
        results = process_chapter(chapter_num, str(PARALLEL_CORPORA_DIR))
        
        # Define output path
        output_path = PARALLEL_CORPORA_DIR / f"Chapter_{chapter_num}" / "parallel_text_blocks.jsonl"
        
        # Save results
        if results:
            save_results_as_jsonl(results, str(output_path))
        else:
            print(f"  No results to save for Chapter {chapter_num}")
        
        print()  # Blank line between chapters
    
    print("=" * 60)
    print("Processing complete!")


if __name__ == "__main__":
    main()
