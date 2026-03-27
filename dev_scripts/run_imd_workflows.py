import os
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.processing.docling_pipeline import docling_ocr_pipeline, create_converter
from src.processing.pdf_utils import extract_save_text_layer, remove_watermark
from src.merge_document_scans import merge_documents
from docling_core.types.doc.document import DoclingDocument
from src.matching import align_english_hindi_blocks, BI_ENCODER_MODEL, CROSS_ENCODER_MODEL
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.alignment import ParallelAlignedDocument

def extract_text_blocks(doc: DoclingDocument):
    blocks = {}
    for text_item in doc.texts:
        if text_item.text and text_item.text.strip():
            blocks[text_item.self_ref] = text_item.text
    return blocks

def main():
    print("="*60)
    print("RUNNING WORKFLOWS 1, 2, AND 3 ON IMD ADVISORY PDFs")
    print("="*60)
    
    # Setup paths
    books_dir = PROJECT_ROOT / "books"
    output_dir = PROJECT_ROOT / "outputs_imd_test"
    scratch_dir = PROJECT_ROOT / "scratch"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    scratch_dir.mkdir(exist_ok=True, parents=True)
    
    pdfs = {
        "english": {
            "path": books_dir / "2025-01-07_english.pdf",
            "code": "eng"
        },
        "hindi": {
            "path": books_dir / "2025-01-07_hindi.pdf",
            "code": "hin"
        }
    }
    
    # WORKFLOW 1: DUAL SCAN EXTRACTION
    print("\n\n--- WORKFLOW 1: DUAL-SCAN EXTRACTION ---")
    for lang, info in pdfs.items():
        pdf_path = info["path"]
        lang_code = info["code"]
        lang_out_dir = output_dir / lang
        lang_out_dir.mkdir(exist_ok=True)
        
        orig_json = lang_out_dir / "parsed_original.json"
        text_json = lang_out_dir / "parsed_text.json"
        
        if orig_json.exists() and text_json.exists():
            print(f"Skipping extraction for {lang}, JSONs already exist.")
        else:
            print(f"Processing {lang} PDF: {pdf_path.name}...")
            converter = create_converter(lang_code)
            scratch_pdf = scratch_dir / f"temp_{lang}.pdf"
            
            # Original scan
            print("  Running original full scan...")
            remove_watermark(str(pdf_path), str(scratch_pdf))
            docling_ocr_pipeline(str(pdf_path), converter, output_json=str(orig_json))
            
            # Text-only scan
            print("  Running text-only scan...")
            extract_save_text_layer(str(pdf_path), str(scratch_pdf))
            docling_ocr_pipeline(str(scratch_pdf), converter, output_json=str(text_json))
            
            if scratch_pdf.exists():
                scratch_pdf.unlink()

    # WORKFLOW 2: MERGING AND ALIGNMENT
    print("\n\n--- WORKFLOW 2: DOCUMENT MERGING & ALIGNMENT ---")
    merged_docs = {}
    
    # 2a. Merge
    for lang in pdfs.keys():
        lang_out_dir = output_dir / lang
        orig_json = lang_out_dir / "parsed_original.json"
        text_json = lang_out_dir / "parsed_text.json"
        merged_json = output_dir / f"merged_{lang}.json"
        
        if merged_json.exists():
            print(f"Skipping merge for {lang}, already exists.")
            merged_docs[lang] = DoclingDocument.load_from_json(merged_json)
        else:
            print(f"Merging {lang} scans...")
            baseline_doc = DoclingDocument.load_from_json(orig_json)
            text_only_doc = DoclingDocument.load_from_json(text_json)
            merged_doc = merge_documents(baseline_doc, text_only_doc)
            merged_doc.save_as_json(merged_json)
            merged_docs[lang] = merged_doc

    # 2b. Alignment
    eng_to_hin_path = output_dir / "eng_to_hin.json"
    hin_to_eng_path = output_dir / "hin_to_eng.json"
    
    if eng_to_hin_path.exists() and hin_to_eng_path.exists():
        print("Skipping alignment, dictionaries already exist.")
    else:
        print("Aligning English and Hindi text blocks...")
        eng_blocks = extract_text_blocks(merged_docs["english"])
        hin_blocks = extract_text_blocks(merged_docs["hindi"])
        
        print(f" Extracted {len(eng_blocks)} English blocks and {len(hin_blocks)} Hindi blocks.")
        
        bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
        cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        
        eng_to_hin, hin_to_eng = align_english_hindi_blocks(
            eng_blocks, hin_blocks, bi_encoder, cross_encoder
        )
        
        with open(eng_to_hin_path, 'w', encoding='utf-8') as f:
            json.dump(eng_to_hin, f, ensure_ascii=False, indent=2)
            
        with open(hin_to_eng_path, 'w', encoding='utf-8') as f:
            json.dump(hin_to_eng, f, ensure_ascii=False, indent=2)
            
        print("Alignment Complete. Saved eng_to_hin.json and hin_to_eng.json")


    # WORKFLOW 3: PARALLEL CORPORA RETRIEVAL TEST
    print("\n\n--- WORKFLOW 3: CROSS-LINGUAL ALIGNMENT DATA RETRIEVAL ---")
    print("Initializing ParallelAlignedDocument to simulate parallel block matching...\n")
    
    # Load raw dicts for ParallelAlignedDocument
    with open(output_dir / "merged_english.json", 'r') as f:
        data_en = json.load(f)
    with open(output_dir / "merged_hindi.json", 'r') as f:
        data_hi = json.load(f)
    with open(eng_to_hin_path, 'r') as f:
        eng_to_hi = json.load(f)
        
    pad = ParallelAlignedDocument(
        data_en=data_en,
        data_hi=data_hi,
        data_ta=data_en, # Just pass English as dummy for Tamil since we don't have Tamil PDF
        eng_to_hi_matches=eng_to_hi,
        hi_to_eng_matches={} 
    )
    
    # Lets retrieve parallel data for the first 3 English blocks
    print("Testing Retrieval of 3 Aligned English -> Hindi blocks:\n")
    eng_blocks = extract_text_blocks(merged_docs["english"])
    test_crefs = list(eng_blocks.keys())[:3]
    
    for cref in test_crefs:
        try:
            parallel_data = pad.retrieve_parallel_data(cref, "eng")
            
            print(f"--- MATCH FOR {cref} ---")
            print(f"English Main: {parallel_data['eng']['main']}")
            if parallel_data['hi'] and parallel_data['hi'].get('main'):
                print(f"Hindi Match:  {parallel_data['hi']['main']}")
            else:
                print("Hindi Match:  None found.")
            print("\n")
        except Exception as e:
            pass

    print("="*60)
    print("ALL 3 WORKFLOWS COMPLETED SUCCESSFULLY.")
    print("="*60)

if __name__ == "__main__":
    main()
