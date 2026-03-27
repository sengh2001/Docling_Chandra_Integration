import os
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from docling_core.types.doc.document import DoclingDocument
from src.matching import BI_ENCODER_MODEL

def extract_text_blocks(doc: DoclingDocument):
    blocks = {}
    for text_item in doc.texts:
        if text_item.text and text_item.text.strip():
            blocks[text_item.self_ref] = text_item.text
    return blocks

def main():
    print("Loading models and data for evaluation...")
    output_dir = PROJECT_ROOT / "outputs_imd_test"
    
    eng_json = output_dir / "merged_english.json"
    hin_json = output_dir / "merged_hindi.json"
    eng_to_hin_path = output_dir / "eng_to_hin.json"
    
    if not eng_json.exists() or not hin_json.exists() or not eng_to_hin_path.exists():
        print("Required output files not found. Did you run run_imd_workflows.py first?")
        sys.exit(1)
        
    doc_eng = DoclingDocument.load_from_json(str(eng_json))
    doc_hin = DoclingDocument.load_from_json(str(hin_json))
    
    eng_blocks = extract_text_blocks(doc_eng)
    hin_blocks = extract_text_blocks(doc_hin)
    
    with open(eng_to_hin_path, 'r', encoding='utf-8') as f:
        eng_to_hin = json.load(f)
        
    print(f"Loaded {len(eng_to_hin)} alignment pairs.")
    
    # Load bi-encoder natively to compute actual similarity scores
    model = SentenceTransformer(BI_ENCODER_MODEL)
    
    results = []
    scores = []
    
    print("Computing similarity scores for aligned pairs...")
    for eng_id, hin_id in eng_to_hin.items():
        if eng_id in eng_blocks and hin_id in hin_blocks:
            eng_text = eng_blocks[eng_id]
            hin_text = hin_blocks[hin_id]
            
            # Compute similarity
            emb_eng = model.encode(eng_text, convert_to_tensor=True)
            emb_hin = model.encode(hin_text, convert_to_tensor=True)
            sim = util.cos_sim(emb_eng, emb_hin).item()
            
            scores.append(sim)
            results.append({
                "eng_id": eng_id,
                "hin_id": hin_id,
                "eng_text": eng_text,
                "hin_text": hin_text,
                "score": sim
            })

    # Sort results by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # --- GRAPHICAL PLOT ---
    print("Generating graphical plot...")
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(scores, bins=10, range=(0.0, 1.0), color='skyblue', edgecolor='black')
    
    # Color-code the histogram (Green for >0.85, Yellow 0.7-0.85, Red <0.7)
    for i in range(len(patches)):
        if bins[i] >= 0.85:
            patches[i].set_facecolor('lightgreen')
        elif bins[i] >= 0.70:
            patches[i].set_facecolor('khaki')
        else:
            patches[i].set_facecolor('salmon')
            
    plt.axvline(x=sum(scores)/len(scores), color='r', linestyle='dashed', linewidth=2, label=f'Mean: {sum(scores)/len(scores):.2f}')
    plt.title('Distribution of Aligned Text Similarity Scores', fontsize=16)
    plt.xlabel('Cosine Similarity Score', fontsize=12)
    plt.ylabel('Number of Aligned Pairs (Blocks)', fontsize=12)
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    
    plot_path = output_dir / 'alignment_score_distribution.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # --- HTML REPORT ---
    print("Generating HTML Visual Report...")
    html_out = output_dir / 'alignment_report.html'
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Alignment Success Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; color: #333; margin: 0; padding: 20px; }}
            h1 {{ text-align: center; color: #2c3e50; }}
            .summary {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 30px; text-align: center;}}
            .card {{ background: white; margin-bottom: 20px; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 8px solid; }}
            .excellent {{ border-color: #2ecc71; }}
            .good {{ border-color: #f1c40f; }}
            .poor {{ border-color: #e74c3c; }}
            .score {{ font-weight: bold; font-size: 1.2em; margin-bottom: 15px; }}
            .grid {{ display: flex; gap: 20px; }}
            .col {{ flex: 1; padding: 15px; background: #fdfdfd; border: 1px solid #eee; border-radius: 4px;}}
            .lang {{ font-size: 0.8em; text-transform: uppercase; color: #7f8c8d; margin-bottom: 5px; font-weight: bold;}}
        </style>
    </head>
    <body>
        <h1>Docling OCR: Cross-Lingual Alignment Report</h1>
        <div class="summary">
            <h2>Average Match Confidence: {sum(scores)/len(scores):.2%}</h2>
            <img src="alignment_score_distribution.png" width="600" style="margin-top:20px; border: 1px solid #ddd; border-radius: 8px;"/>
        </div>
    """
    
    for r in results:
        status = "excellent" if r["score"] >= 0.85 else ("good" if r["score"] >= 0.70 else "poor")
        html_content += f"""
        <div class="card {status}">
            <div class="score">Similarity Score: {r['score']:.3f}</div>
            <div class="grid">
                <div class="col">
                    <div class="lang">English [ID: {r['eng_id']}]</div>
                    <div>{r['eng_text']}</div>
                </div>
                <div class="col">
                    <div class="lang">Hindi [ID: {r['hin_id']}]</div>
                    <div>{r['hin_text']}</div>
                </div>
            </div>
        </div>
        """
        
    html_content += """
    </body>
    </html>
    """
    
    with open(html_out, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    print(f"Done! Created:\n- {plot_path}\n- {html_out}")

if __name__ == "__main__":
    main()
