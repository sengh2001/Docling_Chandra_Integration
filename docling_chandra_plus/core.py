import os
from pathlib import Path
from typing import Optional
from .adapter import ChandraAdapter
from .visualizer import ChandraVisualizer

class ChandraPipeline:
    """
    Unified orchestrator for processing a single PDF with its Chandra OCR 2 JSON.
    Automates conversion and visualization.
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter: Optional[ChandraAdapter] = None
        self.visualizer = ChandraVisualizer()

    def process_doc(
        self, 
        pdf_path: str, 
        chandra_json_path: str, 
        doc_name: Optional[str] = None,
        annotate: bool = True
    ) -> str:
        """
        Convert a PDF using Chandra OCR 2 result and create output artifacts.
        
        Returns:
            The path to the generated DoclingDocument JSON.
        """
        doc_name = doc_name or Path(pdf_path).stem
        
        # 1. Initialize Adapter
        self.adapter = ChandraAdapter(pdf_path=pdf_path)
        
        # 2. Convert to DoclingDocument
        doc = self.adapter.convert(chandra_json_path, document_name=doc_name)
        
        # 3. Save Structured JSONs
        json_out = self.output_dir / f"{doc_name}_docling.json"
        
        # Save both ASCII-safe and Readable JSONs
        self.adapter.save(str(json_out), ensure_ascii=True)
        self.adapter.save(str(json_out), ensure_ascii=False)
        
        # 4. Optional: Generate Annotated PDF
        if annotate:
            pdf_out = self.output_dir / f"{doc_name}_annotated.pdf"
            self.visualizer.annotate_document(doc, pdf_path, str(pdf_out))
            print(f"[OK] Annotated PDF created: {pdf_out}")

        self.adapter.close()
        print(f"[SUCCESS] Processed {doc_name} -> {json_out}")
        return str(json_out)

if __name__ == "__main__":
    # Example minimal CLI for the orchestrator
    import argparse
    parser = argparse.ArgumentParser(description="Unified Chandra-Docling Pipeline")
    parser.add_argument("pdf", help="Input PDF path")
    parser.add_argument("json", help="Input Chandra JSON path")
    parser.add_argument("--out", default="outputs_batch", help="Output directory")
    
    args = parser.parse_args()
    
    pipeline = ChandraPipeline(output_dir=args.out)
    pipeline.process_doc(args.pdf, args.json)
