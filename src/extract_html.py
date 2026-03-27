"""
Driver code to run the docling extration pipeline on all the science textbooks, multiprocessed
"""

import time
import os
from multiprocessing import Pool

from src.processing.docling_pipeline import docling_ocr_pipeline
from src.bbox_draw import draw_bboxes_on_pdf

if __name__ == "__main__":
    # languages = {"Hindi": "hin", "Tamil": "tam"}
    languages = {"English": "eng", "Hindi": "hin", "Tamil": "tam"}
    chapters = range(1, 13)  # adjust range as needed

    total_files = len(languages) * len(chapters)
    current = 0

    def process_chapter(args):
        lang_name, lang_code, chap_num, current, total_files = args
        input_pdf = f"./books/Class_6-Science-{lang_name}/Chapter_{chap_num}.pdf"

        if not os.path.exists(input_pdf):
            print(f"[{current}/{total_files}] Skipping {input_pdf} (not found)")
            return

        output_dir = f"./outputs/Class_6-Science-{lang_name}/Chapter_{chap_num}"
        os.makedirs(output_dir, exist_ok=True)

        output_html = f"{output_dir}/converted_html.html"
        output_json = f"{output_dir}/parsed.json"
        output_pdf = f"{output_dir}/annoted.pdf"

        print(f"[{current}/{total_files}] Processing {lang_name} Chapter {chap_num}...")
        start = time.time()

        doc = docling_ocr_pipeline(
            input_pdf, None, output_html, output_json, lang=lang_code
        )
        draw_bboxes_on_pdf(doc, input_pdf, output_pdf)

        elapsed = time.time() - start
        print(f"[{current}/{total_files}] Completed in {elapsed:.2f}s\n")

    tasks = [
        (lang_name, lang_code, chap_num, i + 1, total_files)
        for i, (lang_name, lang_code) in enumerate(languages.items())
        for chap_num in chapters
    ]

    with Pool(processes=3) as pool:
        pool.map(process_chapter, tasks)
