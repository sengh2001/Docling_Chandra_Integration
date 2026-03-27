import os
import sys
import time
from multiprocessing import Pool

# Add parent directory to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.processing.docling_pipeline import docling_ocr_pipeline, create_converter  # noqa: E402
from src.processing.pdf_utils import extract_save_text_layer, remove_watermark  # noqa: E402

NUM_PROCESSES = 2
# CHAPS = list(range(1, 13))
CHAPS = [12]
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs_split")
SCRATCH_DIR = os.path.join(PROJECT_ROOT, "scratch")
# LANGUAGES = {"Tamil": "tam"}
LANGUAGES = {"Hindi": "hin"}
# LANGUAGES = {"English": "eng", "Hindi": "hin", "Tamil": "tam"}

current = 0


def process_chapter(args):
    lang_name, lang_code, chap_num, current, total_files = args
    pid = os.getpid()

    ## load chapter and make output dir
    input_pdf = os.path.join(PROJECT_ROOT, f"books/Class_6-Science-{lang_name}/Chapter_{chap_num}.pdf")

    if not os.path.exists(input_pdf):
        print(f"[{current}/{total_files}] Skipping {input_pdf} (not found)")
        return

    output_dir = os.path.join(OUTPUT_DIR, f"Class_6-Science-{lang_name}/Chapter_{chap_num}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(SCRATCH_DIR, exist_ok=True)

    ## output and temp paths
    original_output_json = f"{output_dir}/parsed_original.json"
    text_output_json = f"{output_dir}/parsed_text.json"
    scratch_pdf = f"{SCRATCH_DIR}/temp_{pid}.pdf"

    print(f"[{current}/{total_files}] Processing {lang_name} Chapter {chap_num}...")

    start = time.time()

    ## save a text layer

    ## run docling and save the two outputs
    converter = create_converter(lang_code)

    ## process the default book without watermark
    remove_watermark(input_pdf, scratch_pdf)
    docling_ocr_pipeline(input_pdf, converter, output_json=original_output_json)

    ## proecss book with only the text
    extract_save_text_layer(input_pdf, scratch_pdf)
    docling_ocr_pipeline(scratch_pdf, converter, output_json=text_output_json)

    elapsed = time.time() - start
    current += 1
    print(f"[{current}/{total_files}] Completed in {elapsed:.2f}s\n")

    os.remove(scratch_pdf)


if __name__ == "__main__":
    chapters = CHAPS

    total_files = len(LANGUAGES) * len(chapters)

    tasks = [
        (lang_name, lang_code, chap_num, i + 1, total_files)
        for i, (lang_name, lang_code) in enumerate(LANGUAGES.items())
        for chap_num in chapters
    ]

    with Pool(processes=NUM_PROCESSES) as pool:
        pool.map(process_chapter, tasks)
