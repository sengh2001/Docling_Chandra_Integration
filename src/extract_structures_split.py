import time
import os
from multiprocessing import Pool


from processing.docling_pipeline import docling_ocr_pipeline, create_converter
from processing.pdf_utils import extract_save_text_layer, remove_watermark

NUM_PROCESSES = 2
# CHAPS = list(range(9, 13))
CHAPS = [11, 12]
OUTPUT_DIR = "outputs_split"
SCRATCH_DIR = "scratch"
LANGUAGES = {"Tamil": "tam"}
# LANGUAGES = {"English": "eng", "Hindi": "hin", "Tamil": "tam"}

current = 0


def process_chapter(args):
    """
    Runs the docling OCR on the base pdf + no image pdf, stores the outputs
    to be merged in the future
    """
    lang_name, lang_code, chap_num, current, total_files = args
    pid = os.getpid()

    ## load chapter and make output dir
    input_pdf = f"./books/Class_6-Science-{lang_name}/Chapter_{chap_num}.pdf"

    if not os.path.exists(input_pdf):
        print(f"[{current}/{total_files}] Skipping {input_pdf} (not found)")
        return

    output_dir = f"./{OUTPUT_DIR}/Class_6-Science-{lang_name}/Chapter_{chap_num}"
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
