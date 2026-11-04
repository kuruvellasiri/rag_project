# Phase 3: Chunking

This phase demonstrates five separate chunking strategies over the extracted Student Handbook pages: fixed-size, sentence-based, recursive, token-based, and semantic chunking. Each strategy remains in its own module under `src/chunking/`.

## Inputs and outputs

The original `GU_STUDENT_HANDBOOK.pdf` is stored once in the top-level `shared_data/`. OCR/text extraction records are in `shared_data/ocr_output/extracted_pages.json`. The PDF loader in `src/document_loader.py` can regenerate that file. The chunking pipeline reads the extracted records and writes the five existing JSON outputs to `outputs/`.

The small fictional practice text is retained in `data/student_handbook.txt`. The extracted-page file and generated chunks are preserved as existing data.

## Dependencies

Install this phase independently with `python -m pip install -r requirements.txt`. OCR may also require a system Tesseract installation; no dependencies were installed during restructuring.

## Run

From this directory, use `python .\src\document_loader.py` to extract the handbook, or `python .\src\chunking_pipeline.py` to chunk the preserved extraction. The pipeline imports each strategy from its independent module.
