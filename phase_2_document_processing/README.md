# Phase 2: Document Processing

This phase loads TXT, PDF, DOCX, CSV, XLSX, HTML, and JSON documents into consistent text records. It demonstrates document extraction and cleanup; it does not perform chunking or retrieval. The seven handbook formats in `data/` are small fictional examples.

## Setup and run

From this directory, install this phase?s dependencies and run the loader:

```powershell
python -m pip install -r requirements.txt
python .\src\main.py
```

The implementation is in `src/main.py`; sample files are in `data/`; tests are in `tests/`.

## Test

```powershell
python -m pytest -q
```

Each record contains `document_id`, `source`, `file_type`, `page`, and `text`. PDF records have a 1-based page number; other formats use `null`. Excel workbooks return one record per worksheet.
