from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from docx import Document as WordDocument
from openpyxl import load_workbook
from pypdf import PdfReader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DOCUMENTS_DIR = PROJECT_DIR / "data"
SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".csv", ".xlsx", ".html", ".htm", ".json"}


def clean_text(text: str) -> str:
    """Trim line noise while keeping paragraph and row boundaries readable."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned: list[str] = []
    for line in lines:
        if line or (cleaned and cleaned[-1]):  # Keep at most one blank line.
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def _record(path: Path, text: str, page: int | None = None) -> dict[str, Any]:
    return {"document_id": str(path.resolve()), "source": str(path),
            "file_type": path.suffix.lower().lstrip("."), "page": page,
            "text": clean_text(text)}


def load_txt(path: Path) -> list[dict[str, Any]]:
    return [_record(path, path.read_text(encoding="utf-8-sig"))]


def load_pdf(path: Path) -> list[dict[str, Any]]:
    return [_record(path, page.extract_text() or "", number)
            for number, page in enumerate(PdfReader(str(path)).pages, start=1)]


def load_docx(path: Path) -> list[dict[str, Any]]:
    doc = WordDocument(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    # Keep table rows together to retain their relationships.
    parts.extend(" | ".join(cell.text.strip() for cell in row.cells)
                 for table in doc.tables for row in table.rows)
    return [_record(path, "\n".join(parts))]


def load_csv(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return [_record(path, frame.to_csv(index=False, lineterminator="\n"))]


def load_excel(path: Path) -> list[dict[str, Any]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    records = []
    try:
        for sheet in workbook.worksheets:
            rows = [" | ".join("" if value is None else str(value) for value in row)
                    for row in sheet.iter_rows(values_only=True)]
            records.append(_record(path, f"Sheet: {sheet.title}\n" + "\n".join(rows)))
    finally:
        workbook.close()
    return records


def load_html(path: Path) -> list[dict[str, Any]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8-sig"), "html.parser")
    for node in soup(["script", "style", "noscript"]):
        node.decompose()
    return [_record(path, soup.get_text("\n", strip=True))]


def load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return [_record(path, json.dumps(data, ensure_ascii=False, indent=2))]


LOADERS = {".txt": load_txt, ".pdf": load_pdf, ".docx": load_docx,
           ".csv": load_csv, ".xlsx": load_excel,
           ".html": load_html, ".htm": load_html, ".json": load_json}


def load_document(path: str | Path) -> list[dict[str, Any]]:
    """Load one supported document; errors are left for the caller to report."""
    file_path = Path(path)
    try:
        loader = LOADERS[file_path.suffix.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported file type: {file_path.suffix or '(no extension)'}") from exc
    return loader(file_path)


def process_documents(directory: Path = DOCUMENTS_DIR) -> tuple[list[dict[str, Any]], list[tuple[Path, str]]]:
    """Load files independently so one broken file cannot stop the batch."""
    records: list[dict[str, Any]] = []
    failures: list[tuple[Path, str]] = []
    if not directory.exists():
        return records, [(directory, "Documents directory does not exist")]
    paths = sorted(path for path in directory.rglob("*")
                   if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
    for path in paths:
        try:
            records.extend(load_document(path))
        except Exception as exc:  # Report this file, then continue with the next one.
            failures.append((path, f"{type(exc).__name__}: {exc}"))
    return records, failures


def main() -> int:
    print(f"Reading documents from: {DOCUMENTS_DIR}\n")
    records, failures = process_documents()
    for record in records:
        text = record["text"]
        preview = text.replace("\n", " ")[:160]
        print(f"[{record['file_type'].upper()}] {record['source']} (page={record['page']})")
        print(f"  {preview}{'...' if len(text) > 160 else ''}\n")
    print("Summary")
    print(f"  Successful records: {len(records)}")
    print(f"  Failed files: {len(failures)}")
    for path, error in failures:
        print(f"  - {path}: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
