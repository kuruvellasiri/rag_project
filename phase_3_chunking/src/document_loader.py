
from pathlib import Path
import re
import json
from pypdf import PdfReader
import pypdfium2 as pdfium
import pytesseract


# Locate the PDF
SCRIPT_DIR = Path(__file__).resolve().parent.parent

SHARED_DATA_DIR = SCRIPT_DIR.parent / "shared_data"

PDF_PATH = SHARED_DATA_DIR / "GU_STUDENT_HANDBOOK.pdf"

OUTPUT_DIR = SHARED_DATA_DIR / "ocr_output"

OUTPUT_DIR.mkdir(exist_ok=True)


def clean_text(text):
    """Clean unnecessary whitespace while preserving line breaks."""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.split("\n")
    ]

    return "\n".join(lines).strip()


def load_pdf(pdf_path):

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pdf_document = pdfium.PdfDocument(str(pdf_path))

    records = []

    try:
        for page_index, page in enumerate(reader.pages):

            page_number = page_index + 1

            # First, try normal PDF text extraction
            text = page.extract_text() or ""

            extraction_method = "pypdf"

            # Use OCR if normal extraction produces insufficient text
            if len(text.strip()) < 30:

                pdf_page = pdf_document[page_index]

                image = pdf_page.render(scale=2).to_pil()

                text = pytesseract.image_to_string(
                    image,
                    lang="eng"
                )

                extraction_method = "ocr"

            # Clean the extracted text
            text = clean_text(text)

            # Store text and metadata
            record = {
                "source": pdf_path.name,
                "page_number": page_number,
                "extraction_method": extraction_method,
                "text": text
            }

            records.append(record)

            print(
                f"Processed page {page_number}: "
                f"{extraction_method}"
            )

    finally:
        pdf_document.close()

    return records

def save_records(records):

    output_path = OUTPUT_DIR / "extracted_pages.json"

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            records,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(f"\nExtracted data saved to: {output_path}")



if __name__ == "__main__":

    print("Starting PDF processing...\n")

    records = load_pdf(PDF_PATH)

    save_records(records)

    print("\nProcessing Summary")

    print("Total pages:", len(records))

    print(
        "Pages processed using pypdf:",
        sum(r["extraction_method"] == "pypdf" for r in records)
    )

    print(
        "Pages processed using OCR:",
        sum(r["extraction_method"] == "ocr" for r in records)
    )

    print(
        "Pages with no extracted text:",
        sum(not r["text"] for r in records)
    )

    print("\nFirst page preview:")
    print(records[0]["text"][:500] if records else "No pages found")

