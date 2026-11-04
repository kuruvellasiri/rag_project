
import json
from pathlib import Path

from chunking.fixedsize_chunking import fixed_size_chunking
from chunking.sentence_chunking import sentence_based_chunking
from chunking.recursive_chunking import recursive_chunking
from chunking.token_chunking import token_based_chunking
from chunking.semantic_chunking import semantic_chunking


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
SHARED_DATA_DIR = BASE_DIR.parent / "shared_data"
INPUT_FILE = SHARED_DATA_DIR / "ocr_output" / "extracted_pages.json"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def load_extracted_pages():
    """Load extracted page records from JSON."""

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        pages = json.load(file)

    return pages


def save_chunks(chunks, filename):
    """Save generated chunks into a JSON file."""

    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=4, ensure_ascii=False)

    print(f"Saved: {output_path}")


def run_chunking_pipeline():

    # 1. Load extracted pages
    pages = load_extracted_pages()

    print(f"Total pages loaded: {len(pages)}")

    # 2. Store chunks from all strategies
    fixed_chunks = []
    sentence_chunks = []
    recursive_chunks = []
    token_chunks = []
    semantic_chunks = []

    # Load the semantic model only once
    from sentence_transformers import SentenceTransformer

    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

    # 3. Process every page
    for page in pages:

        text = page["text"]
        source = page["source"]
        page_number = page["page_number"]

        if not text.strip():
            continue

        # Fixed-size chunking
        fixed_chunks.extend(
            fixed_size_chunking(
                text, source, page_number
            )
        )

        # Sentence-based chunking
        sentence_chunks.extend(
            sentence_based_chunking(
                text, source, page_number
            )
        )

        # Recursive chunking
        recursive_chunks.extend(
            recursive_chunking(
                text, source, page_number
            )
        )

        # Token-based chunking
        token_chunks.extend(
            token_based_chunking(
                text, source, page_number
            )
        )

        # Semantic chunking
        semantic_chunks.extend(
            semantic_chunking(
                text,
                source,
                page_number,
                model=semantic_model
            )
        )

        print(f"Processed page {page_number}")

    # 4. Save separate output files
    save_chunks(fixed_chunks, "fixed_chunks.json")
    save_chunks(sentence_chunks, "sentence_chunks.json")
    save_chunks(recursive_chunks, "recursive_chunks.json")
    save_chunks(token_chunks, "token_chunks.json")
    save_chunks(semantic_chunks, "semantic_chunks.json")

    # 5. Print summary
    print("\n========== CHUNKING SUMMARY ==========")

    print(f"Fixed-size chunks: {len(fixed_chunks)}")
    print(f"Sentence-based chunks: {len(sentence_chunks)}")
    print(f"Recursive chunks: {len(recursive_chunks)}")
    print(f"Token-based chunks: {len(token_chunks)}")
    print(f"Semantic chunks: {len(semantic_chunks)}")

    print("\nAll chunking strategies completed!")


if __name__ == "__main__":
    run_chunking_pipeline()
