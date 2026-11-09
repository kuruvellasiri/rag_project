"""Phase 4 entry point; embedding lessons will be added later."""
import json
from pathlib import Path


# Locate the project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent

CHUNK_FILE = (
    PROJECT_DIR
    / "phase_3_chunking"
    / "outputs"
    / "recursive_chunks.json"
)


def load_chunks():
    """Load recursive chunks from Phase 3."""

    with open(CHUNK_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks


if __name__ == "__main__":

    chunks = load_chunks()

    print("Total chunks:", len(chunks))

    print("\nFirst chunk:")
    print(chunks[0])

    print("\nChunk text:")
    print(chunks[0]["text"])

    print("\nPage number:")
    print(chunks[0]["page_number"])