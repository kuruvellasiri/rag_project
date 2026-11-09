
import json
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer


# Project directories
PROJECT_DIR = Path(__file__).resolve().parents[2]

CHUNK_FILE = (
    PROJECT_DIR
    / "phase_3_chunking"
    / "outputs"
    / "recursive_chunks.json"
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load pretrained embedding model
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_chunks():
    """Load chunks generated in Phase 3."""

    with open(CHUNK_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks


def generate_embeddings(chunks):
    """Convert chunk text into numerical vectors."""

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings


def save_embeddings(chunks, embeddings):
    """Save vectors and their corresponding metadata."""

    np.save(
        OUTPUT_DIR / "handbook_embeddings.npy",
        embeddings
    )

    with open(
        OUTPUT_DIR / "embedding_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )


if __name__ == "__main__":

    chunks = load_chunks()

    print("Total chunks:", len(chunks))

    embeddings = generate_embeddings(chunks)

    print("Embedding shape:", embeddings.shape)

    save_embeddings(chunks, embeddings)

    print("Embeddings generated successfully!")
