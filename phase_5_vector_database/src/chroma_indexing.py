
import json
import numpy as np

from pathlib import Path
from src.chroma_store import create_chroma_client, create_collection

PHASE5_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PHASE5_DIR.parent

EMBEDDINGS_FILE = (
    PROJECT_DIR
    / "phase_4_embeddings"
    / "outputs"
    / "handbook_embeddings.npy"
)

METADATA_FILE = (
    PROJECT_DIR
    / "phase_4_embeddings"
    / "outputs"
    / "embedding_metadata.json"
)


def load_embeddings():
    embeddings = np.load(EMBEDDINGS_FILE)

    print("Embeddings loaded:", embeddings.shape)

    return embeddings


def load_metadata():
    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    print("Metadata records loaded:", len(metadata))

    return metadata


def insert_data(collection, embeddings, metadata):
    if len(embeddings) != len(metadata):
        raise ValueError("Embedding and metadata counts do not match")

    ids = [item["chunk_id"] for item in metadata]

    documents = [item["text"] for item in metadata]

    metadatas = [
        {
            "source": item["source"],
            "file_type": item["file_type"],
            "page_number": item["page_number"],
            "chunk_index": item["chunk_index"],
            "chunking_strategy": item["chunking_strategy"]
        }
        for item in metadata
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas
    )

    print("Data inserted successfully!")
    print("Total records:", collection.count())


if __name__ == "__main__":
    client = create_chroma_client()
    collection = create_collection(client)

    embeddings = load_embeddings()
    metadata = load_metadata()

    insert_data(collection, embeddings, metadata)
