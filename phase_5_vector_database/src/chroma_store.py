
import chromadb
from pathlib import Path

# Phase 5 project directory
PHASE5_DIR = Path(__file__).resolve().parents[1]

# Persistent database directory
CHROMA_DB_DIR = PHASE5_DIR / "outputs" / "chroma_db"


def create_chroma_client():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DB_DIR)
    )

    print("ChromaDB client created successfully")

    return client


def create_collection(client):
    collection = client.get_or_create_collection(
        name="student_handbook",
        metadata={"description": "GU Student Handbook chunks"}
    )

    print("Collection created successfully")
    print("Collection name:", collection.name)

    return collection


if __name__ == "__main__":
    client = create_chroma_client()
    collection = create_collection(client)

    print("Total stored chunks:", collection.count())
