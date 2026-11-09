
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.chroma_store import create_chroma_client, create_collection

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_query_embedding(query):
    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    return query_embedding.tolist()


def search_documents(query, top_k=5):
    # Connect to existing ChromaDB
    client = create_chroma_client()

    # Get existing collection
    collection = create_collection(client)

    # Generate query embedding
    query_embedding = generate_query_embedding(query)

    # Search similar vectors
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    return results


def display_results(results):
    print("\n========== CHROMADB SEARCH RESULTS ==========")

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (chunk_id, document, metadata, distance) in enumerate(
        zip(ids, documents, metadatas, distances),
        start=1
    ):
        print(f"\nRank: {rank}")
        print("Chunk ID:", chunk_id)
        print("Page:", metadata["page_number"])
        print("Distance:", round(distance, 4))
        print("Text:", document[:500])
        print("-" * 60)


if __name__ == "__main__":
    user_query = input("\nEnter your question: ")

    results = search_documents(user_query, top_k=5)

    display_results(results)
