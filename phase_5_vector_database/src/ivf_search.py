
import json
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer

PHASE5_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PHASE5_DIR.parent

INDEX_FILE = PHASE5_DIR / "outputs" / "handbook_ivf.index"

PHASE4_OUTPUT_DIR = PROJECT_DIR / "phase_4_embeddings" / "outputs"
METADATA_FILE = PHASE4_OUTPUT_DIR / "embedding_metadata.json"

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_ivf_index():
    index = faiss.read_index(str(INDEX_FILE))

    print("IVF index loaded successfully")
    print("Total vectors:", index.ntotal)
    print("Number of clusters:", index.nlist)

    return index


def load_metadata():
    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def generate_query_embedding(query):
    query_vector = model.encode(
        query,
        convert_to_numpy=True
    )

    query_vector = np.asarray(query_vector, dtype=np.float32)
    query_vector = query_vector.reshape(1, -1)

    faiss.normalize_L2(query_vector)

    return query_vector


def search_documents(query, top_k=5, nprobe=2):
    index = load_ivf_index()
    metadata = load_metadata()

    if index.ntotal != len(metadata):
        raise ValueError("Index and metadata counts do not match")

    # Set the number of clusters to search
    index.nprobe = nprobe

    query_vector = generate_query_embedding(query)

    scores, indices = index.search(query_vector, top_k)

    results = []

    for score, index_position in zip(scores[0], indices[0]):
        if index_position == -1:
            continue

        results.append({
            "chunk_id": metadata[index_position]["chunk_id"],
            "page_number": metadata[index_position]["page_number"],
            "text": metadata[index_position]["text"],
            "similarity_score": float(score)
        })

    return results


def display_results(results):
    print("\n========== IVF SEARCH RESULTS ==========")

    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print("Chunk ID:", result["chunk_id"])
        print("Page:", result["page_number"])
        print("Similarity:", round(result["similarity_score"], 4))
        print("Text:", result["text"][:500])
        print("-" * 60)


if __name__ == "__main__":
    user_query = input("\nEnter your question: ")

    results = search_documents(
        user_query,
        top_k=5,
        nprobe=2
    )

    display_results(results)
