
import json
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer


# -----------------------------------------
# 1. Locate project directories
# -----------------------------------------

PHASE5_DIR = Path(__file__).resolve().parents[1]

PROJECT_DIR = PHASE5_DIR.parent

OUTPUT_DIR = PHASE5_DIR / "outputs"

INDEX_FILE = OUTPUT_DIR / "handbook.index"

PHASE4_OUTPUT_DIR = (
    PROJECT_DIR / "phase_4_embeddings" / "outputs"
)

METADATA_FILE = (
    PHASE4_OUTPUT_DIR / "embedding_metadata.json"
)


# -----------------------------------------
# 2. Load embedding model
# -----------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------------
# 3. Load FAISS index
# -----------------------------------------

def load_faiss_index():
    """Load the saved FAISS vector index."""

    index = faiss.read_index(str(INDEX_FILE))

    print("FAISS index loaded successfully")
    print("Total vectors:", index.ntotal)
    print("Vector dimensions:", index.d)

    return index


# -----------------------------------------
# 4. Load document metadata
# -----------------------------------------

def load_metadata():
    """Load chunk information corresponding to embeddings."""

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return metadata


# -----------------------------------------
# 5. Generate query embedding
# -----------------------------------------

def generate_query_embedding(query):
    """Convert user question into a numerical vector."""

    query_vector = model.encode(
        query,
        convert_to_numpy=True
    )

    # FAISS expects float32 vectors
    query_vector = np.asarray(
        query_vector,
        dtype=np.float32
    )

    # Reshape from (384,) to (1, 384)
    query_vector = query_vector.reshape(1, -1)

    # Normalize query vector
    faiss.normalize_L2(query_vector)

    return query_vector


# -----------------------------------------
# 6. Perform FAISS semantic search
# -----------------------------------------

def search_documents(query, top_k=5):
    """Retrieve relevant chunks using FAISS."""

    # Load index and metadata
    index = load_faiss_index()

    metadata = load_metadata()

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index and metadata counts do not match"
        )

    # Generate normalized query embedding
    query_vector = generate_query_embedding(query)

    print("\nQuery embedding shape:", query_vector.shape)

    # Search FAISS index
    scores, indices = index.search(
        query_vector,
        top_k
    )

    results = []

    # Retrieve corresponding metadata
    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        # FAISS returns -1 when no result is available
        if index_position == -1:
            continue

        result = {
            "chunk_id": metadata[index_position]["chunk_id"],
            "page_number": metadata[index_position]["page_number"],
            "text": metadata[index_position]["text"],
            "similarity_score": float(score)
        }

        results.append(result)

    return results


# -----------------------------------------
# 7. Display search results
# -----------------------------------------

def display_results(results):

    print("\n========== FAISS SEARCH RESULTS ==========")

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")

        print("Chunk ID:", result["chunk_id"])

        print("Page:", result["page_number"])

        print(
            "Cosine Similarity:",
            round(result["similarity_score"], 4)
        )

        print("Text:", result["text"][:500])

        print("-" * 60)


# -----------------------------------------
# 8. Main execution
# -----------------------------------------

if __name__ == "__main__":

    user_query = input("\nEnter your question: ")

    results = search_documents(
        user_query,
        top_k=5
    )

    display_results(results)
