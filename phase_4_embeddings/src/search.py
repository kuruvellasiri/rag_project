
import json
import numpy as np

from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from src.similarity import cosine_similarity, euclidean_distance


# -----------------------------------------
# 1. Locate Phase 4 directories
# -----------------------------------------

PHASE4_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PHASE4_DIR / "outputs"

EMBEDDINGS_FILE = OUTPUT_DIR / "handbook_embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "embedding_metadata.json"


# -----------------------------------------
# 2. Load the embedding model
# -----------------------------------------

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------------
# 3. Load stored document embeddings
# -----------------------------------------

def load_stored_data():
    """Load document embeddings and their metadata."""

    embeddings = np.load(EMBEDDINGS_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if len(embeddings) != len(metadata):
        raise ValueError("Embedding and metadata counts do not match")

    return embeddings, metadata


# -----------------------------------------
# 4. Generate query embedding
# -----------------------------------------

def generate_query_embedding(query):
    """Convert user query into a numerical vector."""

    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    return query_embedding


# -----------------------------------------
# 5. Normalize document and query vectors
# -----------------------------------------

def prepare_normalized_vectors(query):
    """Normalize document embeddings and query embedding."""

    embeddings, metadata = load_stored_data()

    # Normalize all document vectors
    embeddings = normalize(embeddings, norm="l2")

    # Generate query vector
    query_vector = generate_query_embedding(query)

    # Normalize query vector
    query_vector = normalize(
        query_vector.reshape(1, -1),
        norm="l2"
    )[0]

    # Verify normalization
    print("\nNormalization Verification")
    print("Document vector norm:", np.linalg.norm(embeddings[0]))
    print("Query vector norm:", np.linalg.norm(query_vector))

    return embeddings, query_vector, metadata


# -----------------------------------------
# 6. Cosine Similarity Search
# -----------------------------------------

def search_documents(query, top_k=5):
    """Retrieve documents using cosine similarity."""

    embeddings, query_vector, metadata = prepare_normalized_vectors(query)

    scores = []

    # Compare query against every document vector
    for document_vector in embeddings:

        score = cosine_similarity(
            query_vector,
            document_vector
        )

        scores.append(score)

    # Highest similarity comes first
    ranked_indices = np.argsort(scores)[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        result = {
            "chunk_id": metadata[index]["chunk_id"],
            "page_number": metadata[index]["page_number"],
            "text": metadata[index]["text"],
            "similarity_score": float(scores[index])
        }

        results.append(result)

    return results


# -----------------------------------------
# 7. Euclidean Distance Search
# -----------------------------------------

def search_documents_euclidean(query, top_k=5):
    """Retrieve documents using Euclidean distance."""

    embeddings, query_vector, metadata = prepare_normalized_vectors(query)

    distances = []

    # Calculate distance between query and each document
    for document_vector in embeddings:

        distance = euclidean_distance(
            query_vector,
            document_vector
        )

        distances.append(distance)

    # Lowest distance comes first
    ranked_indices = np.argsort(distances)

    results = []

    for index in ranked_indices[:top_k]:

        result = {
            "chunk_id": metadata[index]["chunk_id"],
            "page_number": metadata[index]["page_number"],
            "text": metadata[index]["text"],
            "euclidean_distance": float(distances[index])
        }

        results.append(result)

    return results


# -----------------------------------------
# 8. Display search results
# -----------------------------------------

def display_results(results, metric_name, score_key):

    print(f"\n========== {metric_name} RESULTS ==========")

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print("Chunk ID:", result["chunk_id"])
        print("Page:", result["page_number"])

        print(
            f"{metric_name}:",
            round(result[score_key], 4)
        )

        print("Text:", result["text"][:300])
        print("-" * 60)


# -----------------------------------------
# 9. Main execution
# -----------------------------------------

if __name__ == "__main__":

    user_query = input("\nEnter your question: ")

    # Cosine Similarity Search
    cosine_results = search_documents(
        user_query,
        top_k=5
    )

    display_results(
        cosine_results,
        "COSINE SIMILARITY",
        "similarity_score"
    )

    # Euclidean Distance Search
    euclidean_results = search_documents_euclidean(
        user_query,
        top_k=5
    )

    display_results(
        euclidean_results,
        "EUCLIDEAN DISTANCE",
        "euclidean_distance"
    )
