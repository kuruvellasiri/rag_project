
import time
import numpy as np
import faiss

from pathlib import Path
from sentence_transformers import SentenceTransformer


# -----------------------------------------
# 1. Locate files
# -----------------------------------------

PHASE5_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PHASE5_DIR.parent

PHASE4_OUTPUT_DIR = PROJECT_DIR / "phase_4_embeddings" / "outputs"

EMBEDDINGS_FILE = PHASE4_OUTPUT_DIR / "handbook_embeddings.npy"

INDEX_FILE = PHASE5_DIR / "outputs" / "handbook.index"


# -----------------------------------------
# 2. Load embeddings and FAISS index
# -----------------------------------------

print("Loading document embeddings...")

embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)

# Normalize document vectors
faiss.normalize_L2(embeddings)

print("Embedding shape:", embeddings.shape)

print("\nLoading FAISS index...")

index = faiss.read_index(str(INDEX_FILE))

print("Total FAISS vectors:", index.ntotal)


# -----------------------------------------
# 3. Load embedding model
# -----------------------------------------

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------------------
# 4. Generate query embedding
# -----------------------------------------

query = input("\nEnter your question: ")

query_vector = model.encode(
    query,
    convert_to_numpy=True
).astype(np.float32)

query_vector = query_vector.reshape(1, -1)

faiss.normalize_L2(query_vector)

print("Query embedding shape:", query_vector.shape)


# -----------------------------------------
# 5. NumPy semantic search
# -----------------------------------------

def numpy_search(query_vector, embeddings, top_k=5):

    # Dot product of normalized vectors = cosine similarity
    scores = embeddings @ query_vector[0]

    # Highest similarity first
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    return scores[ranked_indices], ranked_indices


# -----------------------------------------
# 6. FAISS semantic search
# -----------------------------------------

def faiss_search(query_vector, index, top_k=5):

    scores, indices = index.search(
        query_vector,
        top_k
    )

    return scores[0], indices[0]


# -----------------------------------------
# 7. Benchmark function
# -----------------------------------------

def benchmark(search_function, *args, iterations=100):

    # Warm-up execution
    search_function(*args)

    execution_times = []

    for _ in range(iterations):

        start_time = time.perf_counter()

        search_function(*args)

        end_time = time.perf_counter()

        execution_times.append(
            (end_time - start_time) * 1000
        )

    return {
        "average_ms": np.mean(execution_times),
        "median_ms": np.median(execution_times),
        "minimum_ms": np.min(execution_times)
    }


# -----------------------------------------
# 8. Run performance comparison
# -----------------------------------------

if __name__ == "__main__":

    TOP_K = 5
    ITERATIONS = 100

    print("\nRunning NumPy benchmark...")

    numpy_results = benchmark(
        numpy_search,
        query_vector,
        embeddings,
        TOP_K,
        iterations=ITERATIONS
    )

    print("\nRunning FAISS benchmark...")

    faiss_results = benchmark(
        faiss_search,
        query_vector,
        index,
        TOP_K,
        iterations=ITERATIONS
    )

    # Verify that both methods return the same results
    numpy_scores, numpy_indices = numpy_search(
        query_vector,
        embeddings,
        TOP_K
    )

    faiss_scores, faiss_indices = faiss_search(
        query_vector,
        index,
        TOP_K
    )

    print("\n========== RESULT VERIFICATION ==========")

    print("NumPy indices:", numpy_indices)
    print("FAISS indices:", faiss_indices)

    print(
        "Same ranking:",
        np.array_equal(numpy_indices, faiss_indices)
    )

    print(
        "Scores match:",
        np.allclose(numpy_scores, faiss_scores, atol=1e-5)
    )

    # Display benchmark results
    print("\n========== PERFORMANCE COMPARISON ==========")

    print("\nNumPy:")
    print(f"Average: {numpy_results['average_ms']:.4f} ms")
    print(f"Median:  {numpy_results['median_ms']:.4f} ms")
    print(f"Minimum: {numpy_results['minimum_ms']:.4f} ms")

    print("\nFAISS:")
    print(f"Average: {faiss_results['average_ms']:.4f} ms")
    print(f"Median:  {faiss_results['median_ms']:.4f} ms")
    print(f"Minimum: {faiss_results['minimum_ms']:.4f} ms")

    print("\nBenchmark completed!")
