
import numpy as np
import faiss

from pathlib import Path


# -----------------------------------------
# 1. Locate directories
# -----------------------------------------

PHASE5_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PHASE5_DIR.parent

PHASE4_OUTPUT_DIR = PROJECT_DIR / "phase_4_embeddings" / "outputs"

EMBEDDINGS_FILE = PHASE4_OUTPUT_DIR / "handbook_embeddings.npy"

OUTPUT_DIR = PHASE5_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------
# 2. Load document embeddings
# -----------------------------------------

def load_embeddings():

    embeddings = np.load(EMBEDDINGS_FILE)

    embeddings = embeddings.astype(np.float32)

    print("Embeddings loaded successfully")
    print("Embedding shape:", embeddings.shape)

    return embeddings


# -----------------------------------------
# 3. Normalize embeddings
# -----------------------------------------

def normalize_embeddings(embeddings):

    faiss.normalize_L2(embeddings)

    print("Embeddings normalized successfully")

    return embeddings


# -----------------------------------------
# 4. Create IVF index
# -----------------------------------------

def create_ivf_index(embeddings):

    dimension = embeddings.shape[1]

    # Number of clusters
    nlist = 5

    # Create a coarse quantizer
    quantizer = faiss.IndexFlatIP(dimension)

    # Create IVF index using inner product
    index = faiss.IndexIVFFlat(
        quantizer,
        dimension,
        nlist,
        faiss.METRIC_INNER_PRODUCT
    )

    # Train the index to identify clusters
    print("\nTraining IVF index...")

    index.train(embeddings)

    print("IVF training completed")

    # Add document vectors
    index.add(embeddings)

    print("Vectors added successfully")
    print("Total vectors:", index.ntotal)
    print("Number of clusters:", nlist)

    return index


# -----------------------------------------
# 5. Save IVF index
# -----------------------------------------

def save_ivf_index(index):

    index_path = OUTPUT_DIR / "handbook_ivf.index"

    faiss.write_index(index, str(index_path))

    print("\nIVF index saved successfully")
    print("Saved at:", index_path)


# -----------------------------------------
# 6. Main execution
# -----------------------------------------

if __name__ == "__main__":

    embeddings = load_embeddings()

    # Normalize a copy of the embeddings
    embeddings = normalize_embeddings(embeddings.copy())

    # Create IVF index
    index = create_ivf_index(embeddings)

    # Save index
    save_ivf_index(index)
