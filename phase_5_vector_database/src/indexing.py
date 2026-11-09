
import numpy as np
import faiss
from pathlib import Path


# --------------------------------
# 1. Locate project directories
# --------------------------------

PHASE5_DIR = Path(__file__).resolve().parents[1]

PROJECT_DIR = PHASE5_DIR.parent

PHASE4_OUTPUT_DIR = (
    PROJECT_DIR / "phase_4_embeddings" / "outputs"
)

EMBEDDINGS_FILE = (
    PHASE4_OUTPUT_DIR / "handbook_embeddings.npy"
)

OUTPUT_DIR = PHASE5_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------
# 2. Load document embeddings
# --------------------------------

def load_embeddings():

    embeddings = np.load(EMBEDDINGS_FILE)

    print("Embeddings loaded successfully")
    print("Embedding shape:", embeddings.shape)

    return embeddings


# --------------------------------
# 3. Normalize embeddings
# --------------------------------

def normalize_embeddings(embeddings):

    faiss.normalize_L2(embeddings)

    print("Embeddings normalized successfully")

    return embeddings


# --------------------------------
# 4. Create FAISS index
# --------------------------------

def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print("FAISS index created successfully")
    print("Vector dimensions:", dimension)
    print("Total vectors stored:", index.ntotal)

    return index


# --------------------------------
# 5. Save FAISS index
# --------------------------------

def save_faiss_index(index):

    index_path = OUTPUT_DIR / "handbook.index"

    faiss.write_index(index, str(index_path))

    print("FAISS index saved at:", index_path)


# --------------------------------
# 6. Main execution
# --------------------------------

if __name__ == "__main__":

    embeddings = load_embeddings()

    # FAISS normalization modifies the array in place,
    # so normalize a copy rather than the original loaded data.
    embeddings = normalize_embeddings(embeddings.copy())

    index = create_faiss_index(embeddings)

    save_faiss_index(index)
