
import numpy as np


def cosine_similarity(vector_a, vector_b):
    """
    Calculate cosine similarity between two vectors.
    """

    # Convert inputs into NumPy arrays
    vector_a = np.asarray(vector_a, dtype=float)
    vector_b = np.asarray(vector_b, dtype=float)

    # Check whether vector dimensions match
    if vector_a.shape != vector_b.shape:
        raise ValueError("Both vectors must have the same dimensions")

    # Calculate dot product
    dot_product = np.dot(vector_a, vector_b)

    # Calculate vector magnitudes
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    # Prevent division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Zero vectors are not allowed")

    # Calculate cosine similarity
    similarity = dot_product / (magnitude_a * magnitude_b)

    return float(similarity)


def euclidean_distance(vector_a, vector_b):
    """
    Calculate Euclidean distance between two vectors.
    """

    vector_a = np.asarray(vector_a, dtype=float)
    vector_b = np.asarray(vector_b, dtype=float)

    if vector_a.shape != vector_b.shape:
        raise ValueError("Both vectors must have the same dimensions")

    distance = np.linalg.norm(vector_a - vector_b)

    return float(distance)



if __name__ == "__main__":

    vector_a = np.array([1, 2, 3])
    vector_b = np.array([4, 5, 6])

    cosine = cosine_similarity(vector_a, vector_b)
    distance = euclidean_distance(vector_a, vector_b)

    print("Cosine Similarity:", cosine)
    print("Euclidean Distance:", distance)