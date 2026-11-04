import re
from sentence_transformers import SentenceTransformer, util


def semantic_chunking(
    text,
    source,
    page_number,
    similarity_threshold=0.5,
    max_chunk_size=1000,
    model_name="all-MiniLM-L6-v2",
    model=None,
):
    """Group adjacent sentences when their embedding similarity is high."""
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if not sentences:
        return []

    # Allow a shared model to be passed in so it is loaded only once for a whole PDF.
    if model is None:
        model = SentenceTransformer(model_name)

    embeddings = model.encode(sentences, convert_to_tensor=True)
    chunks = []
    current_sentences = [sentences[0]]
    chunk_index = 0

    for index in range(1, len(sentences)):
        similarity = util.cos_sim(embeddings[index - 1], embeddings[index]).item()
        sentence = sentences[index]
        candidate = " ".join(current_sentences + [sentence])

        if similarity >= similarity_threshold and len(candidate) <= max_chunk_size:
            current_sentences.append(sentence)
        else:
            chunks.append({
                "chunk_id": f"semantic_p{page_number:03d}_c{chunk_index:03d}",
                "source": source,
                "file_type": "pdf",
                "page_number": page_number,
                "chunk_index": chunk_index,
                "chunking_strategy": "semantic",
                "text": " ".join(current_sentences),
            })
            chunk_index += 1
            current_sentences = [sentence]

    if current_sentences:
        chunks.append({
            "chunk_id": f"semantic_p{page_number:03d}_c{chunk_index:03d}",
            "source": source,
            "file_type": "pdf",
            "page_number": page_number,
            "chunk_index": chunk_index,
            "chunking_strategy": "semantic",
            "text": " ".join(current_sentences),
        })

    return chunks


if __name__ == "__main__":
    sample = "Attendance is mandatory. Students must maintain 75% attendance. The library opens at 9 AM."
    result = semantic_chunking(sample, "GU_STUDENT_HANDBOOK.pdf", 20, 0.5, 1000)
    for item in result:
        print(item)
