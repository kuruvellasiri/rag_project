
import re
from pathlib import Path



import re


def sentence_based_chunking(
    text,
    source,
    page_number,
    chunk_size=500,
    overlap_sentences=1
):

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    if overlap_sentences < 0:
        raise ValueError("Overlap sentences cannot be negative")

    # Step 1: Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []

    current_sentences = []
    current_length = 0
    chunk_index = 0

    # Step 2: Process each sentence
    for sentence in sentences:

        sentence_length = len(sentence)

        additional_length = sentence_length

        if current_sentences:
            additional_length += 1

        # Step 3: Check whether adding the sentence exceeds chunk size
        if current_sentences and current_length + additional_length > chunk_size:

            chunk_text = " ".join(current_sentences)

            chunks.append({
                "chunk_id": f"sentence_p{page_number:03d}_c{chunk_index:03d}",
                "source": source,
                "file_type": "pdf",
                "page_number": page_number,
                "chunk_index": chunk_index,
                "chunking_strategy": "sentence_based",
                "text": chunk_text
            })

            chunk_index += 1

            # Step 4: Preserve previous sentences for overlap
            if overlap_sentences > 0:
                current_sentences = current_sentences[-overlap_sentences:]
            else:
                current_sentences = []

            current_length = len(" ".join(current_sentences))

        # Step 5: Add the new sentence
        if current_sentences:
            current_length += 1

        current_sentences.append(sentence)
        current_length += sentence_length

    # Step 6: Save the remaining sentences
    if current_sentences:

        chunk_text = " ".join(current_sentences)

        chunks.append({
            "chunk_id": f"sentence_p{page_number:03d}_c{chunk_index:03d}",
            "source": source,
            "file_type": "pdf",
            "page_number": page_number,
            "chunk_index": chunk_index,
            "chunking_strategy": "sentence_based",
            "text": chunk_text
        })

    return chunks


# Test using the actual Student Handbook
if __name__ == "__main__":

    project_dir = Path(__file__).resolve().parent.parent

    document_path = (
        project_dir / "documents" / "student_handbook.txt"
    )

    text = document_path.read_text(encoding="utf-8-sig")

    chunks = sentence_based_chunking(
            text=text,
            source=document_path.name,
            chunk_size=500,
            overlap_sentences=1
    )

    for chunk in chunks:

        print("\n" + "=" * 60)
        print("Chunk ID:", chunk["chunk_id"])
        print("Chunk Index:", chunk["chunk_index"])
        print("Content:")
        print(chunk["text"])

    print("\nTotal chunks:", len(chunks))
