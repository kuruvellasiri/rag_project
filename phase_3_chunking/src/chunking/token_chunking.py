import tiktoken


def token_based_chunking(text, source, page_number, chunk_size=100, overlap=20, encoding_name="o200k_base"):
    """Split text into token windows and decode each window back to text."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks = []
    step = chunk_size - overlap

    for chunk_index, start in enumerate(range(0, len(tokens), step)):
        end = min(start + chunk_size, len(tokens))
        token_slice = tokens[start:end]
        chunk_text = encoding.decode(token_slice)

        if chunk_text.strip():
            chunks.append({
                "chunk_id": f"token_p{page_number:03d}_c{chunk_index:03d}",
                "source": source,
                "file_type": "pdf",
                "page_number": page_number,
                "chunk_index": chunk_index,
                "chunking_strategy": "token_based",
                "start_token": start,
                "end_token": end,
                "token_count": len(token_slice),
                "text": chunk_text,
            })

        if end == len(tokens):
            break

    return chunks


if __name__ == "__main__":
    sample = "Students must maintain attendance. Attendance is important for examinations. Follow university rules."
    result = token_based_chunking(sample, "GU_STUDENT_HANDBOOK.pdf", 20, 20, 5)
    for item in result:
        print(item)
