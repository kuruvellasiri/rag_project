def fixed_size_chunking(text, source, page_number, chunk_size=500, overlap=50):
    """Split page text into fixed character windows with character overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "chunk_id": f"fixed_p{page_number:03d}_c{chunk_index:03d}",
                "source": source,
                "file_type": "pdf",
                "page_number": page_number,
                "chunk_index": chunk_index,
                "chunking_strategy": "fixed_size",
                "start_char": start,
                "end_char": end,
                "text": chunk_text,
            })
            chunk_index += 1

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    sample = "Students must maintain attendance. Attendance is important for examinations. Follow university rules."
    result = fixed_size_chunking(sample, "GU_STUDENT_HANDBOOK.pdf", 20, 50, 10)
    for item in result:
        print(item)
