
def _split_recursive(text, chunk_size, separators):
    """Recursively split text using progressively finer separators."""
    if len(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        # Final fallback: hard split when no separators remain.
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator = separators[0]
    if separator == "":
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(separator)
    if len(parts) == 1:
        return _split_recursive(text, chunk_size, separators[1:])

    # Reattach separators to the preceding part where possible.
    pieces = []
    for index, part in enumerate(parts):
        if index < len(parts) - 1:
            part += separator
        if part:
            pieces.append(part)

    # Merge small pieces while respecting the target size.
    merged = []
    current = ""
    for piece in pieces:
        candidate = current + piece
        if current and len(candidate) > chunk_size:
            merged.extend(_split_recursive(current, chunk_size, separators[1:]))
            current = piece
        else:
            current = candidate
    if current:
        merged.extend(_split_recursive(current, chunk_size, separators[1:]))
    return merged


def recursive_chunking(text, source, page_number, chunk_size=500, overlap=50):
    """Split by paragraphs, lines, sentences, spaces, then characters."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    separators = ["\n\n", "\n", ". ", " ", ""]
    raw_chunks = _split_recursive(text, chunk_size, separators)

    # Add character overlap between the resulting chunks.
    chunks = []
    previous_tail = ""
    for index, raw in enumerate(raw_chunks):
        chunk_text = (previous_tail + raw).strip()
        if chunk_text:
            chunks.append({
                "chunk_id": f"recursive_p{page_number:03d}_c{index:03d}",
                "source": source,
                "file_type": "pdf",
                "page_number": page_number,
                "chunk_index": index,
                "chunking_strategy": "recursive",
                "text": chunk_text,
            })
        previous_tail = raw[-overlap:] if overlap else ""

    return chunks


if __name__ == "__main__":
    sample = "Attendance rules\n\nStudents must attend classes.\nStudents must follow university rules.\n\nLibrary rules\n\nReturn books on time."
    result = recursive_chunking(sample, "GU_STUDENT_HANDBOOK.pdf", 20, 60, 10)
    for item in result:
        print(item)
