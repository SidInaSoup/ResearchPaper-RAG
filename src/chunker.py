"""
Text chunking with overlap for RAG indexing.
"""


def chunk_text(
    pages: list[dict],
    filename: str,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    """
    Split page-level text into overlapping word-based chunks with metadata.

    Args:
        pages: List of dicts from pdf_loader (keys: 'page', 'text').
        filename: Original PDF filename.
        chunk_size: Target number of words per chunk.
        overlap: Number of overlapping words between consecutive chunks.

    Returns:
        List of chunk dicts with keys:
            - text: chunk text
            - filename: source PDF filename
            - page: starting page number for the chunk
            - chunk_index: sequential chunk index
    """
    # Flatten all words with page tracking
    word_entries = []  # list of (word, page_number)
    for page_obj in pages:
        words = page_obj["text"].split()
        for w in words:
            word_entries.append((w, page_obj["page"]))

    if not word_entries:
        return []

    chunks = []
    chunk_index = 0
    start = 0
    total_words = len(word_entries)

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = word_entries[start:end]

        # The page number is the page of the first word in the chunk
        page_number = chunk_words[0][1]

        chunk_text_str = " ".join(w for w, _ in chunk_words)

        chunks.append({
            "text": chunk_text_str,
            "filename": filename,
            "page": page_number,
            "chunk_index": chunk_index,
        })

        chunk_index += 1
        start += chunk_size - overlap

        # If the next chunk would be too small, just stop
        if start >= total_words:
            break

    return chunks
