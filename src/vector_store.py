"""
FAISS vector store: build, save, load, and search.
"""

import json
import os

import faiss
import numpy as np

INDEX_DIR = os.path.join("data", "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a FAISS L2 index from embeddings.

    Args:
        embeddings: numpy array of shape (n, dim).

    Returns:
        FAISS IndexFlatL2 with all embeddings added.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatL2, metadata: list[dict]) -> None:
    """Save the FAISS index and chunk metadata to disk."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_index() -> tuple[faiss.IndexFlatL2, list[dict]]:
    """
    Load the FAISS index and chunk metadata from disk.

    Returns:
        Tuple of (FAISS index, list of chunk metadata dicts).

    Raises:
        FileNotFoundError if index files do not exist.
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            "Index files not found. Please build the index first."
        )

    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def search(
    index: faiss.IndexFlatL2,
    query_embedding: np.ndarray,
    metadata: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Search the FAISS index for the top-k most similar chunks.

    Args:
        index: FAISS index.
        query_embedding: numpy array of shape (1, dim).
        metadata: List of chunk metadata dicts.
        top_k: Number of results to return.

    Returns:
        List of chunk metadata dicts (with added 'score' key),
        sorted by relevance.
    """
    top_k = min(top_k, index.ntotal)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        chunk = metadata[idx].copy()
        chunk["score"] = float(distances[0][i])
        results.append(chunk)

    return results


def index_exists() -> bool:
    """Check whether a FAISS index has been saved to disk."""
    return os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH)
