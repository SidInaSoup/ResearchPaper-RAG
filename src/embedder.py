"""
Sentence-transformer embedding generation.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Lazy-loaded singleton
_model = None
MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of text strings into dense embeddings.

    Args:
        texts: List of text strings to embed.

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.astype("float32")


def get_embedding_dim() -> int:
    """Return the dimensionality of the embedding model."""
    model = _get_model()
    return model.get_sentence_embedding_dimension()
