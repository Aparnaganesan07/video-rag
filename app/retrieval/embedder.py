"""Turn text into vectors with bge-small-en-v1.5 (384-dim, ~130 MB, fast on CPU)."""

from app import config

_model = None  # loaded lazily, once per process


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # heavy import

        try:
            # Use the copy on disk if we have one: skips slow network checks at startup.
            _model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu", local_files_only=True)
        except Exception:
            _model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")  # first run: download
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed chunk texts. Normalized, so dot product == cosine similarity."""
    vectors = _get_model().encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return vectors.tolist()


def embed_query(question: str) -> list[float]:
    """Embed a search query. bge expects an instruction prefix on queries only."""
    vector = _get_model().encode(config.BGE_QUERY_PREFIX + question, normalize_embeddings=True)
    return vector.tolist()
