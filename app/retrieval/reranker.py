"""Second-stage ranking with a cross-encoder (bge-reranker-base).

Bi-encoder (our embedder): question and chunk are embedded SEPARATELY, then compared.
  Fast (chunk vectors are precomputed), but the model never sees them together.
Cross-encoder (this): reads "question + chunk" TOGETHER and outputs one relevance score.
  Much more accurate, but must run once per (question, chunk) pair at query time,
  so we only use it to re-order the top candidates the fast retrievers found.
"""

from app import config
from app.ingest.chunker import searchable_text

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder  # heavy import

        opts = {"max_length": config.RERANK_MAX_LENGTH, "device": "cpu"}
        try:
            _model = CrossEncoder(config.RERANKER_MODEL, local_files_only=True, **opts)
        except Exception:
            print(f"[rerank] downloading {config.RERANKER_MODEL} (first run only, ~1 GB)")
            _model = CrossEncoder(config.RERANKER_MODEL, **opts)
    return _model


def rerank(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """Re-order hits by cross-encoder relevance and keep the best top_k."""
    if not hits:
        return []
    scores = _get_model().predict([(query, searchable_text(h)) for h in hits], show_progress_bar=False)
    rescored = [{**h, "score": float(s), "rerank_score": float(s)} for h, s in zip(hits, scores)]
    return sorted(rescored, key=lambda h: h["rerank_score"], reverse=True)[:top_k]
