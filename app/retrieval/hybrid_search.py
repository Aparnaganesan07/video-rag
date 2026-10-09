"""Hybrid retrieval: vector search (meaning) + BM25 (exact words), fused with RRF.

Why both?
- Vectors understand paraphrases ("how does it learn" ~ "gradient descent")
  but are weak on rare exact terms, acronyms and names ("ReLU", "HNSW").
- BM25 is the opposite: great at exact terms, blind to synonyms.

Reciprocal Rank Fusion (RRF) merges the two ranked lists using only ranks:
    score(doc) = sum over lists of 1 / (RRF_K + rank)
No score normalisation is needed: cosine similarities and BM25 scores live on
different scales, but ranks are always comparable.
"""

import re

from rank_bm25 import BM25Okapi

from app import config
from app.ingest.chunker import searchable_text
from app.retrieval import embedder, vector_store

# Common words that carry no meaning for keyword search.
STOPWORDS = set(
    "a an and are as at be but by can do does for from has have how i if in into is it its "
    "of on or so than that the their then there these this to was we what when where which "
    "who why will with you your".split()
)

_bm25_cache: dict = {"version": None, "count": None, "bm25": None, "docs": []}


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, minus stopwords: 'What is ReLU?' -> ['relu']."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def _get_bm25():
    """Build the BM25 index over all chunks; rebuild only when the store changed."""
    count = vector_store.get_collection().count()
    if _bm25_cache["version"] != vector_store.version or _bm25_cache["count"] != count:
        docs = vector_store.get_all()
        corpus = [tokenize(searchable_text(d)) or ["_empty_"] for d in docs]
        _bm25_cache.update(
            version=vector_store.version, count=count, docs=docs, bm25=BM25Okapi(corpus) if docs else None
        )
    return _bm25_cache["bm25"], _bm25_cache["docs"]


def bm25_search(query: str, k: int, video_id: str | None = None) -> list[dict]:
    """Top-k chunks by BM25 keyword score (only chunks that share at least one word)."""
    bm25, docs = _get_bm25()
    tokens = tokenize(query)
    if bm25 is None or not tokens:
        return []
    scores = bm25.get_scores(tokens)
    ranked = sorted(
        (i for i, d in enumerate(docs) if scores[i] > 0 and (video_id is None or d["video_id"] == video_id)),
        key=lambda i: scores[i],
        reverse=True,
    )
    return [{**docs[i], "score": float(scores[i]), "bm25_score": float(scores[i])} for i in ranked[:k]]


def vector_search(query: str, k: int, video_id: str | None = None) -> list[dict]:
    return vector_store.query(embedder.embed_query(query), k=k, video_id=video_id)


def rrf_fuse(result_lists: list[list[dict]], rrf_k: int = config.RRF_K) -> list[dict]:
    """Merge ranked lists by Reciprocal Rank Fusion. Keeps each chunk's per-retriever scores."""
    fused: dict[str, dict] = {}
    for results in result_lists:
        for rank, hit in enumerate(results, 1):
            entry = fused.setdefault(hit["id"], {**hit, "rrf_score": 0.0})
            entry.update({k: v for k, v in hit.items() if k.endswith("_score")})  # keep vector_ & bm25_score
            entry["rrf_score"] += 1.0 / (rrf_k + rank)
    ranked = sorted(fused.values(), key=lambda h: h["rrf_score"], reverse=True)
    for h in ranked:
        h["score"] = h["rrf_score"]
    return ranked


def hybrid_search(query: str, k: int = config.CANDIDATES_K, video_id: str | None = None) -> list[dict]:
    """Vector top-k and BM25 top-k, fused into one ranking."""
    return rrf_fuse([vector_search(query, k, video_id), bm25_search(query, k, video_id)])[:k]
