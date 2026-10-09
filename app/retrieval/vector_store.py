"""ChromaDB wrapper: store chunks with their timestamps, search by vector."""

import chromadb
from chromadb.config import Settings

from app import config

_collection = None


def get_collection():
    global _collection
    if _collection is None:
        config.ensure_dirs()
        client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        # cosine distance matches our normalized bge embeddings
        _collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_chunks(chunks: list[dict], embeddings: list[list[float]], title: str) -> None:
    """Replace any existing chunks for this video, then insert the new ones."""
    col = get_collection()
    video_id = chunks[0]["video_id"]
    col.delete(where={"video_id": video_id})  # re-indexing shouldn't create duplicates
    col.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "video_id": c["video_id"],
                "title": title,
                "chunk_index": c["chunk_index"],
                "start": float(c["start"]),  # timestamps travel with every hit
                "end": float(c["end"]),
            }
            for c in chunks
        ],
    )


def query(embedding: list[float], k: int, video_id: str | None = None) -> list[dict]:
    """Return the k nearest chunks, optionally restricted to one video."""
    res = get_collection().query(
        query_embeddings=[embedding],
        n_results=k,
        where={"video_id": video_id} if video_id else None,
    )
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({**meta, "text": doc, "score": 1.0 - dist})  # cosine distance -> similarity
    return hits


def list_videos() -> list[dict]:
    """Unique indexed videos with their chunk counts."""
    metas = get_collection().get(include=["metadatas"])["metadatas"]
    videos: dict[str, dict] = {}
    for m in metas:
        v = videos.setdefault(m["video_id"], {"video_id": m["video_id"], "title": m["title"], "chunks": 0})
        v["chunks"] += 1
    return list(videos.values())
