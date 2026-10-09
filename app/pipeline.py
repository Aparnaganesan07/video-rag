"""Glue code: the two things the app does — index a video, search it."""

from app import config
from app.generation.answer import generate_answer
from app.ingest.chunker import chunk_segments
from app.ingest.downloader import download_audio
from app.ingest.transcriber import transcribe
from app.retrieval import embedder, vector_store
from app.utils import save_json


def index_video(source: str, force: bool = False) -> dict:
    """download -> transcribe -> chunk -> embed -> store. Returns video metadata."""
    meta = download_audio(source)
    segments = transcribe(meta["audio_path"], meta["video_id"], force=force)
    if not segments:
        raise RuntimeError("Transcription produced no speech segments.")

    chunks = chunk_segments(segments, meta["video_id"])
    print(f"[chunk] {len(chunks)} chunks (~{config.CHUNK_SECONDS:.0f}s, {config.OVERLAP_SECONDS:.0f}s overlap)")

    embeddings = embedder.embed_passages([c["text"] for c in chunks])
    vector_store.add_chunks(chunks, embeddings, title=meta["title"])
    print(f"[store] indexed '{meta['title']}' ({meta['video_id']})")

    # Keep a copy of the chunks on disk: handy for debugging and for building eval sets.
    save_json(chunks, config.TRANSCRIPT_DIR / f"{meta['video_id']}_chunks.json")
    return {**meta, "num_chunks": len(chunks)}


def search(question: str, k: int = config.TOP_K, video_id: str | None = None) -> list[dict]:
    """Semantic search over indexed chunks. Phase 3 adds BM25 + reranking here."""
    return vector_store.query(embedder.embed_query(question), k=k, video_id=video_id)


def ask(question: str, video_id: str | None = None, k: int = config.ANSWER_TOP_K) -> dict:
    """Full RAG: retrieve chunks, then have the LLM answer with [mm:ss] citations.

    Returns {"answer", "citations", "sources", "model"}.
    """
    hits = search(question, k=k, video_id=video_id)
    result = generate_answer(question, hits)
    return {**result, "sources": hits}
