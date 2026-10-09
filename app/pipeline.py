"""Glue code: the things the app does — index videos, search them, answer questions."""

from app import config
from app.generation.answer import generate_answer
from app.ingest.chunker import attach_slides, chunk_segments, searchable_text
from app.ingest.downloader import download_audio, download_video, expand_source
from app.ingest.frames import extract_slides
from app.ingest.transcriber import transcribe
from app.retrieval import embedder, reranker, vector_store
from app.retrieval.hybrid_search import hybrid_search, vector_search
from app.utils import save_json


def index_video(source: str, force: bool = False, slides: bool | None = None) -> dict:
    """download -> transcribe -> (slides OCR) -> chunk -> embed -> store. Returns video metadata."""
    slides = config.ENABLE_SLIDES if slides is None else slides
    meta = download_audio(source)
    segments = transcribe(meta["audio_path"], meta["video_id"], force=force)
    if not segments:
        raise RuntimeError("Transcription produced no speech segments.")

    chunks = chunk_segments(segments, meta["video_id"])
    print(f"[chunk] {len(chunks)} chunks (~{config.CHUNK_SECONDS:.0f}s, {config.OVERLAP_SECONDS:.0f}s overlap)")

    slide_list = []
    if slides:
        # On-screen text is a bonus: if it fails (e.g. audio-only file), index the speech anyway.
        try:
            slide_list = extract_slides(download_video(meta), meta["video_id"], force=force)
        except Exception as e:
            print(f"[slides] skipped: {e}")
    attach_slides(chunks, slide_list)

    embeddings = embedder.embed_passages([searchable_text(c) for c in chunks])
    vector_store.add_chunks(chunks, embeddings, title=meta["title"])
    print(f"[store] indexed '{meta['title']}' ({meta['video_id']})")

    # Keep a copy of the chunks on disk: handy for debugging and for building eval sets.
    save_json(chunks, config.TRANSCRIPT_DIR / f"{meta['video_id']}_chunks.json")
    return {**meta, "num_chunks": len(chunks), "num_slides": len(slide_list)}


def index_source(source: str, force: bool = False, slides: bool | None = None) -> list[dict]:
    """Index a single video or every video in a playlist. One failure doesn't stop the rest."""
    results = []
    urls = expand_source(source)
    for i, url in enumerate(urls, 1):
        if len(urls) > 1:
            print(f"\n=== [{i}/{len(urls)}] {url}")
        try:
            results.append(index_video(url, force=force, slides=slides))
        except Exception as e:
            print(f"[error] {url}: {e}")
            results.append({"source": url, "error": str(e)})
    return results


def search(
    question: str,
    k: int = config.TOP_K,
    video_id: str | None = None,
    hybrid: bool | None = None,
    rerank: bool | None = None,
) -> list[dict]:
    """Two-stage retrieval: get candidates (vector or hybrid), then optionally rerank the best few."""
    hybrid = config.USE_HYBRID if hybrid is None else hybrid
    rerank = config.USE_RERANKER if rerank is None else rerank

    n = max(k, config.CANDIDATES_K)
    candidates = hybrid_search(question, n, video_id) if hybrid else vector_search(question, n, video_id)
    if rerank:
        return reranker.rerank(question, candidates[: max(k, config.RERANK_CANDIDATES)], top_k=k)
    return candidates[:k]


def ask(
    question: str,
    video_id: str | None = None,
    k: int = config.ANSWER_TOP_K,
    hybrid: bool | None = None,
    rerank: bool | None = None,
) -> dict:
    """Full RAG: retrieve chunks, then have the LLM answer with [mm:ss] citations.

    Returns {"answer", "citations", "sources", "model"}.
    """
    hits = search(question, k=k, video_id=video_id, hybrid=hybrid, rerank=rerank)
    result = generate_answer(question, hits)
    return {**result, "sources": hits}
