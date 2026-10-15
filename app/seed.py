"""Ship pre-indexed demo videos with the app.

Why: the hosted demo (Hugging Face Spaces) runs on a cloud server, and YouTube blocks most
downloads from cloud IPs. So instead of indexing at startup, we commit the small results of
indexing (transcripts, OCR slides, chunks: a few hundred KB of JSON) and rebuild the vector
index from them in seconds. No audio, video or model output is needed at runtime.

    python -m app.seed --export   # local: copy data/ -> demo_data/ (for the videos listed below)
    python -m app.seed            # anywhere: if the index is empty, rebuild it from demo_data/
"""

import argparse
import shutil

from app import config
from app.ingest.chunker import searchable_text
from app.ingest.frames import slides_path
from app.ingest.transcriber import transcript_path
from app.retrieval import embedder, vector_store
from app.utils import load_json, save_json

DEMO_DIR = config.ROOT_DIR / "demo_data"
MANIFEST = DEMO_DIR / "videos.json"

# The 4 lectures used in the evaluation (eval/test_questions.json).
DEMO_VIDEOS = ["aircAruvnKk", "IHZwWFHWa-w", "JChPi0CRnDY", "aG7wsIAwXs4"]


def export() -> None:
    """Copy each demo video's transcript, chunks and slide text from data/ into demo_data/."""
    titles = {v["video_id"]: v["title"] for v in vector_store.list_videos()}
    manifest = []
    for vid in DEMO_VIDEOS:
        out = DEMO_DIR / vid
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(transcript_path(vid), out / "transcript.json")
        shutil.copy(config.TRANSCRIPT_DIR / f"{vid}_chunks.json", out / "chunks.json")
        if slides_path(vid).exists():
            # Drop local keyframe paths: only the text is needed to answer questions.
            slides = [{k: s[k] for k in ("start", "end", "text")} for s in load_json(slides_path(vid))]
            save_json(slides, out / "slides.json")
        manifest.append({"video_id": vid, "title": titles.get(vid, vid)})
        print(f"[seed] exported {vid}")
    save_json(manifest, MANIFEST)


def seed(force: bool = False) -> None:
    """Rebuild the vector index from demo_data/ if it's empty (or always, with force)."""
    if not MANIFEST.exists():
        print("[seed] no demo_data/ found, nothing to do")
        return
    if vector_store.get_collection().count() > 0 and not force:
        print("[seed] index already has data, skipping")
        return

    config.ensure_dirs()
    for video in load_json(MANIFEST):
        vid, src = video["video_id"], DEMO_DIR / video["video_id"]
        # Put files where the app expects them (answer.py reads transcript lines + slides from here).
        shutil.copy(src / "transcript.json", transcript_path(vid))
        shutil.copy(src / "chunks.json", config.TRANSCRIPT_DIR / f"{vid}_chunks.json")
        if (src / "slides.json").exists():
            slides_path(vid).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src / "slides.json", slides_path(vid))

        chunks = load_json(src / "chunks.json")
        vector_store.add_chunks(chunks, embedder.embed_passages([searchable_text(c) for c in chunks]), video["title"])
        print(f"[seed] indexed {video['title']} ({len(chunks)} chunks)")


def main():
    parser = argparse.ArgumentParser(description="Export or load the pre-indexed demo videos.")
    parser.add_argument("--export", action="store_true", help="copy data/ -> demo_data/")
    parser.add_argument("--force", action="store_true", help="re-seed even if the index has data")
    args = parser.parse_args()
    export() if args.export else seed(force=args.force)


if __name__ == "__main__":
    main()
