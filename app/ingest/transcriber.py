"""Step 2: speech-to-text with timestamps using faster-whisper.

Transcription is the slowest step, so results are cached as JSON per video.
"""

import time
from functools import lru_cache
from pathlib import Path

from app import config
from app.utils import load_json, save_json

_model = None  # loaded lazily, once per process


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel  # heavy import, only when needed

        print(f"[transcribe] loading Whisper '{config.WHISPER_MODEL}' ({config.WHISPER_COMPUTE_TYPE}, CPU)")
        _model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _model


def transcript_path(video_id: str) -> Path:
    return config.TRANSCRIPT_DIR / f"{video_id}.json"


@lru_cache(maxsize=32)
def load_segments(video_id: str) -> tuple[dict, ...]:
    """Cached transcript segments for an already-indexed video (empty if none on disk)."""
    path = transcript_path(video_id)
    return tuple(load_json(path)["segments"]) if path.exists() else ()


def transcribe(audio_path: str, video_id: str, force: bool = False) -> list[dict]:
    """Return a list of segments: [{"start": 0.0, "end": 4.2, "text": "..."}, ...]."""
    cache = transcript_path(video_id)
    if cache.exists() and not force:
        print(f"[transcribe] cached: {cache.name}")
        return load_json(cache)["segments"]

    model = _get_model()
    t0 = time.time()
    # vad_filter skips long silences -> faster and fewer hallucinated words
    segments_iter, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)

    segments = []
    for seg in segments_iter:  # generator: decoding happens as we iterate
        segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()})
        print(f"\r[transcribe] {seg.end:7.1f}s / {info.duration:.0f}s", end="", flush=True)
    print(f"\n[transcribe] {len(segments)} segments in {time.time() - t0:.0f}s")

    save_json({"video_id": video_id, "language": info.language, "segments": segments}, cache)
    load_segments.cache_clear()  # make sure the fresh transcript is picked up
    return segments
