"""Step 3: group Whisper segments into overlapping time-window chunks.

Why time windows instead of fixed token counts?
Every chunk keeps a precise [start, end] range, so a search hit maps straight
to a moment in the video. We cut only on segment boundaries, so we never split
a sentence mid-way.

Example with 45s chunks and 10s overlap:
    chunk 0: 00:00 - 00:46
    chunk 1: 00:37 - 01:22   <- starts ~10s before chunk 0 ended
    ...
"""

from app import config


def chunk_segments(
    segments: list[dict],
    video_id: str,
    chunk_seconds: float = config.CHUNK_SECONDS,
    overlap_seconds: float = config.OVERLAP_SECONDS,
) -> list[dict]:
    """Return chunks: [{"id", "video_id", "chunk_index", "start", "end", "text"}, ...]."""
    if overlap_seconds >= chunk_seconds:
        raise ValueError("overlap_seconds must be smaller than chunk_seconds")

    chunks = []
    n = len(segments)
    i = 0  # index of the first segment in the current chunk

    while i < n:
        window_start = segments[i]["start"]

        # Grow the window until it spans at least chunk_seconds (or we run out).
        j = i
        while j < n - 1 and segments[j]["end"] - window_start < chunk_seconds:
            j += 1
        window = segments[i : j + 1]

        chunks.append({
            "id": f"{video_id}_{len(chunks):04d}",
            "video_id": video_id,
            "chunk_index": len(chunks),
            "start": window[0]["start"],
            "end": window[-1]["end"],
            "text": " ".join(s["text"].strip() for s in window if s["text"].strip()),
        })

        if j == n - 1:  # reached the last segment -> done
            break

        # Next chunk starts at the first segment inside the overlap zone.
        # k starts at i + 1, so we always move forward (no infinite loop).
        overlap_from = window[-1]["end"] - overlap_seconds
        k = i + 1
        while k <= j and segments[k]["start"] < overlap_from:
            k += 1
        i = k

    return chunks
