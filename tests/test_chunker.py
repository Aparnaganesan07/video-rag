"""Unit tests for the time-window chunker (no models or network needed)."""

import pytest

from app.ingest.chunker import chunk_segments


def make_segments(n: int, length: float = 5.0) -> list[dict]:
    """n back-to-back segments, each `length` seconds long: 'w0', 'w1', ..."""
    return [{"start": i * length, "end": (i + 1) * length, "text": f"w{i}"} for i in range(n)]


def test_empty_input_returns_no_chunks():
    assert chunk_segments([], "vid") == []


def test_short_video_is_a_single_chunk():
    chunks = chunk_segments(make_segments(4), "vid", chunk_seconds=45, overlap_seconds=10)
    assert len(chunks) == 1
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 20.0
    assert chunks[0]["text"] == "w0 w1 w2 w3"


def test_chunks_are_about_target_length():
    chunks = chunk_segments(make_segments(60), "vid", chunk_seconds=45, overlap_seconds=10)
    for c in chunks[:-1]:  # the last chunk may be shorter
        assert 45 <= c["end"] - c["start"] <= 50  # at most one segment over target


def test_consecutive_chunks_overlap_by_about_overlap_seconds():
    chunks = chunk_segments(make_segments(60), "vid", chunk_seconds=45, overlap_seconds=10)
    for a, b in zip(chunks, chunks[1:]):
        overlap = a["end"] - b["start"]
        assert 5 <= overlap <= 10


def test_every_segment_is_covered_and_order_is_kept():
    segs = make_segments(37)
    chunks = chunk_segments(segs, "vid", chunk_seconds=45, overlap_seconds=10)
    covered = {w for c in chunks for w in c["text"].split()}
    assert covered == {s["text"] for s in segs}
    starts = [c["start"] for c in chunks]
    assert starts == sorted(starts)
    assert chunks[-1]["end"] == segs[-1]["end"]


def test_ids_and_metadata():
    chunks = chunk_segments(make_segments(30), "abc123", chunk_seconds=45, overlap_seconds=10)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))
    assert chunks[0]["id"] == "abc123_0000"
    assert all(c["video_id"] == "abc123" for c in chunks)
    assert len({c["id"] for c in chunks}) == len(chunks)  # ids are unique


def test_segment_longer_than_window_does_not_loop_forever():
    segs = [
        {"start": 0, "end": 100, "text": "long"},
        {"start": 100, "end": 105, "text": "short"},
    ]
    chunks = chunk_segments(segs, "vid", chunk_seconds=45, overlap_seconds=10)
    assert chunks[0]["text"] == "long"
    assert chunks[-1]["end"] == 105


def test_gaps_between_segments_are_fine():
    # silence between segments (VAD removed it) — timestamps should still be real ones
    segs = [{"start": i * 20.0, "end": i * 20.0 + 3, "text": f"w{i}"} for i in range(10)]
    chunks = chunk_segments(segs, "vid", chunk_seconds=45, overlap_seconds=10)
    assert chunks[0]["start"] == 0.0
    assert all(c["start"] in {s["start"] for s in segs} for c in chunks)


def test_blank_segment_text_is_skipped():
    segs = [{"start": 0, "end": 5, "text": "hello"}, {"start": 5, "end": 10, "text": "  "}]
    assert chunk_segments(segs, "vid")[0]["text"] == "hello"


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_segments(make_segments(5), "vid", chunk_seconds=10, overlap_seconds=10)
