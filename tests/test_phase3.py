"""Unit tests for Phase 3 helpers: slide OCR cleanup, slide/chunk merging, BM25 tokenizing, RRF, playlists."""

from app.ingest.chunker import attach_slides, searchable_text
from app.ingest.downloader import is_playlist_url
from app.ingest.frames import clean_ocr_text, merge_duplicate_slides, sample_times
from app.retrieval.hybrid_search import rrf_fuse, tokenize


# ---------- frames.py ----------

def test_clean_ocr_text_drops_low_confidence_and_tiny_pieces():
    pieces = [("Gradient", 0.95), ("descent", 0.9), ("~", 0.99), ("xq#", 0.1), ("  steps  ", 0.8)]
    assert clean_ocr_text(pieces, min_conf=0.4) == "Gradient descent steps"


def test_merge_duplicate_slides_merges_near_identical_consecutive_frames():
    frames = [
        {"start": 0, "end": 10, "text": "Problem statement", "frame": "a.jpg"},
        {"start": 10, "end": 20, "text": "Problem statemnt", "frame": "b.jpg"},  # OCR typo, same slide
        {"start": 20, "end": 30, "text": "", "frame": "c.jpg"},                  # no text: dropped
        {"start": 30, "end": 40, "text": "General approaches", "frame": "d.jpg"},
    ]
    slides = merge_duplicate_slides(frames, ratio=0.85)
    assert [(s["start"], s["end"]) for s in slides] == [(0, 20), (30, 40)]


def test_merge_keeps_longer_text_when_a_bullet_appears():
    frames = [
        {"start": 0, "end": 5, "text": "Step-back prompting: abstraction", "frame": "a.jpg"},
        {"start": 5, "end": 9, "text": "Step-back prompting: abstraction more", "frame": "b.jpg"},
    ]
    (slide,) = merge_duplicate_slides(frames, ratio=0.85)
    assert slide["text"].endswith("more") and slide["frame"] == "b.jpg" and slide["end"] == 9


def test_sample_times_one_frame_per_short_scene_and_resamples_long_ones():
    samples = sample_times([(0, 10), (10, 100)], resample=45)
    assert [(s, e) for _t, s, e in samples] == [(0, 10), (10, 55), (55, 100)]
    assert all(s <= t <= e for t, s, e in samples)


# ---------- chunker.py ----------

def test_attach_slides_by_time_overlap():
    chunks = [{"start": 0, "end": 45, "text": "a"}, {"start": 35, "end": 80, "text": "b"}]
    slides = [
        {"start": 0, "end": 40, "text": "Title slide"},   # overlaps both chunks
        {"start": 50, "end": 70, "text": "Results"},      # only the second
        {"start": 90, "end": 99, "text": "Q&A"},          # neither
    ]
    attach_slides(chunks, slides)
    assert chunks[0]["slide_text"] == "Title slide"
    assert chunks[1]["slide_text"] == "Title slide | Results"


def test_searchable_text_includes_slides_only_when_present():
    assert searchable_text({"text": "hello", "slide_text": ""}) == "hello"
    assert searchable_text({"text": "hello", "slide_text": "SLIDE"}) == "hello\nOn screen: SLIDE"


# ---------- hybrid_search.py ----------

def test_tokenize_lowercases_and_removes_stopwords():
    assert tokenize("What is ReLU, and why is it used in 2024?") == ["relu", "used", "2024"]


def _hit(cid, **scores):
    return {"id": cid, "text": cid, **scores}


def test_rrf_rewards_documents_ranked_high_in_both_lists():
    vector = [_hit("a", vector_score=0.9), _hit("b", vector_score=0.8), _hit("c", vector_score=0.7)]
    bm25 = [_hit("c", bm25_score=12.0), _hit("b", bm25_score=9.0)]
    fused = rrf_fuse([vector, bm25], rrf_k=60)
    # c: 1/63 + 1/61 = 0.03227 > b: 1/62 + 1/62 = 0.03226 > a: 1/61 only
    assert [h["id"] for h in fused] == ["c", "b", "a"]
    b = fused[1]
    assert b["vector_score"] == 0.8 and b["bm25_score"] == 9.0  # per-retriever scores are kept
    assert abs(b["score"] - (1 / 62 + 1 / 62)) < 1e-12


def test_rrf_with_one_empty_list_keeps_the_other_order():
    vector = [_hit("a"), _hit("b")]
    assert [h["id"] for h in rrf_fuse([vector, []])] == ["a", "b"]


# ---------- downloader.py ----------

def test_playlist_detection():
    assert is_playlist_url("https://www.youtube.com/playlist?list=PL123")
    assert not is_playlist_url("https://www.youtube.com/watch?v=abc&list=PL123")  # single video in a playlist
    assert not is_playlist_url("https://www.youtube.com/watch?v=abc")
    assert not is_playlist_url(r"C:\videos\lecture.mp4")
