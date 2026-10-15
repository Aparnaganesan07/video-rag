"""Unit tests for the evaluation metrics (so the numbers in the README can be trusted)."""

from eval.evaluate import TOLERANCE, first_correct_rank, is_correct, summarize

Q = {"video_id": "v1", "seconds": [100]}


def chunk(video_id="v1", start=0.0, end=45.0):
    return {"video_id": video_id, "start": start, "end": end}


def test_correct_when_timestamp_inside_chunk():
    assert is_correct(chunk(start=80, end=125), Q)


def test_tolerance_on_both_sides():
    assert is_correct(chunk(start=100 + TOLERANCE, end=150), Q)       # starts 20s after: still ok
    assert is_correct(chunk(start=30, end=100 - TOLERANCE), Q)        # ends 20s before: still ok
    assert not is_correct(chunk(start=100 + TOLERANCE + 1, end=200), Q)


def test_wrong_video_is_never_correct():
    assert not is_correct(chunk(video_id="v2", start=80, end=125), Q)


def test_any_of_several_ground_truth_times_counts():
    q = {"video_id": "v1", "seconds": [10, 500]}
    assert is_correct(chunk(start=480, end=525), q)


def test_first_correct_rank():
    hits = [chunk(start=300, end=345), chunk(video_id="v2", start=80, end=125), chunk(start=90, end=135)]
    assert first_correct_rank(hits, Q) == 3
    assert first_correct_rank(hits[:2], Q) is None


def test_summarize_hit_at_k_and_mrr():
    # ranks of the first correct chunk for 4 questions: 1st, 2nd, 4th, not found
    s = summarize([1, 2, 4, None])
    assert s["hit@1"] == 0.25
    assert s["hit@3"] == 0.5
    assert s["hit@5"] == 0.75
    assert abs(s["mrr"] - (1 + 0.5 + 0.25 + 0) / 4) < 1e-12
