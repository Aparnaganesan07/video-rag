"""Unit tests for citation parsing and prompt building (no LLM call, no network)."""

from app.generation import answer as ans
from app.utils import format_timestamp, parse_timestamp

HITS = [
    {"video_id": "vidA", "title": "Video A", "start": 170.0, "end": 220.0, "text": "neurons hold numbers"},
    {"video_id": "vidB", "title": "Video B", "start": 500.0, "end": 545.0, "text": "gradient descent"},
]


def test_parse_timestamp_roundtrip():
    for seconds in (0, 59, 84, 600, 3725):
        assert parse_timestamp(format_timestamp(seconds)) == seconds


def test_extracts_and_matches_citations_to_sources():
    cites = ans.extract_citations("Neurons hold numbers [02:54]. GD moves downhill [08:30].", HITS)
    assert [c["label"] for c in cites] == ["02:54", "08:30"]
    assert cites[0]["video_id"] == "vidA" and cites[0]["seconds"] == 174
    assert cites[1]["video_id"] == "vidB"
    assert all(c["grounded"] for c in cites)


def test_ungrounded_citation_is_flagged():
    cites = ans.extract_citations("Made-up claim [15:00].", HITS)
    assert cites[0]["grounded"] is False


def test_slack_allows_citation_just_outside_chunk():
    cites = ans.extract_citations("Edge [03:43].", HITS)  # chunk ends at 03:40, slack is 5s
    assert cites[0]["grounded"] is True


def test_multiple_timestamps_in_one_bracket_and_duplicates():
    cites = ans.extract_citations("A [02:54, 03:00] B [02:54-03:10] C [02:54]", HITS)
    assert [c["label"] for c in cites] == ["02:54", "03:00", "03:10"]


def test_hour_timestamps():
    hits = [{"video_id": "v", "title": "t", "start": 3700.0, "end": 3750.0, "text": "x"}]
    cites = ans.extract_citations("Late in the lecture [1:02:05].", hits)
    assert cites[0]["seconds"] == 3725 and cites[0]["grounded"]


def test_non_timestamp_brackets_are_ignored():
    assert ans.extract_citations("See [1] and [note] and [12].", HITS) == []


def test_fullwidth_brackets_are_normalized():
    text = ans.normalize_citations("Claim【02:54】【03:00】.")
    assert text == "Claim[02:54][03:00]."
    assert len(ans.extract_citations(text, HITS)) == 2


def test_context_has_source_headers_and_timestamped_lines(monkeypatch):
    fake_segments = (
        {"start": 170.0, "end": 180.0, "text": "first line"},
        {"start": 181.0, "end": 190.0, "text": "second line"},
        {"start": 400.0, "end": 410.0, "text": "outside the chunk"},
    )
    monkeypatch.setattr(ans, "load_segments", lambda vid: fake_segments if vid == "vidA" else ())
    ctx = ans.build_context(HITS)
    assert 'SOURCE 1 | video: "Video A" | 02:50-03:40' in ctx
    assert "[02:50] first line" in ctx and "[03:01] second line" in ctx
    assert "outside the chunk" not in ctx
    assert "[08:20] gradient descent" in ctx  # vidB has no transcript -> chunk-level fallback


def test_no_hits_returns_not_found_without_calling_llm():
    result = ans.generate_answer("anything", [])
    assert result["answer"] == ans.NOT_FOUND and result["citations"] == []
