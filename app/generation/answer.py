"""Step 5: turn retrieved chunks into a cited answer with an LLM (Groq).

Key idea: we give the LLM the transcript *line by line with timestamps*
(not just one timestamp per 45s chunk), plus the slide text shown in that
window. So when it cites [03:01] it points at the exact sentence or slide,
not just "somewhere in this chunk".

After generation we check every citation: a timestamp that falls outside
all retrieved sources is flagged as ungrounded (a likely hallucination).
"""

import re

from app import config
from app.ingest.frames import load_slides
from app.ingest.transcriber import load_segments
from app.utils import format_timestamp, parse_timestamp

NOT_FOUND = "I couldn't find this in the video."

SYSTEM_PROMPT = f"""You answer questions about videos using ONLY the transcript sources provided.
Lines marked "(on screen)" are text read from the slides shown at that time (OCR, may contain typos).

Rules:
- Every sentence that states a fact must end with its own citation: the timestamp of the
  transcript line that supports THAT sentence, copied exactly as written, e.g. [03:01].
  Put citations after each sentence, never all together at the end of the answer.
- Multiple citations look like [03:01][04:12]. Never cite a time that is not in the sources.

Example:
Question: What does a neuron hold?
Answer: A neuron holds a single number between 0 and 1 [02:56]. That number is called its activation [03:20].
- If the sources only partly answer the question, give what they do support (with citations)
  and say briefly what the video doesn't cover.
- Only if the sources contain nothing relevant at all, reply exactly: "{NOT_FOUND}"
- Be concise: 2-6 sentences, or a short bulleted list if the question asks for steps.
- Plain text only (bullets with "- " are fine). No headings, no tables."""

# A bracket holding one or more timestamps: [03:01], [1:02:05], [03:01, 03:40], [03:01-03:40]
CITATION_GROUP_RE = re.compile(r"\[(\d{1,2}(?::\d{2}){1,2}(?:\s*[,–-]\s*\d{1,2}(?::\d{2}){1,2})*)\]")
TIMESTAMP_RE = re.compile(r"\d{1,2}(?::\d{2}){1,2}")


def _source_lines(hit: dict) -> list[tuple[float, str]]:
    """Transcript lines + on-screen text (start_seconds, text) inside a retrieved chunk, in time order."""
    lines = [
        (s["start"], s["text"])
        for s in load_segments(hit["video_id"])
        if s["start"] >= hit["start"] - 0.01 and s["end"] <= hit["end"] + 0.01
    ]
    # Fallback (e.g. transcript file missing): the whole chunk under its start time.
    lines = lines or [(hit["start"], hit["text"])]

    for slide in load_slides(hit["video_id"]):
        if slide["start"] < hit["end"] and slide["end"] > hit["start"]:
            # A slide may appear before the chunk starts; clamp so citations stay inside the chunk.
            lines.append((max(slide["start"], hit["start"]), f"(on screen) {slide['text']}"))
    return sorted(lines, key=lambda line: line[0])


def build_context(hits: list[dict]) -> str:
    """Format retrieved chunks as numbered sources with timestamped lines."""
    blocks = []
    for i, hit in enumerate(hits, 1):
        header = (
            f'SOURCE {i} | video: "{hit["title"]}" | '
            f'{format_timestamp(hit["start"])}-{format_timestamp(hit["end"])}'
        )
        lines = [f"[{format_timestamp(t)}] {text}" for t, text in _source_lines(hit)]
        blocks.append(header + "\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{build_context(hits)}\n\nQuestion: {question}"},
    ]


def normalize_citations(answer: str) -> str:
    """gpt-oss sometimes uses its own 【03:01】 brackets; convert them to [03:01]."""
    return answer.replace("【", "[").replace("】", "]")


def extract_citations(answer: str, hits: list[dict]) -> list[dict]:
    """Find every [mm:ss] in the answer and match it to the source it came from.

    Returns one entry per unique timestamp, in order of first appearance:
    {"label": "03:01", "seconds": 181, "video_id", "title", "grounded": bool}
    """
    citations, seen = [], set()
    for group in CITATION_GROUP_RE.finditer(answer):
        for label in TIMESTAMP_RE.findall(group.group(1)):
            if label in seen:
                continue
            seen.add(label)
            seconds = parse_timestamp(label)
            source = _match_source(seconds, hits)
            citations.append({
                "label": label,
                "seconds": seconds,
                "video_id": source["video_id"] if source else (hits[0]["video_id"] if hits else None),
                "title": source["title"] if source else None,
                "grounded": source is not None,
            })
    return citations


def _match_source(seconds: int, hits: list[dict]) -> dict | None:
    """The highest-ranked source whose time range contains `seconds` (with slack)."""
    slack = config.CITATION_SLACK_SECONDS
    for hit in hits:  # hits are in rank order, so the best match wins ties across videos
        if hit["start"] - slack <= seconds <= hit["end"] + slack:
            return hit
    return None


def generate_answer(question: str, hits: list[dict]) -> dict:
    """Ask the LLM. Returns {"answer", "citations", "model"}."""
    if not hits:
        return {"answer": NOT_FOUND, "citations": [], "model": None}
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")

    from groq import Groq  # imported here so search-only use doesn't need it

    client = Groq(api_key=config.GROQ_API_KEY)
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=build_messages(question, hits),
        temperature=config.LLM_TEMPERATURE,
        max_completion_tokens=1024,
        **_reasoning_options(config.LLM_MODEL),
    )
    answer = normalize_citations((response.choices[0].message.content or "").strip()) or NOT_FOUND
    return {"answer": answer, "citations": extract_citations(answer, hits), "model": config.LLM_MODEL}


def _reasoning_options(model: str) -> dict:
    """gpt-oss models 'think' before answering: keep it short and out of the reply."""
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": config.LLM_REASONING_EFFORT, "include_reasoning": False}
    return {}
