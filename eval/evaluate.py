"""Phase 4: measure retrieval quality, one improvement at a time.

Run:
    python -m eval.evaluate              # retrieval metrics for 4 setups (no LLM calls)
    python -m eval.evaluate --answers    # also check the LLM's citations for the final setup

Ground truth: eval/test_questions.json, one entry per question:
    {"id": "nn-01", "video_id": "aircAruvnKk", "question": "...", "timestamps": ["02:56"], "type": "paraphrase"}
A retrieved chunk is CORRECT if it's from the right video and a ground-truth timestamp lies
inside the chunk's [start, end] range, with TOLERANCE seconds of slack on each side.

Metrics:
    Hit@k  share of questions with at least one correct chunk in the top k
    MRR    mean of 1/rank of the first correct chunk (0 if none in the top 10).
           Rewards putting the right moment FIRST, not just somewhere in the list.

Setups compared (each adds one thing to the previous):
    1. transcript-only   vector search over speech only
    2. + slides          vector search over speech + OCR slide text
    3. + hybrid          vector + BM25, fused with Reciprocal Rank Fusion
    4. + reranker        cross-encoder reorders the top candidates

Every setup is built IN MEMORY from data/transcripts/<id>_chunks.json, so they all see
exactly the same chunks and differ only in the one thing being tested.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from app import config
from app.generation.answer import generate_answer
from app.ingest.chunker import searchable_text
from app.retrieval import embedder, reranker, vector_store
from app.retrieval.hybrid_search import rrf_fuse, tokenize
from app.utils import load_json, parse_timestamp

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS_FILE = EVAL_DIR / "test_questions.json"
TOLERANCE = 20.0  # seconds
MAX_RANK = 10     # MRR looks at the top 10

SETUPS = [
    # name,               slides, hybrid, rerank
    ("transcript-only",   False,  False,  False),
    ("+ slides",          True,   False,  False),
    ("+ hybrid (BM25)",   True,   True,   False),
    ("+ reranker",        True,   True,   True),
]


# ---------- in-memory index (one per text variant) ----------

class MemoryIndex:
    """Vector + BM25 search over a fixed list of chunks, without touching ChromaDB."""

    def __init__(self, chunks: list[dict], use_slides: bool):
        # Without slides, blank out slide_text so embedding, BM25 AND the reranker all ignore it.
        self.chunks = [{**c, "slide_text": c.get("slide_text", "") if use_slides else ""} for c in chunks]
        texts = [searchable_text(c) for c in self.chunks]
        self.vectors = np.array(embedder.embed_passages(texts))  # normalised -> dot = cosine
        self.bm25 = BM25Okapi([tokenize(t) or ["_empty_"] for t in texts])

    def vector_search(self, question: str, k: int) -> list[dict]:
        scores = self.vectors @ np.array(embedder.embed_query(question))
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def bm25_search(self, question: str, k: int) -> list[dict]:
        tokens = tokenize(question)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = [i for i in np.argsort(-scores)[:k] if scores[i] > 0]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def search(self, question: str, hybrid: bool, rerank: bool, k: int = MAX_RANK) -> list[dict]:
        """Same two-stage logic as app.pipeline.search, on this in-memory index."""
        n = max(k, config.CANDIDATES_K)
        if hybrid:
            candidates = rrf_fuse([self.vector_search(question, n), self.bm25_search(question, n)])
        else:
            candidates = self.vector_search(question, n)
        if rerank:
            return reranker.rerank(question, candidates[: config.RERANK_CANDIDATES], top_k=k)
        return candidates[:k]


# ---------- scoring ----------

def is_correct(hit: dict, q: dict) -> bool:
    if hit["video_id"] != q["video_id"]:
        return False
    return any(hit["start"] - TOLERANCE <= t <= hit["end"] + TOLERANCE for t in q["seconds"])


def first_correct_rank(hits: list[dict], q: dict) -> int | None:
    """1-based rank of the first correct hit, or None."""
    return next((rank for rank, h in enumerate(hits, 1) if is_correct(h, q)), None)


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks)
    hit = lambda k: sum(1 for r in ranks if r is not None and r <= k) / n  # noqa: E731
    mrr = sum(1 / r for r in ranks if r is not None) / n
    return {"hit@1": hit(1), "hit@3": hit(3), "hit@5": hit(5), "mrr": mrr, "n": n}


# ---------- data loading ----------

def load_questions(path: Path) -> list[dict]:
    questions = load_json(path)
    for q in questions:
        q["seconds"] = [parse_timestamp(t) for t in q["timestamps"]]
    return questions


def load_chunks(video_ids: set[str]) -> list[dict]:
    chunks = []
    for vid in sorted(video_ids):
        path = config.TRANSCRIPT_DIR / f"{vid}_chunks.json"
        if not path.exists():
            raise SystemExit(f"Missing {path}. Index the video first: python -m app.cli index <url>")
        chunks.extend(load_json(path))
    return chunks


# ---------- main ----------

def run_retrieval(questions: list[dict], chunks: list[dict]) -> dict:
    results, indexes = {}, {}
    for name, slides, hybrid, rerank in SETUPS:
        if slides not in indexes:
            print(f"[eval] embedding {len(chunks)} chunks ({'speech + slides' if slides else 'speech only'})")
            indexes[slides] = MemoryIndex(chunks, use_slides=slides)
        index = indexes[slides]

        ranks, per_question, t0 = [], [], time.time()
        for q in questions:
            hits = index.search(q["question"], hybrid=hybrid, rerank=rerank)
            rank = first_correct_rank(hits, q)
            ranks.append(rank)
            per_question.append({"id": q["id"], "rank": rank})
        latency = (time.time() - t0) / len(questions)

        by_type = {
            qtype: summarize([r for r, q in zip(ranks, questions) if q["type"] == qtype])
            for qtype in sorted({q["type"] for q in questions})
        }
        results[name] = {**summarize(ranks), "latency_s": latency, "by_type": by_type, "per_question": per_question}
        r = results[name]
        print(f"[eval] {name:18s} Hit@1 {r['hit@1']:.2f}  Hit@3 {r['hit@3']:.2f}  MRR {r['mrr']:.3f}  ({latency:.2f}s/q)")
    return results


def run_answers(questions: list[dict], chunks: list[dict]) -> dict:
    """Final setup end to end: does the LLM cite the right moment?"""
    titles = {v["video_id"]: v["title"] for v in vector_store.list_videos()}
    index = MemoryIndex(chunks, use_slides=True)
    cited_right = first_right = not_found = ungrounded = 0
    rows = []
    for i, q in enumerate(questions, 1):
        hits = [{**h, "title": titles.get(h["video_id"], h["video_id"])}
                for h in index.search(q["question"], hybrid=True, rerank=True, k=config.ANSWER_TOP_K)]
        result = generate_answer(q["question"], hits)
        cites = result["citations"]

        def close(c):
            return c["video_id"] == q["video_id"] and any(abs(c["seconds"] - t) <= TOLERANCE for t in q["seconds"])

        any_ok = any(close(c) for c in cites)
        cited_right += any_ok
        first_right += bool(cites) and close(cites[0])
        not_found += not cites
        ungrounded += sum(1 for c in cites if not c["grounded"])
        rows.append({"id": q["id"], "any_citation_correct": any_ok, "citations": [c["label"] for c in cites]})
        print(f"\r[eval] answers {i}/{len(questions)}", end="", flush=True)
        time.sleep(1.0)  # stay well under Groq's free-tier rate limit
    print()
    n = len(questions)
    return {
        "any_citation_within_tol": cited_right / n,
        "first_citation_within_tol": first_right / n,
        "answered_not_found": not_found / n,
        "ungrounded_citations": ungrounded,
        "model": config.LLM_MODEL,
        "reasoning_effort": config.LLM_REASONING_EFFORT,
        "per_question": rows,
    }


def to_markdown(results: dict, answers: dict | None, n_questions: int, n_chunks: int, n_videos: int) -> str:
    lines = [
        f"Evaluation: {n_questions} questions over {n_videos} videos ({n_chunks} chunks). "
        f"A hit = right video and ground-truth time inside the chunk ±{TOLERANCE:.0f}s. Search spans all videos.",
        "",
        "| Setup | Hit@1 | Hit@3 | Hit@5 | MRR | Latency/query (CPU) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | {r['hit@5']:.2f} | {r['mrr']:.3f} | {r['latency_s']:.2f}s |"
        )
    types = list(next(iter(results.values()))["by_type"])
    lines += ["", "Hit@1 by question type:", "", "| Setup | " + " | ".join(f"{t} (n={results[next(iter(results))]['by_type'][t]['n']})" for t in types) + " |",
              "|---|" + "---|" * len(types)]
    for name, r in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{r['by_type'][t]['hit@1']:.2f}" for t in types) + " |")
    if answers:
        lines += [
            "",
            f"End-to-end answers ({answers['model']}, reasoning effort {answers['reasoning_effort']}, final setup):",
            "",
            f"- At least one citation within ±{TOLERANCE:.0f}s of the ground truth: **{answers['any_citation_within_tol']:.0%}**",
            f"- First citation within ±{TOLERANCE:.0f}s: {answers['first_citation_within_tol']:.0%}",
            f"- Answered \"not found\": {answers['answered_not_found']:.0%}",
            f"- Ungrounded (hallucinated) citations: {answers['ungrounded_citations']}",
        ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval (and optionally answers).")
    parser.add_argument("--questions", default=str(QUESTIONS_FILE))
    parser.add_argument("--answers", action="store_true", help="also run the LLM and score its citations")
    args = parser.parse_args()

    questions = load_questions(Path(args.questions))
    video_ids = {q["video_id"] for q in questions}
    chunks = load_chunks(video_ids)
    print(f"[eval] {len(questions)} questions, {len(video_ids)} videos, {len(chunks)} chunks")

    results = run_retrieval(questions, chunks)
    answers = run_answers(questions, chunks) if args.answers else None

    md = to_markdown(results, answers, len(questions), len(chunks), len(video_ids))
    (EVAL_DIR / "results.md").write_text(md, encoding="utf-8")
    (EVAL_DIR / "results.json").write_text(
        json.dumps({"retrieval": results, "answers": answers}, indent=2), encoding="utf-8"
    )
    print("\n" + md)
    print(f"Saved {EVAL_DIR / 'results.md'} and results.json")


if __name__ == "__main__":
    main()
