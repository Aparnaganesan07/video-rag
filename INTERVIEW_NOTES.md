# Interview Notes – Video RAG "Jump to the Moment"

My revision sheet for this project: what it does, why each piece exists, the numbers,
and the questions I'm likely to get. Live demo and code: see [README.md](README.md).

---

## 1. The 30-second pitch

> "I built a RAG system for lecture videos. You ask a question, it answers with timestamps
> like [17:33] after every claim, and clicking one jumps the video to that second.
> It transcribes speech with Whisper, reads slide text with OCR, retrieves with hybrid
> search (vectors + BM25) plus a cross-encoder reranker, and the LLM must cite timestamps
> that I verify against the sources. I evaluated it on 50 hand-labelled questions:
> each component was added only when it measurably helped. Top-1 accuracy went from 40% to 66%,
> and 92% of answers cite the right moment, with zero made-up timestamps. It's deployed
> on Streamlit Cloud."

What's NEW compared to my document-RAG project: speech (Whisper), vision (scene detection +
OCR), hybrid retrieval, reranking, and a real evaluation with ablations.

---

## 2. How it works, step by step (in plain words)

**Indexing (once per video)**
1. **Download** with yt-dlp, *twice*: audio only (for Whisper) and video only ≤720p (for OCR).
   Two separate streams = never need ffmpeg to merge them.
2. **Transcribe** with faster-whisper (`base` model, int8 on CPU). Output: sentences with
   start/end times. Cached as JSON because it's the slowest step.
3. **Slides**: PySceneDetect finds when the picture changes (= new slide), I grab one frame
   per scene (+ one every 45s inside long scenes), EasyOCR reads the text, and near-identical
   consecutive frames are merged into one "slide" with a time range.
4. **Chunk**: group Whisper sentences into ~45s windows with 10s overlap, cutting only between
   sentences. Attach the slide text visible during each window.
5. **Embed** each chunk (speech + slide text) with bge-small-en-v1.5, store in ChromaDB with
   `video_id`, `start`, `end` as metadata.

**Answering (per question)**
1. **Vector search** top 20 (meaning) + **BM25** top 20 (exact words).
2. **Reciprocal Rank Fusion** merges the two lists by rank.
3. **Cross-encoder reranker** (bge-reranker-base) rescores the top 10, keep the best 5.
4. **LLM** (Groq, gpt-oss-120b) gets each transcript line and each slide *with its own
   [mm:ss]* and must cite after every sentence.
5. **Grounding check**: every cited time must fall inside a retrieved chunk (±5s), otherwise
   it's flagged red as a likely hallucination.
6. **UI**: clicking [mm:ss] calls the YouTube IFrame API `player.seekTo(seconds)` in the browser,
   with no page reload.

---

## 3. Key numbers to remember

| | |
|---|---|
| Eval set | 50 questions, 4 lectures, 104 chunks; types: 26 paraphrase, 15 keyword, 9 slide-only |
| "Correct" means | right video AND true moment inside the chunk ±20s |
| Hit@1 progression | 0.40 → 0.48 (+slides) → 0.54 (+hybrid) → **0.66** (+reranker) |
| MRR | 0.61 → **0.79** |
| Hit@5 (hybrid) | 1.00: the right moment is always in the top 5 |
| Slide-only questions | 0.33 → 0.89 |
| End to end | 92% of answers cite within ±20s; 0 hallucinated timestamps; 2% "not found" |
| Latency (CPU) | ~0.04s retrieval without reranker; ~8–10s with reranker |
| Memory | ~0.7 GB without reranker, ~1.9 GB with (free host limit 2.7 GB) |

---

## 4. "Why did you choose X?" — short answers

**Why 45-second chunks with 10s overlap?**
Long enough to hold one complete idea (so the embedding means something), short enough that a hit
points to a precise moment. Overlap so an explanation that crosses a boundary is whole in at least
one chunk. I cut only between Whisper sentences so no sentence is split.

**Why time-based chunks instead of token-based?**
Every chunk keeps an exact [start, end]: a search hit maps straight to a moment in the video,
which is the whole product.

**Why bge-small-en-v1.5?**
Small (384 dims, ~130 MB), fast on CPU, strong on retrieval benchmarks for its size. Queries get
bge's instruction prefix; passages don't (that's how the model was trained). Vectors are normalised,
so cosine = dot product.

**Why hybrid search (BM25 + vectors)?**
Vectors understand paraphrases but blur rare exact terms ("ReLU", "Zipf"). BM25 is the opposite.
Evidence: slide-only questions went 0.56 → 0.89 when BM25 was added.

**Why Reciprocal Rank Fusion?**
Cosine similarity (0–1) and BM25 scores (0–20+) are on different scales. RRF uses only ranks:
`score = Σ 1/(60 + rank)`. No normalisation, no tuning. 60 is the standard constant from the
original paper.

**Bi-encoder vs cross-encoder?**
Bi-encoder (embedder): question and chunk embedded separately. Fast, precomputable, but they never
"see" each other. Cross-encoder (reranker): reads question + chunk together → more accurate, but
must run per pair at query time. So: bi-encoder to get candidates, cross-encoder to order the top few.
It gave the biggest Hit@1 jump (+12 points).

**Why OCR the slides?**
Whisper mishears jargon: "Heaps' law" → "keep slow", "Zipf's law" → "zip slot",
"RAG-Fusion" → "rag views". The slides spell them correctly.

**Why detect scenes on brightness + edges instead of colour?**
When a slide changes, the colours barely change but the text (= edges) does. Colour-only
detection found 2 scenes in a lecture; edges found 20.

**Why make the LLM cite per transcript line?**
Retrieval works on 45s chunks, but the LLM sees each sentence with its own timestamp, so a citation
points to the exact sentence, not "somewhere in this 45 seconds".

**How do you prevent hallucinated timestamps?**
Prompt rule (cite only times in the sources) + a code check after generation: each cited time must
fall inside a retrieved chunk. Measured result: 0 ungrounded citations on 50 questions.

**Why Groq / gpt-oss-120b?**
Fast and free tier; Llama models were no longer available on my key. The model is a config
setting (`LLM_MODEL`).

---

## 5. The best story: "an evaluation caught my own mistake"

To make the reranker faster I cut its input to 320 tokens. Overall Hit@1 still went up, but the
**per-question-type** table showed slide questions **dropping** (0.89 → 0.67). Reason: I put slide
text *after* the speech, so truncation cut it off. With 512 tokens slide questions recovered and
Hit@1 rose 0.60 → 0.66. I also tested "slides first at 320 tokens": 0.62, so it didn't fix it.
**Lesson:** look at breakdowns, not just the average; one number hides regressions.

---

## 6. Weaknesses (say them before they ask)

- **Small eval set**: 50 questions, so one question = 2 points of Hit@1. Treat differences of a
  few points as noise.
- **I wrote the questions** by reading the transcripts and slides, which may favour their wording.
  Mitigated with 26 paraphrased questions; ideally use questions from real users.
- **Recaps across videos**: chapter 2 recaps chapter 1, so search sometimes returns the recap and
  it's scored "wrong", though it's arguably a fine answer.
- **Reranker is slow on CPU** (~8–10s). On a GPU it'd be well under a second; or use a smaller
  cross-encoder (MiniLM was ~7x faster in my test with a slightly different top-3).
- **The LLM is sometimes over-cautious** ("not found" for RAG-Fusion although a slide names it).
- **OCR on screen recordings is noisy** (browser menus, notebook UI).
- **Hosted demo can't index new YouTube videos**: YouTube blocks cloud IPs, so it ships 4
  pre-indexed lectures (~340 KB of JSON) and rebuilds the index at startup.

---

## 7. "What would you do next?"

- **Bigger, real eval set**, plus an LLM-as-judge for answer quality (not just citations).
- **Faster reranking**: GPU, ONNX/quantised model, or a distilled cross-encoder.
- **Smarter chunking**: split on slide changes / topic shifts instead of fixed 45s windows.
- **De-duplicate recaps** across videos (detect near-identical chunks and prefer the original).
- **Stretch features**: auto chapters, "explain this moment", quiz mode with timestamp links,
  speaker diarization.
- **Better hosting**: a GPU host, or a background job queue so indexing doesn't block the UI.

---

## 8. Likely questions — quick answers

- **What is RAG?** Retrieve relevant text first, then let the LLM answer *from* it, so answers are
  grounded and up to date instead of from the model's memory.
- **What is Hit@k?** Share of questions where a correct result is in the top k.
- **What is MRR?** Average of 1/rank of the first correct result: rewards putting it *first*.
- **Why ±20s tolerance?** A chunk is 45s; landing within 20s means clicking gets you to the
  explanation almost immediately.
- **How did you make the comparison fair?** Each setup is built in memory from the same chunks and
  differs in exactly one component (ablation).
- **What does int8 mean for Whisper?** Weights stored as 8-bit integers: ~4x faster on CPU,
  tiny accuracy loss.
- **Why ChromaDB?** Simple, local, persistent, supports metadata filters (e.g. one video only).
- **What breaks with ffmpeg missing?** Nothing: audio-only m4a is decoded by PyAV inside
  faster-whisper; video-only mp4 is read by OpenCV.
- **How does clicking a timestamp work?** The player and the answer live in one HTML iframe;
  a click calls `player.seekTo(t)` (or `loadVideoById` if the citation is from another video).
  No server round trip, so the video doesn't reload.
- **How did you test it?** 36 unit tests (chunker, citation parsing, OCR helpers, RRF, BM25
  tokenizer, eval metrics) + the 50-question evaluation + manual end-to-end checks.

---

## 9. Files to know

| File | What it does |
|---|---|
| `app/ingest/chunker.py` | time-window chunking + attaching slide text |
| `app/ingest/frames.py` | scene detection + OCR + slide merging |
| `app/retrieval/hybrid_search.py` | BM25 + RRF |
| `app/retrieval/reranker.py` | cross-encoder |
| `app/generation/answer.py` | prompt, Groq call, citation parsing, grounding check |
| `app/pipeline.py` | `index_video()`, `search()`, `ask()`: the whole flow in ~100 lines |
| `eval/evaluate.py` | the ablation + metrics |
| `ui/streamlit_app.py` | UI + YouTube seekTo |
