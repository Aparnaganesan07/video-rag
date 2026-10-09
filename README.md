# Video RAG – Jump to the Moment

Ask a question about a lecture video, get an answer where every claim cites a timestamp
like **[17:33]**, and click it to play the video from that exact second.

> **Status:** Phase 2 done (cited LLM answers + Streamlit UI + FastAPI).
> Next: slide OCR, hybrid search, reranking (Phase 3), evaluation (Phase 4), deployment (Phase 5).

## How it works

```
INDEXING
YouTube URL ──► yt-dlp (audio only, m4a) ──► faster-whisper (timestamped segments, cached JSON)
            ──► chunker (~45s windows, 10s overlap) ──► bge-small-en-v1.5 embeddings
            ──► ChromaDB (each chunk stores video_id + start/end seconds)

QUERYING
Question ──► bge query embedding ──► top-5 chunks
         ──► prompt with transcript lines, each prefixed by its [mm:ss]
         ──► Groq LLM (gpt-oss-120b) answers, citing [mm:ss] after every sentence
         ──► citations parsed + checked against the retrieved sources (ungrounded ones flagged)
         ──► UI: clicking [mm:ss] calls YouTube IFrame API player.seekTo(seconds)
```

## Setup (Windows, CPU only)

```powershell
cd "C:\Users\aparn\Documents\video rag"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch, much smaller
pip install -r requirements.txt
copy .env.example .env   # then put your Groq key (console.groq.com/keys) in .env
```

Node.js (or Deno) must be installed: yt-dlp uses it to solve YouTube's JS challenge.

## Usage

**Web UI**
```powershell
streamlit run ui/streamlit_app.py
```
Add a YouTube URL in the sidebar, ask a question, click any timestamp.

**CLI**
```powershell
python -m app.cli index "https://www.youtube.com/watch?v=aircAruvnKk"
python -m app.cli ask "why did people stop using sigmoid?"
python -m app.cli search "what does a neuron hold?" -k 5        # retrieval only, no LLM
python -m app.cli list
```

**REST API**
```powershell
uvicorn app.api:app --reload      # docs at http://127.0.0.1:8000/docs
```
`GET /videos`, `POST /index {"source": url}`, `POST /ask {"question": "...", "video_id": null}`

**Tests**
```powershell
pytest
```

## Project layout

```
app/ingest/      downloader, transcriber, chunker, (frames: Phase 3)
app/retrieval/   embedder, vector_store, (hybrid_search, reranker: Phase 3)
app/generation/  answer.py: prompt, Groq call, citation parsing + grounding check
app/pipeline.py  index_video(), search(), ask()
app/cli.py       command-line entry point
app/api.py       FastAPI backend
ui/              Streamlit app (player + clickable answer in one component)
tests/           unit tests (chunker, citations)
eval/            evaluation set + metrics (Phase 4)
data/            downloads, transcripts, vector DB (gitignored)
```

## Key design choices

- **Audio-only download** (`bestaudio[ext=m4a]`): no ffmpeg needed; PyAV inside faster-whisper decodes it.
- **Whisper `base`, int8 on CPU**: good accuracy/speed trade-off; transcripts cached so re-indexing is instant.
- **45s chunks, 10s overlap, cut on segment boundaries**: long enough to hold one idea, short enough
  that a hit points to a precise moment; the overlap keeps answers that straddle a boundary retrievable.
- **bge-small-en-v1.5**: small (384-dim), fast on CPU, strong on retrieval benchmarks; queries get bge's
  instruction prefix, passages don't.
- **Retrieve by chunk, cite by sentence**: chunks are what we search, but the LLM sees each transcript
  line with its own timestamp, so a citation points at the exact sentence, not just a 45s window.
- **Grounding check**: every cited timestamp must fall inside a retrieved source (±5s); otherwise it's
  shown struck-through in red as a likely hallucination.
- **Player + answer in one HTML component**: clicking a timestamp seeks the player in the browser,
  without a Streamlit rerun, so the video doesn't reload. Citations from another video switch videos.
- **Shared pipeline**: CLI, API and UI all call the same `pipeline.ask()`, so there's one code path to test.
