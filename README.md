# Video RAG – Jump to the Moment

Ask a question about a lecture video and jump straight to the second where it's answered.

> **Status:** Phase 1 (core pipeline + CLI search). LLM answers, UI, slide OCR, hybrid search,
> reranking and evaluation are coming in later phases.

## How it works (Phase 1)

```
YouTube URL ──► yt-dlp (audio only, m4a) ──► faster-whisper (timestamped segments, cached JSON)
            ──► chunker (~45s windows, 10s overlap) ──► bge-small-en-v1.5 embeddings
            ──► ChromaDB (each chunk stores start/end seconds)

Question ──► bge query embedding ──► top-k nearest chunks ──► [mm:ss] + YouTube deep link
```

## Setup (Windows, CPU only)

```powershell
cd "C:\Users\aparn\Documents\video rag"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only torch, much smaller
pip install -r requirements.txt
copy .env.example .env   # add GROQ_API_KEY (needed from Phase 2)
```

Node.js (or Deno) must be installed: yt-dlp uses it to solve YouTube's JS challenge.

## Usage

```powershell
python -m app.cli index "https://www.youtube.com/watch?v=aircAruvnKk"
python -m app.cli search "what does a neuron hold?"
python -m app.cli search "what is a weight?" --video aircAruvnKk -k 5
python -m app.cli list
pytest
```

## Project layout

```
app/ingest/      downloader, transcriber, chunker, (frames: Phase 3)
app/retrieval/   embedder, vector_store, (hybrid_search, reranker: Phase 3)
app/generation/  LLM answers with citations (Phase 2)
app/pipeline.py  index_video() and search()
app/cli.py       command-line entry point
tests/           unit tests (chunker)
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
