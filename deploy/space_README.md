---
title: Video RAG – Jump to the Moment
emoji: 🎬
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Ask a lecture video a question, click [mm:ss] to jump there
---

# Video RAG – Jump to the Moment

Ask a question about a lecture; every claim in the answer cites a timestamp like **[17:33]**,
and clicking it plays the video from that second.

Pipeline: faster-whisper transcription + slide OCR (PySceneDetect + EasyOCR) → hybrid search
(bge-small vectors + BM25, Reciprocal Rank Fusion) → bge-reranker-base → Groq LLM with
grounded [mm:ss] citations.

This demo comes with 4 lectures pre-indexed. On 50 labelled questions the full pipeline puts the right
moment first 66% of the time (vs 40% for transcript-only vector search), and 92% of answers cite it.

Code, evaluation and design notes: https://github.com/Aparnaganesan07/video-rag
