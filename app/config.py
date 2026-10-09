"""Central settings. Everything tunable lives here so experiments are one edit away."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root (if present)

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"            # downloaded audio files
TRANSCRIPT_DIR = DATA_DIR / "transcripts" # cached Whisper output (JSON)
CHROMA_DIR = DATA_DIR / "chroma"          # persistent vector DB

# --- Transcription (CPU-friendly defaults) ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"  # quantized weights: ~4x faster on CPU, tiny accuracy loss

# --- Chunking ---
CHUNK_SECONDS = 45.0    # target length of one chunk
OVERLAP_SECONDS = 10.0  # how much consecutive chunks share

# --- Embeddings / vector store ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# bge models retrieve better when the *query* (not the passage) gets this prefix
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
COLLECTION_NAME = "video_chunks"

# --- Search ---
TOP_K = 3

# --- LLM (used from Phase 2) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def ensure_dirs() -> None:
    """Create data folders if they don't exist yet."""
    for d in (AUDIO_DIR, TRANSCRIPT_DIR, CHROMA_DIR):
        d.mkdir(parents=True, exist_ok=True)
