"""Central settings. Everything tunable lives here so experiments are one edit away."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env in the project root (if present)


def _env_flag(name: str, default: bool) -> bool:
    """Read a true/false setting from the environment, e.g. USE_RERANKER=false."""
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


# Hosted demo mode: questions only, over the pre-indexed demo videos (see app/seed.py).
# Set HOSTED_DEMO=true in the host's secrets; Hugging Face Spaces is detected via SPACE_ID.
IS_HOSTED = _env_flag("HOSTED_DEMO", False) or bool(os.getenv("SPACE_ID"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # harmless Windows cache notice

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))  # override to test with a clean folder
AUDIO_DIR = DATA_DIR / "audio"            # downloaded audio files
TRANSCRIPT_DIR = DATA_DIR / "transcripts" # cached Whisper output (JSON)
CHROMA_DIR = DATA_DIR / "chroma"          # persistent vector DB
VIDEO_DIR = DATA_DIR / "video"            # video-only downloads (for slide OCR)
FRAMES_DIR = DATA_DIR / "frames"          # keyframes + cached OCR per video

# --- Transcription (CPU-friendly defaults) ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"  # quantized weights: ~4x faster on CPU, tiny accuracy loss

# --- Slides: keyframes + OCR ---
ENABLE_SLIDES = _env_flag("ENABLE_SLIDES", True)  # read on-screen text when indexing (slower)
VIDEO_MAX_HEIGHT = 720        # 720p is enough for slide text; higher only costs time
SCENE_THRESHOLD = 20.0        # PySceneDetect ContentDetector (brightness + edges): lower = more sensitive
MIN_SCENE_SECONDS = 2.0       # ignore flickers shorter than this
RESAMPLE_SECONDS = 45.0       # also grab a frame this often inside long scenes (slide builds, fades)
MAX_KEYFRAMES = 200           # cap OCR work per video
OCR_MIN_CONFIDENCE = 0.4      # drop low-confidence OCR words (usually noise)
OCR_IGNORE_TOP = 0.05         # ignore text in the top 5% of the frame (OS menu bars, browser tabs)
SAME_FRAME_DIFF = 4.0         # mean pixel difference (0-255) below which a frame = the previous one
SLIDE_DUPLICATE_RATIO = 0.85  # consecutive frames with text this similar = same slide

# --- Chunking ---
CHUNK_SECONDS = 45.0    # target length of one chunk
OVERLAP_SECONDS = 10.0  # how much consecutive chunks share

# --- Embeddings / vector store ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# bge models retrieve better when the *query* (not the passage) gets this prefix
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
COLLECTION_NAME = "video_chunks"

# --- Search ---
TOP_K = 3          # results shown by `cli search`
ANSWER_TOP_K = 5   # chunks given to the LLM as context
USE_HYBRID = _env_flag("USE_HYBRID", True)      # vector + BM25 keyword search, fused with RRF
USE_RERANKER = _env_flag("USE_RERANKER", True)  # rescore candidates with a cross-encoder
CANDIDATES_K = 20        # how many results each retriever returns before fusion/reranking
RRF_K = 60               # Reciprocal Rank Fusion constant (60 is the standard value from the paper)
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
# Cross-encoders cost ~1s per (question, chunk) pair on CPU at 512 tokens, so we rerank
# only the top 10 fused candidates. Phase 4 eval (50 questions) chose the length:
#   320 tokens: Hit@1 0.60, ~5s/query  <- slide text (after the speech) got cut off: slide Qs 0.67
#   512 tokens: Hit@1 0.66, ~8s/query  <- slide Qs back to 0.89
#   320 tokens with slide text first: Hit@1 0.62 (didn't fix it)
RERANK_CANDIDATES = 10
RERANK_MAX_LENGTH = 512

# --- LLM answers (Groq) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_TEMPERATURE = 0.1  # low = sticks to the sources, less creative wording
# gpt-oss "thinks" before answering. low: ~1-8s, occasionally too cautious ("not found");
# medium: better on partial answers but up to ~20s. Revisit with the Phase 4 evaluation.
LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "low")
CITATION_SLACK_SECONDS = 5.0  # a citation counts as grounded if within this of a source chunk


def ensure_dirs() -> None:
    """Create data folders if they don't exist yet."""
    for d in (AUDIO_DIR, TRANSCRIPT_DIR, CHROMA_DIR, VIDEO_DIR, FRAMES_DIR):
        d.mkdir(parents=True, exist_ok=True)
