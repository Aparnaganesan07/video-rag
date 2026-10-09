"""Small helpers shared across modules."""

import json
from pathlib import Path


def format_timestamp(seconds: float) -> str:
    """84.2 -> '01:24'; 3725 -> '1:02:05' (hours only when needed)."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def youtube_link(video_id: str, seconds: float) -> str:
    """Deep link that opens the video at a given second."""
    return f"https://www.youtube.com/watch?v={video_id}&t={int(seconds)}s"


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
