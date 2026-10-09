"""Step 1: get audio for a video (YouTube URL or a local file).

We download audio only (bestaudio[ext=m4a]) so no ffmpeg is needed:
faster-whisper decodes m4a directly through PyAV.
"""

import re
from pathlib import Path

import yt_dlp

from app import config


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _local_video_id(path: Path) -> str:
    """Make a safe, stable id from a local filename, e.g. 'My Lecture.mp4' -> 'local_my_lecture'."""
    slug = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
    return f"local_{slug}"


def download_audio(source: str) -> dict:
    """Return video metadata plus the path to a local audio file.

    Keys: video_id, title, source, audio_path, duration, is_youtube
    """
    config.ensure_dirs()

    # Local upload: nothing to download, PyAV can read the audio track of mp4/mkv/etc.
    if not _is_url(source):
        path = Path(source).resolve()
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")
        return {
            "video_id": _local_video_id(path),
            "title": path.stem,
            "source": str(path),
            "audio_path": str(path),
            "duration": None,
            "is_youtube": False,
        }

    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio",  # m4a first: no ffmpeg conversion needed
        "outtmpl": str(config.AUDIO_DIR / "%(id)s.%(ext)s"),
        # YouTube now requires solving a JS challenge; without a runtime you get HTTP 403.
        "js_runtimes": {"deno": {}, "node": {}},
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # First fetch metadata only, so we can skip the download if we already have the file.
        info = ydl.extract_info(source, download=False)
        audio_path = Path(ydl.prepare_filename(info))
        if audio_path.exists():
            print(f"[download] cached: {audio_path.name}")
        else:
            print(f"[download] downloading audio for '{info['title']}' ...")
            ydl.process_info(info)

    return {
        "video_id": info["id"],
        "title": info.get("title", info["id"]),
        "source": source,
        "audio_path": str(audio_path),
        "duration": info.get("duration"),
        "is_youtube": True,
    }
