"""Step 1: get media for a video (YouTube URL or a local file).

- Audio only (bestaudio[ext=m4a]) for transcription: faster-whisper decodes it via PyAV.
- Video only (no audio track, <=720p) for slide OCR: OpenCV reads it directly.
Downloading the two streams separately means we never need ffmpeg to merge them.
"""

import re
from pathlib import Path

import yt_dlp

from app import config

BASE_OPTS = {
    # YouTube now requires solving a JS challenge; without a runtime you get HTTP 403.
    "js_runtimes": {"deno": {}, "node": {}},
    "noplaylist": True,
    "quiet": True,
    "no_warnings": True,
    "noprogress": True,
}


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _local_video_id(path: Path) -> str:
    """Make a safe, stable id from a local filename, e.g. 'My Lecture.mp4' -> 'local_my_lecture'."""
    slug = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
    return f"local_{slug}"


def is_playlist_url(source: str) -> bool:
    """Only pure playlist links count; 'watch?v=X&list=Y' is treated as the single video X."""
    return _is_url(source) and "list=" in source and "/playlist" in source


def expand_source(source: str) -> list[str]:
    """A playlist URL -> one URL per video. Anything else -> [source]."""
    if not is_playlist_url(source):
        return [source]
    opts = {**BASE_OPTS, "noplaylist": False, "extract_flat": "in_playlist"}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(source, download=False)
    urls = [f"https://www.youtube.com/watch?v={e['id']}" for e in info.get("entries") or [] if e and e.get("id")]
    print(f"[download] playlist '{info.get('title')}': {len(urls)} videos")
    return urls


def _download(source: str, fmt: str, out_dir: Path, what: str) -> tuple[dict, Path]:
    """Download one stream with yt-dlp unless it's already on disk."""
    opts = {**BASE_OPTS, "format": fmt, "outtmpl": str(out_dir / "%(id)s.%(ext)s")}
    with yt_dlp.YoutubeDL(opts) as ydl:
        # First fetch metadata only, so we can skip the download if we already have the file.
        info = ydl.extract_info(source, download=False)
        path = Path(ydl.prepare_filename(info))
        if path.exists():
            print(f"[download] cached {what}: {path.name}")
        else:
            print(f"[download] downloading {what} for '{info['title']}' ...")
            ydl.process_info(info)
    return info, path


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

    info, audio_path = _download(source, "bestaudio[ext=m4a]/bestaudio", config.AUDIO_DIR, "audio")
    return {
        "video_id": info["id"],
        "title": info.get("title", info["id"]),
        "source": source,
        "audio_path": str(audio_path),
        "duration": info.get("duration"),
        "is_youtube": True,
    }


def download_video(meta: dict) -> str:
    """Path to a video file for keyframes: the local file itself, or a <=720p video-only download."""
    if not meta["is_youtube"]:
        return meta["source"]
    h = config.VIDEO_MAX_HEIGHT
    # mp4 (h264) first: OpenCV decodes it everywhere; fall back to any codec at <=720p.
    fmt = f"bestvideo[height<={h}][ext=mp4][vcodec^=avc1]/bestvideo[height<={h}][ext=mp4]/bestvideo[height<={h}]"
    _info, path = _download(meta["source"], fmt, config.VIDEO_DIR, "video")
    return str(path)
