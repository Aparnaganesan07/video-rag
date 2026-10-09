"""Step 2b: read on-screen text (slides, code, diagrams) from the video.

1. PySceneDetect finds moments where the picture changes a lot (= slide changes).
2. We grab one keyframe per scene (plus one every RESAMPLE_SECONDS inside long scenes).
3. EasyOCR reads the text in each keyframe.
4. Consecutive frames with near-identical text are merged into one "slide"
   with a [start, end] time range.

Result (cached as JSON): [{"start": 12.0, "end": 47.5, "text": "...", "frame": "path.jpg"}, ...]
"""

import re
import time
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from app import config
from app.utils import load_json, save_json

_reader = None  # EasyOCR reader, loaded lazily


def slides_path(video_id: str) -> Path:
    return config.FRAMES_DIR / video_id / "slides.json"


@lru_cache(maxsize=32)
def load_slides(video_id: str) -> tuple[dict, ...]:
    """Cached OCR slides for an indexed video (empty if slides were never extracted)."""
    path = slides_path(video_id)
    return tuple(load_json(path)) if path.exists() else ()


# ---------- pure helpers (unit-tested) ----------

def clean_ocr_text(pieces: list[tuple[str, float]], min_conf: float = config.OCR_MIN_CONFIDENCE) -> str:
    """Join confident OCR pieces into one line of text; drop noise like stray symbols."""
    words = [t.strip() for t, conf in pieces if conf >= min_conf and len(t.strip()) >= 2]
    text = " ".join(words)
    return re.sub(r"\s+", " ", text).strip()


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def merge_duplicate_slides(frames: list[dict], ratio: float = config.SLIDE_DUPLICATE_RATIO) -> list[dict]:
    """Merge consecutive frames whose text is (nearly) the same into one slide.

    `frames` must be sorted by start time; frames with no text are dropped.
    When text grows (a bullet appears), we keep the longer version.
    """
    slides: list[dict] = []
    for f in frames:
        if not f["text"]:
            continue
        if slides and text_similarity(slides[-1]["text"], f["text"]) >= ratio:
            prev = slides[-1]
            prev["end"] = max(prev["end"], f["end"])
            if len(f["text"]) > len(prev["text"]):
                prev["text"], prev["frame"] = f["text"], f["frame"]
        else:
            slides.append(dict(f))
    return slides


def sample_times(scenes: list[tuple[float, float]], resample: float = config.RESAMPLE_SECONDS) -> list[tuple[float, float, float]]:
    """For each scene pick frame times: (frame_time, piece_start, piece_end).

    Short scenes get one frame (1s in, so we skip transition blur).
    Long scenes are split into pieces of `resample` seconds, one frame each.
    """
    out = []
    for start, end in scenes:
        t = start
        while t < end:
            piece_end = min(t + resample, end)
            out.append((min(t + 1.0, (t + piece_end) / 2), t, piece_end))
            t = piece_end
    return out


# ---------- heavy steps ----------

def detect_scenes(video_path: str) -> list[tuple[float, float]]:
    """(start, end) seconds of each visually distinct scene."""
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    min_len_frames = max(1, int(config.MIN_SCENE_SECONDS * video.frame_rate))
    # Compare brightness + EDGES, not colour: when a slide changes, colours barely move
    # but the text (= edges) does. The default colour-only setting missed most slide changes.
    weights = ContentDetector.Components(delta_hue=0.0, delta_sat=0.0, delta_lum=1.0, delta_edges=1.0)
    manager = SceneManager()
    manager.auto_downscale = True  # analyse a small copy of each frame: much faster
    manager.add_detector(
        ContentDetector(threshold=config.SCENE_THRESHOLD, min_scene_len=min_len_frames, weights=weights)
    )
    manager.detect_scenes(video, frame_skip=1)  # look at every 2nd frame: 2x faster, fine for slides

    scenes = [(s.get_seconds(), e.get_seconds()) for s, e in manager.get_scene_list(start_in_scene=True)]
    return scenes or [(0.0, video.duration.get_seconds())]


def _get_reader():
    global _reader
    if _reader is None:
        import warnings

        import easyocr  # heavy import (torch)

        warnings.filterwarnings("ignore", category=UserWarning, module="torch")  # harmless CPU notices
        print("[slides] loading EasyOCR (CPU)")
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def ocr_frame(image) -> str:
    """Read text from one frame, ignoring the top band (menu bars, browser tabs)."""
    top_limit = image.shape[0] * config.OCR_IGNORE_TOP
    pieces = [
        (text, conf)
        for box, text, conf in _get_reader().readtext(image)
        if (box[0][1] + box[2][1]) / 2 > top_limit  # vertical centre of the text box
    ]
    return clean_ocr_text(pieces)


def _thumbnail(image):
    """Tiny grayscale copy, used to spot frames that are identical to the previous one."""
    import cv2

    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (64, 36)).astype("float32")


def extract_slides(video_path: str, video_id: str, force: bool = False) -> list[dict]:
    """Detect scenes, OCR one keyframe per scene piece, merge duplicates. Cached."""
    import cv2

    cache = slides_path(video_id)
    if cache.exists() and not force:
        print(f"[slides] cached: {cache.parent.name}/{cache.name}")
        return load_json(cache)

    t0 = time.time()
    scenes = detect_scenes(video_path)
    samples = sample_times(scenes)
    if len(samples) > config.MAX_KEYFRAMES:  # too many: keep an even spread
        step = len(samples) / config.MAX_KEYFRAMES
        samples = [samples[int(i * step)] for i in range(config.MAX_KEYFRAMES)]
    print(f"[slides] {len(scenes)} scenes -> {len(samples)} keyframes ({time.time() - t0:.0f}s)")

    out_dir = cache.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frames, prev_thumb, prev_text, ocr_runs = [], None, "", 0
    for i, (t, start, end) in enumerate(samples, 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, image = cap.read()
        if not ok:
            continue
        thumb = _thumbnail(image)
        if prev_thumb is not None and abs(thumb - prev_thumb).mean() < config.SAME_FRAME_DIFF:
            text = prev_text  # same picture as last time: skip the slow OCR (~6s/frame on CPU)
        else:
            text, ocr_runs = ocr_frame(image), ocr_runs + 1
        prev_thumb, prev_text = thumb, text

        frame_file = out_dir / f"{t:08.1f}.jpg"
        cv2.imwrite(str(frame_file), image)
        frames.append({"start": round(start, 2), "end": round(end, 2), "text": text, "frame": str(frame_file)})
        print(f"\r[slides] keyframe {i}/{len(samples)} (OCR ran {ocr_runs}x)", end="", flush=True)
    cap.release()

    slides = merge_duplicate_slides(frames)
    print(f"\n[slides] {len(slides)} distinct slides with text in {time.time() - t0:.0f}s")
    save_json(slides, cache)
    load_slides.cache_clear()
    return slides
