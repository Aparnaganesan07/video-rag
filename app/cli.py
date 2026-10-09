"""Command-line interface.

Usage:
    python -m app.cli index "https://www.youtube.com/watch?v=aircAruvnKk"
    python -m app.cli search "what is a neuron?"
    python -m app.cli search "what is a neuron?" --video aircAruvnKk -k 5
    python -m app.cli ask "why did they stop using sigmoid?"
    python -m app.cli list
"""

import argparse
import textwrap

from app import config, pipeline
from app.retrieval import vector_store
from app.utils import format_timestamp, youtube_link


def cmd_index(args):
    meta = pipeline.index_video(args.source, force=args.force)
    print(f"\nDone: {meta['num_chunks']} chunks indexed for video id '{meta['video_id']}'.")


def cmd_search(args):
    hits = pipeline.search(args.question, k=args.k, video_id=args.video)
    if not hits:
        print("No results. Did you index a video first?")
        return
    for rank, h in enumerate(hits, 1):
        span = f"[{format_timestamp(h['start'])} - {format_timestamp(h['end'])}]"
        print(f"\n#{rank} {span}  score={h['score']:.3f}  {h['title']}")
        if not h["video_id"].startswith("local_"):
            print(f"   {youtube_link(h['video_id'], h['start'])}")
        print(textwrap.indent(textwrap.fill(h["text"], width=100), "   "))


def cmd_ask(args):
    result = pipeline.ask(args.question, video_id=args.video)
    print("\n" + textwrap.fill(result["answer"], width=100, replace_whitespace=False) + "\n")
    for c in result["citations"]:
        flag = "" if c["grounded"] else "   (!) not found in sources"
        link = youtube_link(c["video_id"], c["seconds"]) if c["video_id"] else ""
        print(f"  [{c['label']}] {link}{flag}")


def cmd_list(_args):
    videos = vector_store.list_videos()
    if not videos:
        print("No videos indexed yet.")
    for v in videos:
        print(f"{v['video_id']:<20} {v['chunks']:>4} chunks   {v['title']}")


def main():
    parser = argparse.ArgumentParser(prog="video-rag", description="Jump to the moment in a video.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="download, transcribe and index a video")
    p.add_argument("source", help="YouTube URL or path to a local video/audio file")
    p.add_argument("--force", action="store_true", help="re-transcribe even if cached")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="find the moments that answer a question")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=config.TOP_K, help="number of results")
    p.add_argument("--video", help="restrict search to one video id")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer a question with [mm:ss] citations (uses Groq)")
    p.add_argument("question")
    p.add_argument("--video", help="restrict to one video id")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("list", help="show indexed videos")
    p.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
