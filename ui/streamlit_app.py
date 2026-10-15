"""Streamlit UI: ask a question, read a cited answer, click a timestamp to jump the video.

Run from the project root:
    streamlit run ui/streamlit_app.py

How the "jump" works: the YouTube player, the answer and the sources are rendered
together in ONE HTML component. Clicking a timestamp calls the YouTube IFrame API
(player.seekTo) inside that component. It never goes through Streamlit, so the
page doesn't rerun and the player doesn't reload.
"""

import html
import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `import app` works

from app import config, pipeline  # noqa: E402
from app.generation.answer import CITATION_GROUP_RE, TIMESTAMP_RE  # noqa: E402
from app.retrieval import embedder, reranker, vector_store  # noqa: E402
from app.utils import format_timestamp  # noqa: E402

st.set_page_config(page_title="Video RAG – Jump to the Moment", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading the search models (first run only)...")
def warm_up() -> bool:
    # Load the embedder and reranker once per server, so the first question isn't slow.
    embedder.embed_query("warm up")
    if config.USE_RERANKER:
        reranker.rerank("warm up", [{"text": "warm up"}], top_k=1)
    return True


# ---------- HTML building blocks ----------

def answer_to_html(answer: str, citations: list[dict]) -> str:
    """Escape the answer text and turn every [mm:ss] into a clickable link."""
    by_label = {c["label"]: c for c in citations}

    def link(label: str) -> str:
        c = by_label.get(label)
        if not c or not c["video_id"]:
            return html.escape(label)
        cls = "ts" if c["grounded"] else "ts ungrounded"
        tip = "" if c["grounded"] else ' title="Not found in the retrieved sources"'
        return f'<a href="#" class="{cls}" data-v="{c["video_id"]}" data-t="{c["seconds"]}"{tip}>{label}</a>'

    def replace_group(match) -> str:
        inner = TIMESTAMP_RE.sub(lambda m: link(m.group(0)), match.group(1))
        return f"[{inner}]"

    return CITATION_GROUP_RE.sub(replace_group, html.escape(answer))


def _shorten(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "..."


def sources_to_html(sources: list[dict]) -> str:
    items = []
    for s in sources:
        slide = s.get("slide_text") or ""
        slide_html = f'<div class="snippet slide">🖼 On screen: {html.escape(_shorten(slide, 160))}</div>' if slide else ""
        items.append(
            f'<li><a href="#" class="ts" data-v="{s["video_id"]}" data-t="{int(s["start"])}">'
            f'{format_timestamp(s["start"])}–{format_timestamp(s["end"])}</a> '
            f'<span class="muted">· {html.escape(s["title"])} · score {s["score"]:.2f}</span>'
            f'<div class="snippet">{html.escape(_shorten(s["text"], 240))}</div>{slide_html}</li>'
        )
    return f'<ol class="sources">{"".join(items)}</ol>'


def render_player(video_id: str, start: int, answer_html: str = "", sources_html: str = "", dark: bool = False):
    """YouTube player + answer + sources in one component, wired with seekTo."""
    colors = (
        {"bg": "#0e1117", "fg": "#e6e6e6", "muted": "#9aa0a6", "card": "#1a1d24", "link": "#7ab7ff", "border": "#2c313a"}
        if dark else
        {"bg": "#ffffff", "fg": "#1f2328", "muted": "#656d76", "card": "#f6f8fa", "link": "#0b62d6", "border": "#d8dee4"}
    )
    side = (
        f'<div class="card"><h3>Answer</h3><div class="answer">{answer_html}</div></div>'
        f'<div class="card"><h3>Retrieved moments</h3>{sources_html}</div>'
        if answer_html else
        '<div class="card muted">Ask a question above. Timestamps in the answer are clickable.</div>'
    )
    page = f"""
<style>
  body {{ margin:0; font-family: "Source Sans Pro", system-ui, sans-serif; background:{colors["bg"]}; color:{colors["fg"]}; }}
  .grid {{ display:grid; grid-template-columns: 1fr; gap:16px; }}
  @media (min-width: 900px) {{ .grid {{ grid-template-columns: 3fr 2fr; }} }}
  .video {{ position:relative; aspect-ratio:16/9; width:100%; border-radius:10px; overflow:hidden; background:#000; }}
  .video iframe, #player {{ position:absolute; inset:0; width:100%; height:100%; }}
  .card {{ background:{colors["card"]}; border:1px solid {colors["border"]}; border-radius:10px; padding:12px 16px; margin-bottom:12px; }}
  h3 {{ margin:0 0 8px; font-size:15px; text-transform:uppercase; letter-spacing:.04em; color:{colors["muted"]}; }}
  .answer {{ white-space:pre-wrap; line-height:1.55; font-size:16px; }}
  a.ts {{ color:{colors["link"]}; font-weight:600; text-decoration:none; font-variant-numeric:tabular-nums; }}
  a.ts:hover {{ text-decoration:underline; }}
  a.ts.ungrounded {{ color:#d1242f; text-decoration:line-through; }}
  .muted {{ color:{colors["muted"]}; font-size:14px; }}
  .sources {{ margin:0; padding-left:20px; }}
  .sources li {{ margin-bottom:10px; }}
  .snippet {{ font-size:14px; line-height:1.4; color:{colors["muted"]}; margin-top:2px; }}
  .snippet.slide {{ font-style:italic; }}
</style>
<div class="grid">
  <div><div class="video"><div id="player"></div></div></div>
  <div>{side}</div>
</div>
<script>
  let player = null, current = {json.dumps(video_id)};
  function createPlayer() {{
    if (player) return;
    player = new YT.Player("player", {{
      videoId: current,
      playerVars: {{ start: {int(start)}, rel: 0, playsinline: 1 }},
    }});
  }}
  // Register the callback BEFORE loading the API, and cover the case where
  // the API is already loaded (then the callback would never fire).
  window.onYouTubeIframeAPIReady = createPlayer;
  if (window.YT && window.YT.Player) {{
    createPlayer();
  }} else {{
    const tag = document.createElement("script");
    tag.src = "https://www.youtube.com/iframe_api";
    document.head.appendChild(tag);
  }}
  function jump(videoId, seconds) {{
    if (!player) return;
    if (videoId !== current) {{               // citation from another video: switch videos
      player.loadVideoById({{ videoId: videoId, startSeconds: seconds }});
      current = videoId;
    }} else {{
      player.seekTo(seconds, true);           // the core feature: jump to the moment
      player.playVideo();
    }}
  }}
  document.addEventListener("click", (e) => {{
    const a = e.target.closest("a.ts");
    if (!a) return;
    e.preventDefault();
    jump(a.dataset.v, Number(a.dataset.t));
  }});
</script>
"""
    components.html(page, height=640, scrolling=True)


# ---------- Sidebar: add videos, choose scope ----------

videos = [v for v in vector_store.list_videos() if not v["video_id"].startswith("local_")]

with st.sidebar:
    st.header("🎬 Videos")
    if config.IS_HOSTED:
        st.info(
            "Hosted demo: 4 lectures are pre-indexed. Adding new videos may fail here, because "
            "YouTube blocks downloads from cloud servers. Run it locally to index your own."
        )
    with st.form("index_form", clear_on_submit=True):
        url = st.text_input("Add a YouTube video or playlist", placeholder="https://www.youtube.com/watch?v=...")
        use_ocr = st.checkbox(
            "Read on-screen text (slide OCR)",
            value=config.ENABLE_SLIDES,
            help="Finds slide changes and reads their text. Slower: roughly 40s per minute of video on CPU.",
        )
        submitted = st.form_submit_button("Index")
    if submitted and url.strip():
        with st.status("Indexing... (see the terminal for progress)", expanded=True) as status:
            st.write("Download → transcribe → " + ("slide OCR → " if use_ocr else "") + "chunk → embed")
            results = pipeline.index_source(url.strip(), slides=use_ocr)
            ok = [r for r in results if "error" not in r]
            for r in results:
                if "error" in r:
                    st.error(f"{r['source']}: {r['error']}")
                else:
                    st.write(f"✅ {r['title']}: {r['num_chunks']} chunks, {r['num_slides']} slides")
            status.update(label=f"Indexed {len(ok)}/{len(results)} video(s)", state="complete" if ok else "error")
        if ok:
            st.session_state.pop("result", None)
            st.rerun()

    if not videos:
        st.info("No videos yet. Paste a YouTube URL above to get started.")
        st.stop()

    titles = {
        v["video_id"]: f'{"🖼 " if v["has_slides"] else ""}{v["title"]} ({v["chunks"]} chunks)' for v in videos
    }
    scope = st.selectbox(
        "Search in",
        options=[None] + list(titles),
        format_func=lambda vid: "All videos" if vid is None else titles[vid],
    )

    with st.expander("Retrieval settings"):
        use_hybrid = st.toggle("Hybrid search (vector + BM25)", value=config.USE_HYBRID)
        use_rerank = st.toggle("Rerank with cross-encoder", value=config.USE_RERANKER, help="More accurate, ~5s slower on CPU")
    st.caption(
        f"LLM: `{config.LLM_MODEL}` via Groq · Embeddings: `{config.EMBEDDING_MODEL}` · "
        f"Reranker: `{config.RERANKER_MODEL}` · 🖼 = has slide text"
    )


# ---------- Main area: question → answer → player ----------

st.title("Jump to the Moment")
st.caption("Ask a question about a lecture. Every claim in the answer links to the exact second in the video.")

warm_up()

with st.form("ask_form"):
    question = st.text_input("Your question", placeholder="e.g. Why did people stop using sigmoid?")
    asked = st.form_submit_button("Ask", type="primary")

if asked and question.strip():
    if not config.GROQ_API_KEY:
        st.error("GROQ_API_KEY is missing. Copy .env.example to .env and add your key.")
        st.stop()
    with st.spinner("Searching the transcript and writing an answer..."):
        try:
            st.session_state["result"] = pipeline.ask(
                question.strip(), video_id=scope, hybrid=use_hybrid, rerank=use_rerank
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

result = st.session_state.get("result")
dark = getattr(getattr(st.context, "theme", None), "type", None) == "dark"

if result and result["sources"]:
    # Start the player at the first grounded citation (or the best source).
    first = next((c for c in result["citations"] if c["grounded"]), None)
    start_vid = first["video_id"] if first else result["sources"][0]["video_id"]
    start_sec = first["seconds"] if first else int(result["sources"][0]["start"])
    render_player(
        start_vid,
        start_sec,
        answer_to_html(result["answer"], result["citations"]),
        sources_to_html(result["sources"]),
        dark=dark,
    )
    ungrounded = [c["label"] for c in result["citations"] if not c["grounded"]]
    if ungrounded:
        st.warning(f"These citations don't match any retrieved source: {', '.join(ungrounded)}")
else:
    render_player(scope or videos[0]["video_id"], 0, dark=dark)
