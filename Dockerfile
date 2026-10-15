# Video RAG – Jump to the Moment
# Builds a CPU-only image that serves the Streamlit UI on port 7860 (Hugging Face Spaces' default).
#
#   docker build -t video-rag .
#   docker run -p 7860:7860 -e GROQ_API_KEY=your_key video-rag      -> http://localhost:7860

FROM python:3.13-slim

# libgl/libglib: needed by OpenCV (scene detection, OCR) at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Deno: the JavaScript runtime yt-dlp uses to solve YouTube's challenge (otherwise HTTP 403).
COPY --from=denoland/deno:bin-2.1.4 /deno /usr/local/bin/deno

# Hugging Face Spaces runs containers as user 1000, so build everything as that user.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1
WORKDIR $HOME/app

# Dependencies first (this layer is cached until requirements.txt changes).
# CPU-only torch: ~200 MB instead of ~2.5 GB of CUDA libraries.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

# Bake the query-time models into the image so the first question isn't a 1 GB download.
# (Whisper and EasyOCR are only needed to index NEW videos; they download on first use.)
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

COPY --chown=user . .

EXPOSE 7860
# Rebuild the search index from demo_data/ (a few seconds), then start the UI.
CMD ["sh", "-c", "python -m app.seed && streamlit run ui/streamlit_app.py --server.port 7860 --server.address 0.0.0.0 --server.headless true"]
