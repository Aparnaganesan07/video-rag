"""REST API around the pipeline.

Run:   uvicorn app.api:app --reload
Docs:  http://127.0.0.1:8000/docs  (interactive Swagger UI)
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import config, pipeline
from app.retrieval import vector_store

app = FastAPI(title="Video RAG – Jump to the Moment")


class IndexRequest(BaseModel):
    source: str  # YouTube URL or a local file path on the server


class AskRequest(BaseModel):
    question: str
    video_id: str | None = None  # None = search all indexed videos
    k: int = config.ANSWER_TOP_K


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/videos")
def videos():
    return vector_store.list_videos()


# Plain `def` (not async): FastAPI runs these in a thread pool,
# so slow model work doesn't block other requests.
@app.post("/index")
def index(req: IndexRequest):
    try:
        return pipeline.index_video(req.source)
    except Exception as e:  # bad URL, private video, no speech, ...
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")
    return pipeline.ask(req.question, video_id=req.video_id, k=req.k)
