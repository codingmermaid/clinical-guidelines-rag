"""FastAPI backend.

Run:  uvicorn guideline_rag.api:app --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import threading

import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from guideline_rag.config import load_config, project_path
from guideline_rag.generation import Answer
from guideline_rag.guardrails import EDUCATIONAL_NOTICE
from guideline_rag.pipeline import ask
from guideline_rag.retrieval import MODES
from guideline_rag.tracing import enable_from_env

app = FastAPI(title="Clinical Guidelines RAG Assistant", version="0.1.0",
              description=EDUCATIONAL_NOTICE)
enable_from_env()  # optional Phoenix tracing when GUIDELINE_TRACING=1
_lock = threading.Lock()  # the local vector store and reranker are used by one request at a time


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    guideline: str | None = Field(None, description="Limit the search to one guideline code, e.g. NG106")
    mode: str | None = Field(None, description=f"Retrieval mode, one of {MODES}; default is from config.yaml")


class AskResponse(Answer):
    notice: str = EDUCATIONAL_NOTICE


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    if req.mode and req.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    with _lock:
        answer = ask(req.question, guideline=req.guideline, mode=req.mode)
    return AskResponse(**answer.model_dump())


@app.get("/health")
def health() -> dict:
    """Confirm Ollama (with both models) and the indexes are available."""
    cfg = load_config()
    status = {"ollama": False, "models": {}, "vector_index": False, "bm25_index": False}
    try:
        names = {m.model for m in ollama.Client(host=cfg["models"]["ollama_url"], timeout=5).list().models}
        status["ollama"] = True
        for key in ("llm", "embedding"):
            wanted = cfg["models"][key]
            status["models"][wanted] = any(n == wanted or n.startswith(wanted + ":") for n in names)
    except Exception as err:
        status["ollama_error"] = type(err).__name__
    try:
        from guideline_rag.index import COLLECTION, get_qdrant
        with _lock:
            status["vector_index"] = get_qdrant().count(COLLECTION).count > 0
    except Exception as err:
        status["vector_error"] = type(err).__name__
    status["bm25_index"] = project_path(cfg["paths"]["bm25"]).exists()
    status["ok"] = (status["ollama"] and all(status["models"].values()) and status["vector_index"]
                    and status["bm25_index"])
    return status
