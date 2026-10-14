"""Hybrid retrieval: vector search + keyword search, fused, then reranked.

Pipeline for mode "hybrid_rerank":

  question --> vector top-20 ----\
                                   >-- reciprocal rank fusion --> cross-encoder rerank --> top-5
  question --> BM25   top-20 ----/

The mode can be changed in config.yaml (or per call), because the evaluation phase
compares: vector | bm25 | hybrid | hybrid_rerank.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from llama_index.retrievers.bm25 import BM25Retriever
from qdrant_client import models

from guideline_rag.chunking import Chunk, load_chunks
from guideline_rag.config import load_config, project_path
from guideline_rag.index import COLLECTION, embed_query, get_qdrant

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")
RRF_K = 60  # standard constant from the reciprocal-rank-fusion paper; damps the effect of top ranks


@dataclass
class Retrieved:
    """A chunk plus the score that put it in the final list."""

    chunk: Chunk
    score: float
    rank: int


@lru_cache(maxsize=1)
def chunk_lookup() -> dict[str, Chunk]:
    return {c.chunk_id: c for c in load_chunks()}


@lru_cache(maxsize=1)
def _bm25() -> BM25Retriever:
    return BM25Retriever.from_persist_dir(str(project_path(load_config()["paths"]["bm25"])))


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder  # imported lazily: it is slow to import

    return CrossEncoder(load_config()["models"]["reranker"])


def vector_search(query: str, k: int, guideline: str | None = None) -> list[str]:
    """Return chunk_ids ranked by embedding similarity (best first)."""
    flt = None
    if guideline:
        flt = models.Filter(must=[models.FieldCondition(
            key="guideline_code", match=models.MatchValue(value=guideline))])
    hits = get_qdrant().query_points(COLLECTION, query=embed_query(query), limit=k, query_filter=flt).points
    return [h.payload["chunk_id"] for h in hits]


def bm25_search(query: str, k: int, guideline: str | None = None) -> list[str]:
    """Return chunk_ids ranked by BM25 keyword score (best first)."""
    retriever = _bm25()
    lookup = chunk_lookup()
    # With a filter we over-fetch (the whole corpus is small) and filter afterwards.
    retriever.similarity_top_k = len(lookup) if guideline else k
    ids = [n.node.metadata["chunk_id"] for n in retriever.retrieve(query)]
    if guideline:
        ids = [i for i in ids if lookup[i].guideline_code == guideline]
    return ids[:k]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Merge several ranked lists into one.

    Each list gives every chunk a score of 1 / (k + rank). A chunk ranked high in
    *both* lists gets the largest total. We use ranks instead of raw scores because
    vector similarities and BM25 scores are on completely different scales.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def rerank(query: str, chunk_ids: list[str], top_n: int) -> list[tuple[str, float]]:
    """Re-score candidates by reading query and passage *together* (a cross-encoder).

    This is slower than the first-stage searches but much more precise, so we only
    run it on the ~30 fused candidates.
    """
    lookup = chunk_lookup()
    scores = _reranker().predict([(query, lookup[i].embed_text) for i in chunk_ids])
    ranked = sorted(zip(chunk_ids, (float(s) for s in scores)), key=lambda kv: kv[1], reverse=True)
    return ranked[:top_n]


def retrieve(query: str, mode: str | None = None, guideline: str | None = None,
             top_k: int | None = None) -> list[Retrieved]:
    """Find the best passages for a question using the configured (or given) mode."""
    cfg = load_config()["retrieval"]
    mode = mode or cfg["mode"]
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    top_k = top_k or cfg["final_top_k"]
    guideline = guideline.upper() if guideline else None

    if mode == "vector":
        ranked = [(i, 1.0) for i in vector_search(query, cfg["vector_top_k"], guideline)][:top_k]
    elif mode == "bm25":
        ranked = [(i, 1.0) for i in bm25_search(query, cfg["bm25_top_k"], guideline)][:top_k]
    else:
        fused = reciprocal_rank_fusion([vector_search(query, cfg["vector_top_k"], guideline),
                                        bm25_search(query, cfg["bm25_top_k"], guideline)])
        if mode == "hybrid":
            ranked = fused[:top_k]
        else:
            ranked = rerank(query, [i for i, _ in fused], top_k)

    lookup = chunk_lookup()
    return [Retrieved(lookup[i], s, r) for r, (i, s) in enumerate(ranked, start=1)]
