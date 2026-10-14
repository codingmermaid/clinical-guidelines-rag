"""Build the two search indexes over the same chunks.

* Vector index (Qdrant, local mode): finds passages with a similar *meaning*.
* Keyword index (BM25): finds passages that share *exact words* (drug names, doses,
  abbreviations), which semantic search can miss in medical text.

Rebuild everything with:  python -m guideline_rag.index --rebuild
"""

from __future__ import annotations

import argparse
import atexit
import shutil
import uuid
from functools import lru_cache

from llama_index.core.schema import TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from qdrant_client import QdrantClient, models

from guideline_rag.chunking import Chunk, load_chunks
from guideline_rag.config import load_config, project_path

COLLECTION = "guidelines"
# nomic-embed-text was trained with these task prefixes; using them improves retrieval.
DOC_PREFIX, QUERY_PREFIX = "search_document: ", "search_query: "


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be UUIDs or integers, so derive a stable UUID from the chunk id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


@lru_cache(maxsize=1)
def get_embedder() -> OllamaEmbedding:
    cfg = load_config()["models"]
    return OllamaEmbedding(model_name=cfg["embedding"], base_url=cfg["ollama_url"])


def embed_query(text: str) -> list[float]:
    return get_embedder().get_text_embedding(QUERY_PREFIX + text)


@lru_cache(maxsize=1)
def get_qdrant() -> QdrantClient:
    """Local-mode Qdrant stored in a folder. Only one process can open it at a time."""
    client = QdrantClient(path=str(project_path(load_config()["paths"]["qdrant"])))
    atexit.register(client.close)  # close cleanly so Python does not print errors at shutdown
    return client


def build_vector_index(chunks: list[Chunk]) -> None:
    embedder = get_embedder()
    texts = [DOC_PREFIX + c.embed_text for c in chunks]
    vectors = embedder.get_text_embedding_batch(texts, show_progress=True)
    client = get_qdrant()
    client.create_collection(
        COLLECTION, vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE))
    client.create_payload_index(COLLECTION, "guideline_code", models.PayloadSchemaType.KEYWORD)
    client.upsert(COLLECTION, points=[
        models.PointStruct(id=point_id(c.chunk_id), vector=v, payload=c.model_dump())
        for c, v in zip(chunks, vectors)])


def build_bm25_index(chunks: list[Chunk]) -> None:
    nodes = [TextNode(id_=c.chunk_id, text=c.embed_text, metadata={"chunk_id": c.chunk_id,
                                                                  "guideline_code": c.guideline_code})
             for c in chunks]
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=20)
    retriever.persist(str(project_path(load_config()["paths"]["bm25"])))


def rebuild() -> None:
    cfg = load_config()["paths"]
    get_qdrant.cache_clear()
    for key in ("qdrant", "bm25"):  # start from a clean slate so the build is repeatable
        shutil.rmtree(project_path(cfg[key]), ignore_errors=True)
    chunks = load_chunks()
    print(f"Embedding {len(chunks)} chunks with {load_config()['models']['embedding']} ...")
    build_vector_index(chunks)
    build_bm25_index(chunks)
    print(f"Vector index points: {get_qdrant().count(COLLECTION).count}")
    print("BM25 index saved.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="delete and rebuild both indexes")
    if ap.parse_args().rebuild:
        rebuild()
    else:
        ap.print_help()
