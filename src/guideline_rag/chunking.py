"""Structure-aware chunking.

Instead of cutting text every N tokens, we follow the shape of the guideline:

* one numbered recommendation  -> exactly one chunk (it is a complete clinical statement);
* one table                    -> one chunk (split by rows only if huge, header repeated);
* long background text         -> split at paragraph boundaries with a small overlap.

Every chunk also gets a short context header (guideline | section | number) that is
prepended to the text we *embed*, so the chunk still makes sense when read alone.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

from guideline_rag.config import load_config, project_path
from guideline_rag.ingest import Record

# A table bigger than this many tokens is split by rows (header repeated in each part).
MAX_TABLE_TOKENS = 800


class Chunk(BaseModel):
    chunk_id: str  # stable id, e.g. NG106-1.4.3 or NG115-table-3
    text: str  # the exact passage shown to the user and used for quote checking
    embed_text: str  # context header + text; this is what gets embedded / keyword-indexed
    guideline_code: str
    guideline_title: str
    organization: str
    section_path: str
    recommendation_id: str | None
    last_updated: str
    source_url: str
    content_type: str


def count_tokens(text: str) -> int:
    """Cheap token estimate (about 1.33 tokens per word). Good enough for sizing chunks."""
    return round(len(text.split()) * 4 / 3)


def context_header(code: str, section_path: str, rec_id: str | None) -> str:
    """e.g. 'NG106 | Treating heart failure > Pharmacological treatment | 1.4.3'."""
    return " | ".join(p for p in (code, section_path, rec_id) if p)


def _make_chunk(rec: Record, chunk_id: str, text: str) -> Chunk:
    header = context_header(rec.guideline_code, rec.section_path, rec.recommendation_id)
    return Chunk(
        chunk_id=chunk_id, text=text, embed_text=f"{header}\n{text}",
        guideline_code=rec.guideline_code, guideline_title=rec.guideline_title,
        organization=rec.organization, section_path=rec.section_path,
        recommendation_id=rec.recommendation_id, last_updated=rec.last_updated,
        source_url=rec.source_url, content_type=rec.content_type)


def split_table(text: str, max_tokens: int = MAX_TABLE_TOKENS) -> list[str]:
    """Split a Markdown table (optional caption first) by rows, repeating caption + header row."""
    lines = text.split("\n")
    sep = next((i for i, ln in enumerate(lines) if re.fullmatch(r"\|(\s*:?-+:?\s*\|)+", ln.strip())), None)
    if sep is None or count_tokens(text) <= max_tokens:
        return [text]
    head = "\n".join(lines[: sep + 1])  # caption (if any), header row and separator row
    rows = [ln for ln in lines[sep + 1:] if ln.strip()]
    parts: list[str] = []
    current: list[str] = []
    for row in rows:
        if current and count_tokens(head + "\n" + "\n".join(current + [row])) > max_tokens:
            parts.append(head + "\n" + "\n".join(current))
            current = []
        current.append(row)
    if current:
        parts.append(head + "\n" + "\n".join(current))
    return parts


def _tail(text: str, overlap_tokens: int) -> str:
    """Last ~overlap_tokens of text, starting at a sentence boundary when one is available.

    Why: a little repeated text at the start of the next chunk keeps a sentence that
    straddles the cut from losing its context.
    """
    words = text.split()
    tail = " ".join(words[-max(1, overlap_tokens * 3 // 4):])  # ~1.33 tokens per word
    sentence_start = re.search(r"[.!?]\s+(?=\S)", tail)
    return tail[sentence_start.end():] if sentence_start else tail


def split_long_text(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split at paragraph boundaries; each new part starts with ~overlap_tokens of the previous one."""
    if count_tokens(text) <= max_tokens:
        return [text]
    paragraphs: list[str] = []
    for para in text.split("\n"):
        if count_tokens(para) > max_tokens:  # one giant paragraph: fall back to sentences
            paragraphs += re.split(r"(?<=[.!?])\s+", para)
        elif para.strip():
            paragraphs.append(para)
    parts: list[str] = []
    current: list[str] = []
    for para in paragraphs:
        if current and count_tokens("\n".join(current + [para])) > max_tokens:
            parts.append("\n".join(current))
            tail = _tail("\n".join(current), overlap_tokens)  # carry the end of this part forward
            current = [tail] if tail else []
        current.append(para)
    if current:
        parts.append("\n".join(current))
    return parts


def chunk_record(rec: Record, max_tokens: int, overlap_tokens: int) -> list[Chunk]:
    """Turn one parsed record into one or more chunks according to its content type."""
    if rec.content_type == "recommendation":
        return [_make_chunk(rec, rec.record_id, rec.text)]  # never split
    pieces = split_table(rec.text) if rec.content_type == "table" else split_long_text(
        rec.text, max_tokens, overlap_tokens)
    if len(pieces) == 1:
        return [_make_chunk(rec, rec.record_id, pieces[0])]
    return [_make_chunk(rec, f"{rec.record_id}-part{i}", p) for i, p in enumerate(pieces, 1)]


def build_chunks() -> list[Chunk]:
    """Read data/processed/records.jsonl, chunk everything, write data/processed/chunks.jsonl."""
    cfg = load_config()
    max_tokens = cfg["chunking"]["max_section_tokens"]
    overlap = cfg["chunking"]["overlap_tokens"]
    src = project_path(cfg["paths"]["processed"]) / "records.jsonl"
    records = [Record.model_validate_json(line) for line in src.read_text(encoding="utf-8").splitlines()]
    chunks = [c for r in records for c in chunk_record(r, max_tokens, overlap)]
    out = project_path(cfg["paths"]["chunks"])
    out.write_text("".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")
    lengths = [count_tokens(c.text) for c in chunks]
    print(f"{len(chunks)} chunks written to {out}")
    print(f"average length: {sum(lengths) / len(lengths):.0f} tokens (max {max(lengths)})")
    for kind in ("recommendation", "table", "context"):
        print(f"  {kind}: {sum(c.content_type == kind for c in chunks)}")
    return chunks


def load_chunks() -> list[Chunk]:
    path = project_path(load_config()["paths"]["chunks"])
    return [Chunk.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    build_chunks()
