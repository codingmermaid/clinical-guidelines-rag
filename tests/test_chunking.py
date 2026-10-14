"""Tests for structure-aware chunking (synthetic cases plus the real parsed data)."""

import pytest

from guideline_rag.chunking import (Chunk, build_chunks, chunk_record, count_tokens, split_long_text,
                                    split_table)
from guideline_rag.config import load_config, project_path
from guideline_rag.ingest import Record

REQUIRED = ["chunk_id", "text", "embed_text", "guideline_code", "guideline_title", "organization",
            "section_path", "last_updated", "source_url", "content_type"]


def rec(kind="recommendation", text="Offer an ACE inhibitor.", rid="NG106-1.4.3", num="1.4.3"):
    return Record(record_id=rid, guideline_code="NG106", guideline_title="Heart failure",
                  section_path="Treating > Drugs", recommendation_id=num, last_updated="2025-09-03",
                  source_url="https://x/y#z", content_type=kind, text=text)


def test_recommendation_is_never_split_even_when_long():
    long_text = " ".join(["word"] * 3000)
    chunks = chunk_record(rec(text=long_text), max_tokens=500, overlap_tokens=50)
    assert len(chunks) == 1 and chunks[0].text == long_text and chunks[0].chunk_id == "NG106-1.4.3"


def test_embed_text_has_context_header():
    c = chunk_record(rec(), 500, 50)[0]
    assert c.embed_text.startswith("NG106 | Treating > Drugs | 1.4.3\n")


def make_table(n_rows):
    rows = [f"| drug{i} | {'dose ' * 40}|" for i in range(n_rows)]
    return "Table 9: Doses\n\n| Drug | Dose |\n| --- | --- |\n" + "\n".join(rows)


def test_small_table_is_one_chunk():
    assert len(chunk_record(rec("table", make_table(3), "NG106-table-1", None), 500, 50)) == 1


def test_large_table_split_by_rows_repeats_header():
    parts = split_table(make_table(60), max_tokens=300)
    assert len(parts) > 1
    for p in parts:
        assert "Table 9: Doses" in p and "| Drug | Dose |" in p and "| --- | --- |" in p
    rows_in = sum(p.count("| drug") for p in parts)
    assert rows_in == 60  # no row lost or duplicated


def test_long_context_split_at_paragraphs_with_overlap():
    paras = [f"Paragraph {i}. " + "filler " * 60 for i in range(12)]
    parts = split_long_text("\n".join(paras), max_tokens=200, overlap_tokens=50)
    assert len(parts) > 1
    assert all(count_tokens(p) <= 200 + 50 for p in parts)
    # Overlap: the last few words of a part reappear at the start of the next part.
    assert " ".join(parts[0].split()[-5:]) in parts[1][:400]


def test_short_context_not_split():
    assert split_long_text("short text", 500, 50) == ["short text"]


# --- Integration tests on the real parsed guidelines -------------------------------------------

records_file = project_path(load_config()["paths"]["processed"]) / "records.jsonl"


@pytest.fixture(scope="module")
def real():
    if not records_file.exists():
        pytest.skip("run `python -m guideline_rag.ingest` first")
    records = [Record.model_validate_json(l) for l in records_file.read_text(encoding="utf-8").splitlines()]
    return records, build_chunks()


def test_real_no_recommendation_split(real):
    records, chunks = real
    by_id = {c.chunk_id: c for c in chunks}
    for r in records:
        if r.content_type == "recommendation":
            assert by_id[r.record_id].text == r.text  # exactly one chunk, text unchanged


def test_real_every_table_part_has_header(real):
    _, chunks = real
    tables = [c for c in chunks if c.content_type == "table"]
    assert tables
    for c in tables:
        lines = c.text.split("\n")
        assert any(l.startswith("| ---") or l.startswith("|---") for l in lines[:6]), c.chunk_id


def test_real_all_chunks_have_required_metadata_and_unique_ids(real):
    _, chunks = real
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for c in chunks:
        d = c.model_dump()
        for field in REQUIRED:
            assert d[field], f"{c.chunk_id} missing {field}"
        if c.content_type == "recommendation":
            assert c.recommendation_id
