"""Tests for citation verification (no LLM needed) plus an end-to-end check (needs Ollama)."""

import pytest

from guideline_rag.chunking import Chunk
from guideline_rag.generation import LLMAnswer, generate_answer, normalize, verify
from guideline_rag.retrieval import Retrieved


def make(chunk_id, text):
    c = Chunk(chunk_id=chunk_id, text=text, embed_text=text, guideline_code="NG106", guideline_title="HF",
              organization="NICE", section_path="s", recommendation_id="1.1.1", last_updated="2025-01-01",
              source_url="https://x", content_type="recommendation")
    return Retrieved(c, 1.0, 1)


PASSAGES = [make("NG106-1.4.1", "Offer an ACE inhibitor and a beta-blocker to people with HFrEF. [2018]"),
            make("NG106-1.4.2", "Do not offer amlodipine to treat heart failure.")]


def test_normalize_ignores_typography():
    assert normalize("Beta‑blocker, “offer”!") == normalize("beta-blocker offer")


def test_valid_answer_passes():
    a = LLMAnswer(answer="x", citations=["NG106-1.4.1"], supporting_quotes=["Offer an ACE inhibitor"],
                  answerable=True)
    assert verify(a, PASSAGES) == []


def test_unknown_chunk_id_fails():
    a = LLMAnswer(answer="x", citations=["NG999-9.9.9"], supporting_quotes=["Offer an ACE inhibitor"],
                  answerable=True)
    assert any("not among" in p for p in verify(a, PASSAGES))


def test_invented_quote_fails():
    a = LLMAnswer(answer="x", citations=["NG106-1.4.1"], supporting_quotes=["Offer digoxin to everyone"],
                  answerable=True)
    assert any("not found" in p for p in verify(a, PASSAGES))


def test_quote_must_come_from_a_cited_chunk():
    a = LLMAnswer(answer="x", citations=["NG106-1.4.1"], supporting_quotes=["Do not offer amlodipine"],
                  answerable=True)
    assert verify(a, PASSAGES)  # the quote exists, but in a chunk that was not cited


def test_answerable_without_citation_fails():
    a = LLMAnswer(answer="x", citations=[], supporting_quotes=[], answerable=True)
    assert verify(a, PASSAGES)


def test_unanswerable_needs_no_citation():
    a = LLMAnswer(answer="The guidelines do not answer this.", citations=[], supporting_quotes=[],
                  answerable=False)
    assert verify(a, PASSAGES) == []


def test_bad_llm_output_gives_safe_response(monkeypatch):
    import guideline_rag.generation as g
    bad = LLMAnswer(answer="x", citations=["nope"], supporting_quotes=["made up"], answerable=True)
    monkeypatch.setattr(g, "_ask_llm", lambda messages: bad)
    out = generate_answer("q", PASSAGES)
    assert out.answer == g.SAFE_ANSWER and not out.verified and not out.answerable


def test_junk_answer_text_is_replaced_when_unanswerable(monkeypatch):
    import guideline_rag.generation as g
    monkeypatch.setattr(g, "_ask_llm", lambda m: LLMAnswer(answer="false", citations=[],
                                                          supporting_quotes=[], answerable=False))
    out = generate_answer("q", PASSAGES)
    assert out.answer == g.NOT_ANSWERABLE and not out.answerable and out.verified


def test_retry_succeeds_after_correction(monkeypatch):
    import guideline_rag.generation as g
    replies = iter([LLMAnswer(answer="x", citations=["nope"], supporting_quotes=["made up"], answerable=True),
                    LLMAnswer(answer="Offer ACE inhibitors.", citations=["NG106-1.4.1"],
                              supporting_quotes=["Offer an ACE inhibitor"], answerable=True)])
    monkeypatch.setattr(g, "_ask_llm", lambda messages: next(replies))
    out = generate_answer("q", PASSAGES)
    assert out.verified and out.citations == ["NG106-1.4.1"] and out.sources[0].chunk_id == "NG106-1.4.1"
