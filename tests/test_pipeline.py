"""Pipeline routing tests with retrieval and generation mocked (no Ollama, no indexes needed)."""

import guideline_rag.pipeline as p
from guideline_rag.generation import Answer, Source


def fake_answer(q, retrieved, extra=""):
    src = Source(chunk_id="NG115-1.1.1", recommendation_id="1.1.1", guideline_code="NG115", guideline_title="COPD",
                 section_path="s", last_updated="2019-07-26", source_url="u", content_type="recommendation",
                 text="t")
    return Answer(question=q, answer="General info." + (" [personal]" if extra else ""), answerable=True,
                  verified=True, citations=["NG115-1.1.1"], sources=[src])


def setup(monkeypatch):
    calls = {"retrieve": 0}

    def fake_retrieve(*a, **k):
        calls["retrieve"] += 1
        return []

    monkeypatch.setattr(p, "retrieve", fake_retrieve)
    monkeypatch.setattr(p, "generate_answer", fake_answer)
    return calls


def test_urgent_question_never_reaches_retrieval(monkeypatch):
    calls = setup(monkeypatch)
    out = p.ask("I have crushing chest pain and cannot breathe", use_llm_guardrails=False)
    assert out.route == "urgent" and not out.answerable and calls["retrieve"] == 0


def test_out_of_scope_never_reaches_retrieval(monkeypatch):
    calls = setup(monkeypatch)
    out = p.ask("Who won the world cup?", use_llm_guardrails=False)
    assert out.route == "out_of_scope" and calls["retrieve"] == 0


def test_personal_question_is_answered_as_general_info_with_note(monkeypatch):
    setup(monkeypatch)
    out = p.ask("Should I stop taking my metformin?", use_llm_guardrails=False)
    assert out.route == "personal" and out.answerable and "[personal]" in out.answer
    assert "doctor or pharmacist" in out.notes[0]


def test_normal_question_adds_staleness_note_for_old_guideline(monkeypatch):
    setup(monkeypatch)
    out = p.ask("How is COPD diagnosed with spirometry?", use_llm_guardrails=False)
    assert out.route == "ok" and any("NG115" in n for n in out.notes)
