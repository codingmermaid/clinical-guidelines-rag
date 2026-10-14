"""Guardrail routing tests. LLM calls are switched off (use_llm=False) so results are deterministic."""

from datetime import date

import pytest

from guideline_rag.generation import Answer, Source
from guideline_rag.guardrails import (EDUCATIONAL_NOTICE, SAFE_ANSWER, check_input, check_output,
                                      staleness_notes)

URGENT = [
    "I have crushing chest pain and my left arm feels numb",
    "My dad can't breathe properly and his lips are blue lips",
    "I think I am having a stroke, my face drooping on one side",
    "I want to kill myself, what should I do about my heart failure",
    "She collapsed and is unconscious, she has diabetes",
]
PERSONAL = [
    "Should I stop taking metformin?",
    "What dose of ramipril should I take for my blood pressure?",
    "Can I skip my inhaler if my COPD feels better?",
    "I was prescribed amlodipine for hypertension, is it safe for me?",
]
OUT_OF_SCOPE = [
    "What is the recommended treatment for migraine headaches?",
    "How many calories should a marathon runner eat?",
    "Who won the football world cup in 2018?",
    "What dose should I take?",
]
OK = [
    "What are the clinic blood pressure targets for adults under 80?",
    "How is COPD diagnosed with spirometry?",
    "Which medicines are offered first for heart failure with reduced ejection fraction?",
    "When is metformin recommended for type 2 diabetes?",
]


@pytest.mark.parametrize("q", URGENT)
def test_urgent_questions_are_routed_to_emergency_message(q):
    d = check_input(q, use_llm=False)
    assert d.route == "urgent" and "emergency" in d.message.lower()


@pytest.mark.parametrize("q", PERSONAL)
def test_personal_questions_are_labelled(q):
    d = check_input(q, use_llm=False)
    assert d.route == "personal" and "doctor or pharmacist" in d.message


@pytest.mark.parametrize("q", OUT_OF_SCOPE)
def test_out_of_scope_lists_covered_topics(q):
    d = check_input(q, use_llm=False)
    assert d.route == "out_of_scope" and "NG106" in d.message and "NG136" in d.message


@pytest.mark.parametrize("q", OK)
def test_normal_questions_pass(q):
    assert check_input(q, use_llm=False).route == "ok"


def test_general_symptom_question_is_not_an_emergency():
    # No person is affected, so this is an information question, not an emergency.
    assert check_input("How is chest pain assessed in heart failure guidelines?", use_llm=False).route == "ok"


def test_self_harm_is_urgent_even_without_person_marker():
    assert check_input("overdose of insulin", use_llm=False).route == "urgent"


def source(code, updated):
    return Source(chunk_id=f"{code}-1.1.1", recommendation_id="1.1.1", guideline_code=code, guideline_title="t",
                  section_path="s", last_updated=updated, source_url="u", content_type="recommendation", text="t")


def answer(sources, **kw):
    base = dict(question="q", answer="a", answerable=True, verified=True,
                citations=[s.chunk_id for s in sources], sources=sources)
    return Answer(**{**base, **kw})


def test_answer_without_citation_is_replaced_by_safe_response():
    out = check_output(answer([], citations=[]))
    assert out.answer == SAFE_ANSWER and not out.answerable


def test_old_guideline_gets_a_note():
    out = check_output(answer([source("NG115", "2019-07-26")]), today=date(2026, 10, 14))
    assert any("NG115" in n and "may have been updated" in n for n in out.notes)


def test_recent_guideline_gets_no_note():
    assert staleness_notes(answer([source("NG28", "2026-02-18")]), today=date(2026, 10, 14)) == []


def test_notice_text():
    assert EDUCATIONAL_NOTICE.startswith("This assistant summarizes public clinical guidelines")
