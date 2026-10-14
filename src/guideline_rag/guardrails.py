"""Safety checks that run before retrieval (input) and after generation (output).

Everything lives in this one file so it is easy to read and test. Keyword lists come
from ``guardrails.yaml``. Where a keyword list is not enough we add a quick LLM
classification, but the LLM is optional (``use_llm=False``) so tests are deterministic.

Input routes, in priority order:
    urgent        -> possible emergency: do NOT answer from the guidelines, point to emergency help
    out_of_scope  -> nothing to do with the indexed guidelines: say what is covered
    personal      -> asks for advice about the person's own treatment: answer only as general info
    ok            -> normal question
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from functools import lru_cache

import ollama
import yaml
from pydantic import BaseModel

from guideline_rag.config import PROJECT_ROOT, load_config
from guideline_rag.generation import SAFE_ANSWER, Answer

EDUCATIONAL_NOTICE = ("This assistant summarizes public clinical guidelines for learning purposes. "
                      "It does not replace advice from a qualified healthcare professional.")

URGENT_MESSAGE = (
    "What you describe could be a medical emergency, so I can't answer it from the guidelines. "
    "Please contact emergency services now (999 in the UK, 911 in the US, 112 in the EU) or go to the "
    "nearest emergency department. If you are having thoughts of harming yourself, call a crisis line "
    "straight away (in the UK, Samaritans on 116 123; in the US, call or text 988) or tell someone you trust.")

PERSONAL_MESSAGE = (
    "This assistant is an educational tool that summarizes guidelines. It can't advise on your own "
    "treatment, so please speak with your doctor or pharmacist before changing anything. "
    "What follows is general information from the guidelines only.")

STALE_YEARS = 3


class InputDecision(BaseModel):
    route: str  # urgent | out_of_scope | personal | ok
    message: str = ""
    reason: str = ""  # which rule fired (useful for tests and debugging)


@lru_cache(maxsize=1)
def _rules() -> dict:
    with open(PROJECT_ROOT / "guardrails.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _contains_any(text: str, phrases: list[str]) -> str | None:
    text = text.lower().replace("’", "'")
    for p in phrases:
        if p.lower() in text:
            return p
    return None


def _has_person_marker(text: str) -> bool:
    words = set(re.findall(r"[a-z']+", text.lower().replace("’", "'")))
    return any(m in words for m in _rules()["person_markers"])


def _scope_message() -> str:
    topics = "; ".join(_rules()["scope_topics"])
    return f"I can only answer questions about the indexed guidelines: {topics}."


def _llm_yes_no(prompt: str) -> bool:
    """Ask the local model a strict yes/no question. Any failure counts as 'no'."""
    cfg = load_config()["models"]
    try:
        client = ollama.Client(host=cfg["ollama_url"], timeout=120)
        reply = client.chat(model=cfg["llm"], options={"temperature": 0, "num_predict": 3},
                            messages=[{"role": "user", "content": prompt + "\nAnswer with one word: yes or no."}])
        return reply["message"]["content"].strip().lower().startswith("y")
    except Exception:
        return False


def is_urgent(question: str, use_llm: bool = True) -> str | None:
    """Return the reason if the question describes a possible emergency, else None."""
    rules = _rules()
    hit = _contains_any(question, rules["self_harm_phrases"])
    if hit:
        return f"self-harm phrase: {hit}"
    hit = _contains_any(question, rules["emergency_symptoms"])
    if hit and _has_person_marker(question):
        return f"emergency symptom: {hit}"
    # Second opinion from the model, only for first-person statements the keyword list missed.
    if use_llm and _has_person_marker(question) and _llm_yes_no(
            "Does this message describe someone who may be having a medical emergency right now "
            "(for example chest pain, severe breathing difficulty, stroke signs, or thoughts of self-harm)?\n"
            f"Message: {question}"):
        return "LLM classified as emergency"
    return None


def is_personal(question: str) -> str | None:
    q = question.lower().replace("’", "'")
    for pattern in _rules()["personal_patterns"]:
        if re.search(pattern, q):
            return f"personal pattern: {pattern}"
    return None


def is_in_scope(question: str, use_llm: bool = True) -> bool:
    if _contains_any(question, _rules()["scope_keywords"]):
        return True
    if use_llm:
        topics = "; ".join(_rules()["scope_topics"])
        return _llm_yes_no(f"Is this question about any of these topics: {topics}?\nQuestion: {question}")
    return False


def check_input(question: str, use_llm: bool = True) -> InputDecision:
    """Decide how to route a question before spending any effort on retrieval."""
    reason = is_urgent(question, use_llm)
    if reason:
        return InputDecision(route="urgent", message=URGENT_MESSAGE, reason=reason)
    personal = is_personal(question)
    if not is_in_scope(question, use_llm):
        msg = _scope_message()
        if personal:  # e.g. "What dose should I take?" with no topic named
            msg += " For questions about your own treatment, please speak with a doctor or pharmacist."
        return InputDecision(route="out_of_scope", message=msg, reason="no indexed topic found")
    if personal:
        return InputDecision(route="personal", message=PERSONAL_MESSAGE, reason=personal)
    return InputDecision(route="ok")


def staleness_notes(answer: Answer, today: date | None = None) -> list[str]:
    """Warn when a cited guideline was last updated more than three years ago."""
    today = today or date.today()
    cutoff = today - timedelta(days=365 * STALE_YEARS)
    notes, seen = [], set()
    for s in answer.sources:
        if s.guideline_code in seen:
            continue
        seen.add(s.guideline_code)
        try:
            updated = date.fromisoformat(s.last_updated)
        except ValueError:
            continue
        if updated < cutoff:
            notes.append(f"{s.guideline_code} was last updated on {updated:%d %B %Y}, more than "
                         f"{STALE_YEARS} years ago. It may have been updated since; check the NICE website.")
    return notes


def check_output(answer: Answer, today: date | None = None) -> Answer:
    """Final checks on a generated answer."""
    # 1. Never show an answer without a valid citation.
    if answer.answerable and (not answer.citations or not answer.sources):
        return answer.model_copy(update={"answer": SAFE_ANSWER, "answerable": False, "verified": False,
                                         "citations": [], "supporting_quotes": [], "sources": []})
    # 2. Out-of-date warning.
    return answer.model_copy(update={"notes": answer.notes + staleness_notes(answer, today)})
