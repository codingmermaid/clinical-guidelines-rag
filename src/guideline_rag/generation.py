"""Grounded answer generation with citations that are verified in code.

The language model is *asked* to cite its sources, but we never simply trust it.
After generation we check, in plain Python, that

1. every cited chunk_id is one of the passages we actually retrieved, and
2. every supporting quote really appears (after light normalisation) in a cited passage.

If a check fails we retry once with a correction message, and if it fails again we
return a safe "could not find a reliable answer" response instead of an unverified one.
"""

from __future__ import annotations

import re
import unicodedata

import ollama
from pydantic import BaseModel, Field

from guideline_rag.config import load_config
from guideline_rag.retrieval import Retrieved

SAFE_ANSWER = "I could not find a reliable answer in the guidelines."
NOT_ANSWERABLE = "The guidelines provided do not answer this question."

SYSTEM_PROMPT = """You are an assistant that explains public clinical guidelines for learning purposes.

Rules you must follow:
1. Answer ONLY using the numbered passages provided below. Never use outside knowledge.
2. Support every claim by citing the chunk_id of the passage it came from (in the "citations" list).
3. In "supporting_quotes" copy short EXACT quotes (a phrase or one sentence) from the cited passages,
   word for word, with no changes.
4. If the passages do not contain the answer, set "answerable" to false, leave "citations" and
   "supporting_quotes" empty, and write a full sentence in "answer" saying that the guidelines
   provided do not answer the question.
5. Write in plain, clear language. Do not give personal medical advice or tell the person what
   they should do; describe what the guideline recommends.
6. Reply with JSON only, matching the requested schema."""


class LLMAnswer(BaseModel):
    """The structured reply we ask the model for (also used as the JSON schema)."""

    answer: str = Field(description="Plain-language answer based only on the passages")
    citations: list[str] = Field(description="chunk_id values of the passages used")
    supporting_quotes: list[str] = Field(description="Short exact quotes copied from the cited passages")
    answerable: bool = Field(description="False if the passages do not answer the question")


class Source(BaseModel):
    """A cited passage with everything needed to show and link it."""

    chunk_id: str
    recommendation_id: str | None
    guideline_code: str
    guideline_title: str
    section_path: str
    last_updated: str
    source_url: str
    content_type: str
    text: str


class Answer(BaseModel):
    """Final result returned to callers (API, UI, evaluation)."""

    question: str
    answer: str
    answerable: bool
    verified: bool  # True when the citation checks passed (or the model correctly declined)
    citations: list[str] = []
    supporting_quotes: list[str] = []
    sources: list[Source] = []  # the cited passages, in full
    retrieved: list[Source] = []  # everything the retriever found, for transparency
    notes: list[str] = []  # warnings such as "guideline may be out of date"
    route: str = "answer"  # set by the pipeline / guardrails
    attempts: int = 1  # LLM calls used; 2 means the first reply failed citation checks (used by evaluation)


def to_source(r: Retrieved) -> Source:
    c = r.chunk
    return Source(chunk_id=c.chunk_id, recommendation_id=c.recommendation_id, guideline_code=c.guideline_code,
                  guideline_title=c.guideline_title, section_path=c.section_path, last_updated=c.last_updated,
                  source_url=c.source_url, content_type=c.content_type, text=c.text)


def normalize(text: str) -> str:
    """Make quote matching forgiving about typography, not about wording."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[‐-―−]", "-", text)  # all dash/hyphen variants -> "-"
    text = re.sub(r"[^\w\s%/.-]", " ", text)  # drop quotes, brackets and other punctuation
    return re.sub(r"\s+", " ", text).strip(" .")


def verify(result: LLMAnswer, retrieved: list[Retrieved]) -> list[str]:
    """Return a list of problems (empty list = the answer is properly grounded)."""
    if not result.answerable:
        return []
    by_id = {r.chunk.chunk_id: r.chunk for r in retrieved}
    problems: list[str] = []
    if not result.citations:
        problems.append("An answerable reply must cite at least one chunk_id.")
    unknown = [c for c in result.citations if c not in by_id]
    if unknown:
        problems.append(f"These citations are not among the provided passages: {unknown}.")
    if not result.supporting_quotes:
        problems.append("An answerable reply must include at least one supporting quote.")
    cited_text = [normalize(by_id[c].text) for c in result.citations if c in by_id]
    for quote in result.supporting_quotes:
        q = normalize(quote)
        if not q or not any(q in t for t in cited_text):
            problems.append(f'This quote was not found word-for-word in the cited passages: "{quote}".')
    return problems


def format_passages(retrieved: list[Retrieved]) -> str:
    blocks = []
    for r in retrieved:
        c = r.chunk
        label = f"chunk_id: {c.chunk_id} | {c.guideline_code} | {c.section_path}"
        blocks.append(f"[{label}]\n{c.text}")
    return "\n\n".join(blocks)


def _ask_llm(messages: list[dict]) -> LLMAnswer:
    cfg = load_config()["models"]
    client = ollama.Client(host=cfg["ollama_url"], timeout=300)
    reply = client.chat(model=cfg["llm"], messages=messages, format=LLMAnswer.model_json_schema(),
                        options={"temperature": 0})
    return LLMAnswer.model_validate_json(reply["message"]["content"])


def generate_answer(question: str, retrieved: list[Retrieved], extra_instruction: str = "") -> Answer:
    """Ask the model, verify its citations, retry once, then fall back to a safe reply."""
    system = SYSTEM_PROMPT + (f"\n\n{extra_instruction}" if extra_instruction else "")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": f"Passages:\n\n{format_passages(retrieved)}\n\nQuestion: {question}"}]
    all_sources = [to_source(r) for r in retrieved]
    by_id = {s.chunk_id: s for s in all_sources}

    result, problems = None, ["no reply"]
    used = 0
    for attempt in range(2):  # first try + one retry
        used += 1
        try:
            result = _ask_llm(messages)
            problems = verify(result, retrieved)
        except Exception as err:  # malformed JSON etc. counts as a failed attempt
            result, problems = None, [f"The reply was not valid JSON for the schema ({type(err).__name__})."]
        if not problems:
            break
        if attempt == 0:
            messages = messages + [
                {"role": "assistant", "content": result.model_dump_json() if result else "{}"},
                {"role": "user", "content": "Your reply failed verification: " + " ".join(problems)
                 + " Reply again. Cite only provided chunk_ids and quote the passages exactly."}]

    if problems or result is None:
        return Answer(question=question, answer=SAFE_ANSWER, answerable=False, verified=False,
                      retrieved=all_sources, route="unverified", attempts=used)
    if not result.answerable:
        # Small models sometimes put junk such as "false" in the text field; replace it.
        text = result.answer if len(result.answer.strip()) >= 20 else NOT_ANSWERABLE
        return Answer(question=question, answer=text, answerable=False, verified=True,
                      retrieved=all_sources, attempts=used)
    return Answer(question=question, answer=result.answer, answerable=True, verified=True, attempts=used,
                  citations=result.citations, supporting_quotes=result.supporting_quotes,
                  sources=[by_id[c] for c in dict.fromkeys(result.citations)], retrieved=all_sources)
