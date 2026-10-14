"""One function that runs the whole flow: guardrails -> retrieval -> generation -> checks."""

from __future__ import annotations

from guideline_rag.generation import Answer, generate_answer
from guideline_rag.guardrails import EDUCATIONAL_NOTICE, check_input, check_output
from guideline_rag.retrieval import retrieve
from guideline_rag.tracing import span

PERSONAL_INSTRUCTION = ("The user is asking about their own care. Describe only what the guidelines say in "
                        "general terms, label it as general information, and do not tell them what to do.")
NOTICE = EDUCATIONAL_NOTICE


def ask(question: str, guideline: str | None = None, mode: str | None = None,
        use_llm_guardrails: bool = True) -> Answer:
    """Answer a question from the guidelines, with safety checks on both sides."""
    with span("ask", question=question, guideline=guideline, mode=mode) as root:
        with span("input_guardrails") as s:
            decision = check_input(question, use_llm=use_llm_guardrails)
            s.set_attribute("route", decision.route)
            s.set_attribute("reason", decision.reason)

        if decision.route in ("urgent", "out_of_scope"):
            # No retrieval: for emergencies we must not answer from the guidelines at all.
            return Answer(question=question, answer=decision.message, answerable=False, verified=True,
                          route=decision.route)

        with span("retrieval") as s:
            retrieved = retrieve(question, mode=mode, guideline=guideline)
            s.set_attribute("chunk_ids", ", ".join(r.chunk.chunk_id for r in retrieved))

        with span("generation") as s:
            answer = generate_answer(question, retrieved,
                                     PERSONAL_INSTRUCTION if decision.route == "personal" else "")
            s.set_attribute("citations", ", ".join(answer.citations))
            s.set_attribute("attempts", answer.attempts)

        with span("output_checks"):
            answer = check_output(answer)
        answer.route = decision.route
        if decision.route == "personal":
            answer.notes = [decision.message] + answer.notes
        root.set_attribute("answerable", answer.answerable)
        return answer
