"""API tests with the pipeline mocked, so they need neither Ollama nor the indexes."""

from fastapi.testclient import TestClient

import guideline_rag.api as api
from guideline_rag.generation import Answer

client = TestClient(api.app)


def fake_ask(question, guideline=None, mode=None):
    return Answer(question=question, answer="Offer an ACE inhibitor.", answerable=True, verified=True,
                  citations=["NG106-1.4.1"], route="ok")


def test_ask_returns_answer_and_notice(monkeypatch):
    monkeypatch.setattr(api, "ask", fake_ask)
    r = client.post("/ask", json={"question": "What is offered for heart failure?", "guideline": "NG106"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Offer an ACE inhibitor." and body["citations"] == ["NG106-1.4.1"]
    assert body["notice"].startswith("This assistant summarizes public clinical guidelines")


def test_ask_rejects_bad_mode_and_short_question(monkeypatch):
    monkeypatch.setattr(api, "ask", fake_ask)
    assert client.post("/ask", json={"question": "valid question here", "mode": "magic"}).status_code == 422
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
