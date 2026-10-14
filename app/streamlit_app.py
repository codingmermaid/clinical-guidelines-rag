"""Streamlit user interface. It talks to the FastAPI backend (default http://localhost:8000).

Run everything with:  python start.py
"""

import os
from datetime import datetime

import requests
import streamlit as st

API_URL = os.environ.get("GUIDELINE_API_URL", "http://localhost:8000")
NOTICE = ("This assistant summarizes public clinical guidelines for learning purposes. "
          "It does not replace advice from a qualified healthcare professional.")
GUIDELINES = {"All guidelines": None, "NG106 Chronic heart failure": "NG106", "NG28 Type 2 diabetes": "NG28",
              "NG115 COPD": "NG115", "NG136 Hypertension": "NG136"}

st.set_page_config(page_title="Clinical Guidelines Assistant", page_icon="🩺", layout="centered")
st.title("Clinical Guidelines Assistant")
st.info(NOTICE)  # always visible

with st.sidebar:
    st.header("Options")
    mode_label = st.selectbox("Search method", ["Hybrid + reranking (best)", "Hybrid", "Keyword (BM25)", "Meaning (vector)"])
    mode = {"Hybrid + reranking (best)": "hybrid_rerank", "Hybrid": "hybrid", "Keyword (BM25)": "bm25",
            "Meaning (vector)": "vector"}[mode_label]
    st.caption("Answers come only from NICE guidelines NG106, NG28, NG115 and NG136, "
               "run on a small local model. Always check the cited passage.")

with st.form("ask"):
    question = st.text_area("Your question", height=90,
                            placeholder="e.g. What are the clinic blood pressure targets for adults under 80?")
    guideline_label = st.selectbox("Limit to a guideline (optional)", list(GUIDELINES))
    submitted = st.form_submit_button("Ask", type="primary")


def pretty_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d %B %Y")
    except ValueError:
        return iso


def show_source(src: dict, expanded: bool = False) -> None:
    rec = f" · recommendation {src['recommendation_id']}" if src.get("recommendation_id") else f" · {src['content_type']}"
    label = f"{src['guideline_code']}{rec} — {src['guideline_title']}"
    with st.expander(label, expanded=expanded):
        st.caption(f"{src['section_path']}  \nLast updated: {pretty_date(src['last_updated'])}  ·  id: `{src['chunk_id']}`")
        st.markdown(src["text"])
        st.markdown(f"[Open on the NICE website]({src['source_url']})")


if submitted:
    if len(question.strip()) < 3:
        st.warning("Please type a question.")
        st.stop()
    try:
        with st.spinner("Searching the guidelines and writing an answer (this can take 10-30 seconds)..."):
            resp = requests.post(f"{API_URL}/ask", timeout=600, json={
                "question": question.strip(), "guideline": GUIDELINES[guideline_label], "mode": mode})
            resp.raise_for_status()
            data = resp.json()
    except requests.RequestException as err:
        st.error(f"Could not reach the backend at {API_URL}. Start it with `python start.py`. ({err})")
        st.stop()

    route = data["route"]
    if route == "urgent":
        st.error(data["answer"])
    elif route == "out_of_scope":
        st.warning(data["answer"])
    else:
        for note in data["notes"]:
            st.warning(note)
        if data["answerable"]:
            st.subheader("Answer")
            st.write(data["answer"])
            st.subheader(f"Sources ({len(data['sources'])})")
            for s in data["sources"]:
                show_source(s)
            if data["supporting_quotes"]:
                with st.expander("Exact quotes the answer relies on"):
                    for q in data["supporting_quotes"]:
                        st.markdown(f"> {q}")
        else:
            st.subheader("No answer found")
            st.write(data["answer"])
        with st.expander(f"Show retrieved passages ({len(data['retrieved'])})"):
            st.caption("These are the passages the search found, in order. The answer may use only some of them.")
            for i, s in enumerate(data["retrieved"], 1):
                st.markdown(f"**{i}.** `{s['chunk_id']}` — {s['section_path']}")
                st.text(s["text"][:700] + ("…" if len(s["text"]) > 700 else ""))
    st.caption(NOTICE)
