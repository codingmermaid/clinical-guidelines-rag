"""Run the full evaluation for each retrieval mode and save JSON, CSV and a bar chart.

Metrics (all computed per retrieval mode):
  hit_rate@5, mrr          retrieval quality, plain Python from the expected chunk_ids
  faithfulness             every claim supported by the retrieved text (DeepEval, local Ollama judge)
  answer_relevance         the answer addresses the question (DeepEval, local Ollama judge)
  citation_valid_final     answers whose citations pass the code checks (after the one allowed retry)
  citation_valid_first     same, but on the model's first attempt (shows how much the retry helps)
  refusal_accuracy         unanswerable questions correctly declined
  wrong_refusal_rate       answerable questions wrongly declined
  guardrail_accuracy       guardrail questions routed to the expected category (same for every mode)

Usage:
  python eval/run_eval.py                      # all four modes, whole test set
  python eval/run_eval.py --limit 5            # quick smoke test: 5 items per type
  python eval/run_eval.py --modes hybrid       # one mode only
  python eval/run_eval.py --reviewed-only      # only items you have marked reviewed: true
"""

import argparse
import csv
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

import guideline_rag  # noqa: E402,F401  (applies the Windows certificate workaround first)
import ollama  # noqa: E402
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402

from guideline_rag.config import load_config  # noqa: E402
from guideline_rag.generation import generate_answer  # noqa: E402
from guideline_rag.guardrails import check_input, check_output  # noqa: E402
from guideline_rag.retrieval import MODES, retrieve  # noqa: E402

HERE = Path(__file__).parent
RESULTS = HERE / "results"
_retrieval_lock = threading.Lock()  # the reranker and BM25 retriever are not thread-safe
METRICS = ["hit_rate@5", "mrr", "faithfulness", "answer_relevance", "citation_valid_final",
           "citation_valid_first", "refusal_accuracy", "wrong_refusal_rate", "guardrail_accuracy"]


class LocalJudge(DeepEvalBaseLLM):
    """Lets DeepEval use the local Ollama model as its judge (so nothing leaves this computer)."""

    def __init__(self):
        cfg = load_config()["models"]
        self.model, self.client = cfg["llm"], ollama.Client(host=cfg["ollama_url"], timeout=300)

    def load_model(self):
        return self.client

    def generate(self, prompt, schema=None):
        kwargs = {"format": schema.model_json_schema()} if schema else {}
        reply = self.client.chat(model=self.model, messages=[{"role": "user", "content": prompt}],
                                 options={"temperature": 0}, **kwargs)
        text = reply["message"]["content"]
        return schema.model_validate_json(text) if schema else text

    async def a_generate(self, prompt, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model


def judge_scores(question: str, answer: str, contexts: list[str]) -> tuple[float | None, float | None]:
    """Return (faithfulness, answer_relevance) or None for a metric that failed."""
    case = LLMTestCase(input=question, actual_output=answer, retrieval_context=contexts)
    out = []
    for metric_cls in (FaithfulnessMetric, AnswerRelevancyMetric):
        try:
            metric = metric_cls(model=LocalJudge(), async_mode=False, verbose_mode=False)
            metric.measure(case)
            out.append(float(metric.score))
        except Exception as err:  # a judge failure must not crash the whole run
            print(f"  judge failed ({metric_cls.__name__}): {repr(err)[:100]}")
            out.append(None)
    return out[0], out[1]


def reciprocal_rank(ranked_ids: list[str], expected: list[str]) -> float:
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in expected:
            return 1.0 / rank
    return 0.0


def run_item(item: dict, mode: str, use_judge: bool) -> dict:
    """Answer one answerable/unanswerable question with the given retrieval mode."""
    start = time.time()
    with _retrieval_lock:
        found = retrieve(item["question"], mode=mode, top_k=20)  # 20 so MRR sees deeper ranks
    ranked_ids = [r.chunk.chunk_id for r in found]
    top5 = found[:5]
    answer = check_output(generate_answer(item["question"], top5))
    res = {"id": item["id"], "type": item["type"], "question": item["question"], "mode": mode,
           "answer": answer.answer, "answerable": answer.answerable, "verified": answer.verified,
           "route": answer.route, "citations": answer.citations, "attempts": answer.attempts,
           "retrieved_top5": ranked_ids[:5], "latency_s": round(time.time() - start, 1)}
    if item["type"] == "answerable":
        res["hit@5"] = any(c in item["expected_chunk_ids"] for c in ranked_ids[:5])
        res["rr"] = reciprocal_rank(ranked_ids, item["expected_chunk_ids"])
        res["expected_chunk_ids"] = item["expected_chunk_ids"]
        if use_judge and answer.answerable:
            res["faithfulness"], res["answer_relevance"] = judge_scores(
                item["question"], answer.answer, [r.chunk.text for r in top5])
    return res


def mean(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarize(results: list[dict], guardrail_acc: float | None) -> dict:
    ans = [r for r in results if r["type"] == "answerable"]
    una = [r for r in results if r["type"] == "unanswerable"]
    produced = [r for r in results if r["answerable"] or r["route"] == "unverified"]  # model tried to answer
    return {
        "hit_rate@5": mean([float(r["hit@5"]) for r in ans]),
        "mrr": mean([r["rr"] for r in ans]),
        "faithfulness": mean([r.get("faithfulness") for r in ans]),
        "answer_relevance": mean([r.get("answer_relevance") for r in ans]),
        "citation_valid_final": mean([float(r["answerable"] and r["verified"]) for r in produced]),
        "citation_valid_first": mean([float(r["attempts"] == 1 and r["verified"]) for r in produced]),
        "refusal_accuracy": mean([float(not r["answerable"]) for r in una]),
        "wrong_refusal_rate": mean([float(not r["answerable"]) for r in ans]),
        "guardrail_accuracy": guardrail_acc,
        "avg_latency_s": mean([r["latency_s"] for r in results]),
        "n_answerable": len(ans), "n_unanswerable": len(una), "n_answers_produced": len(produced),
    }


def run_guardrails(items: list[dict]) -> tuple[float | None, list[dict]]:
    details = []
    for it in items:
        route = check_input(it["question"], use_llm=True).route
        details.append({"id": it["id"], "question": it["question"], "expected": it["expected_route"],
                        "got": route, "correct": route == it["expected_route"]})
    return mean([float(d["correct"]) for d in details]), details


def save_outputs(summaries: dict[str, dict], meta: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps({"meta": meta, "modes": summaries}, indent=2), encoding="utf-8")
    cols = METRICS + ["avg_latency_s"]
    with open(RESULTS / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode"] + cols)
        for mode, s in summaries.items():
            w.writerow([mode] + [s[c] for c in cols])
    make_chart(summaries)


def make_chart(summaries: dict[str, dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["hit_rate@5", "mrr", "faithfulness", "answer_relevance"]
    modes = list(summaries)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]  # distinguishable, colour-blind friendly
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(modes)
    for i, mode in enumerate(modes):
        vals = [summaries[mode][m] or 0 for m in metrics]
        bars = ax.bar([x + i * width for x in range(len(metrics))], vals, width, label=mode, color=colors[i % 4])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks([x + width * (len(modes) - 1) / 2 for x in range(len(metrics))], metrics)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("Retrieval and answer quality by retrieval mode")
    ax.legend(frameon=False, ncol=len(modes), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison.png", dpi=150)
    plt.close(fig)


def print_table(summaries: dict[str, dict]) -> None:
    cols = METRICS + ["avg_latency_s"]
    print("\n" + f"{'mode':14}" + "".join(f"{c[:12]:>13}" for c in cols))
    for mode, s in summaries.items():
        print(f"{mode:14}" + "".join(f"{'n/a' if s[c] is None else s[c]:>13}" for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    ap.add_argument("--limit", type=int, help="items per type (quick test)")
    ap.add_argument("--reviewed-only", action="store_true")
    ap.add_argument("--no-judge", action="store_true", help="skip faithfulness/relevance (much faster)")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    items = json.loads((HERE / "testset.json").read_text(encoding="utf-8"))
    reviewed = sum(i["reviewed"] for i in items)
    print(f"Test set: {len(items)} items, {reviewed} reviewed by a human, {len(items) - reviewed} NOT reviewed.")
    if reviewed < len(items):
        print("WARNING: unreviewed items may contain errors; treat the numbers below as provisional.\n")
    if args.reviewed_only:
        items = [i for i in items if i["reviewed"]]
    if args.limit:
        items = [i for t in ("answerable", "unanswerable", "guardrail")
                 for i in [x for x in items if x["type"] == t][: args.limit]]
    qa_items = [i for i in items if i["type"] != "guardrail"]
    grd_items = [i for i in items if i["type"] == "guardrail"]

    guardrail_acc, guardrail_details = run_guardrails(grd_items)
    print(f"Guardrail accuracy: {guardrail_acc} ({len(grd_items)} items)")

    RESULTS.mkdir(exist_ok=True)
    summaries, t0 = {}, time.time()
    for mode in args.modes:
        print(f"\n=== mode: {mode} ({len(qa_items)} questions) ===")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda it: run_item(it, mode, not args.no_judge), qa_items))
        summaries[mode] = summarize(results, guardrail_acc)
        (RESULTS / f"details_{mode}.json").write_text(
            json.dumps({"results": results, "guardrails": guardrail_details}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(summaries[mode])
    meta = {"items": len(items), "reviewed_items": reviewed, "llm": load_config()["models"]["llm"],
            "seconds": round(time.time() - t0), "limit": args.limit}
    save_outputs(summaries, meta)
    print_table(summaries)
    print(f"\nSaved to {RESULTS} (summary.csv, summary.json, comparison.png, details_*.json)")


if __name__ == "__main__":
    main()
