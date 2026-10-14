"""Retrieval tests. The pure-logic tests always run; the index tests need the built indexes."""

import pytest

from guideline_rag.config import load_config, project_path
from guideline_rag.retrieval import MODES, reciprocal_rank_fusion


def test_rrf_rewards_chunks_found_by_both_searches():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])
    ids = [i for i, _ in fused]
    assert ids[0] == "a"  # rank 1 + rank 2 beats c's rank 3 + rank 1
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_single_list_keeps_order():
    assert [i for i, _ in reciprocal_rank_fusion([["x", "y", "z"]])] == ["x", "y", "z"]


index_ready = (project_path(load_config()["paths"]["bm25"]) / "retriever.json").exists() or \
    project_path(load_config()["paths"]["bm25"]).exists()
needs_index = pytest.mark.skipif(not index_ready, reason="build indexes first: python -m guideline_rag.index --rebuild")


@needs_index
@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_five_results(mode):
    from guideline_rag.retrieval import retrieve
    results = retrieve("first-line treatment for heart failure with reduced ejection fraction", mode=mode)
    assert len(results) == 5
    assert any(r.chunk.guideline_code == "NG106" for r in results)


@needs_index
def test_guideline_filter_limits_results():
    from guideline_rag.retrieval import retrieve
    for mode in MODES:
        results = retrieve("blood pressure target", mode=mode, guideline="NG136")
        assert results and all(r.chunk.guideline_code == "NG136" for r in results)


@needs_index
def test_exact_drug_name_found_in_hybrid_mode():
    from guideline_rag.retrieval import retrieve
    results = retrieve("ivabradine", mode="hybrid")
    assert any("ivabradine" in r.chunk.text.lower() for r in results[:3])
    assert any(r.chunk.chunk_id == "NG106-1.4.8" for r in results)
