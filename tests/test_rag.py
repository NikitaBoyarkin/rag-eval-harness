"""Tests for the RAG eval harness (offline, no API keys).

Run: uv run --with scikit-learn --with pandas --with numpy --with pytest pytest -q
"""

import json

from src.index import build_index, search
from src.generate import generate
from src.eval import run_eval, faithfulness, correctness


def test_index_loads_corpus():
    idx = build_index()
    assert len(idx) == 12
    assert "cuped" in idx.doc_ids


def test_search_ranks_correct_doc():
    idx = build_index()
    top = search(idx, "How is the CUPED theta coefficient computed?", top_k=3)
    assert top[0][0] == "cuped"


def test_offline_grounding_is_a_construction_invariant():
    """The extractive answer is copied verbatim from the context, so faithfulness is 1.0 by construction.

    This pins an invariant of the zero-key path — it is NOT a quality assertion
    (see prd-improvement REQ-001).
    """
    idx = build_index()
    g = generate("What is MDE in A/B testing?", idx, top_k=3, force_offline=True)
    assert not g.used_llm
    assert g.answer  # non-empty
    # invariant: extractive answer ⊂ context ⇒ faithfulness ≡ 1.0 (not a measurement)
    assert faithfulness(g.answer, g.context_doc_ids, idx) == 1.0


def test_eval_summary_metrics():
    r = run_eval(force_offline=True, k=5)
    s = r["summary"]
    assert s["n"] == 20
    assert s["retrieval"]["recall@k"] == 1.0  # every question's gold doc in top-5
    assert s["retrieval"]["mrr"] >= 0.9
    assert (
        s["generation"]["faithfulness"] == 1.0
    )  # invariant of the extractive path, not a score
    assert (
        s["generation"]["correctness"] > 0.2
    )  # offline extractive still overlaps gold


def test_correctness_jaccard_sanity():
    assert correctness("CUPED reduces variance", "CUPED reduces variance") == 1.0
    assert correctness("completely different words", "nothing shared here zzz") == 0.0


def test_recall_at_k_counts_all_gold_docs(monkeypatch):
    """recall@k is the share of gold docs retrieved, not a binary hit (REQ-007)."""
    from src import eval as ev

    idx = build_index()
    ranked = [("cuped", 0.9), ("rfm", 0.8), ("ltv", 0.7)]

    def fake_search(index, question, top_k):
        return ranked[:top_k]

    monkeypatch.setattr(ev, "search", fake_search)
    eval_set = [{"id": "m", "question": "q", "gold_doc_ids": ["cuped", "rfm", "churn"]}]
    df = ev.retrieval_metrics(idx, eval_set, k=3)
    # 2 of 3 gold docs are in top-3 → recall 2/3, not 1.0
    assert abs(float(df["recall@k"].iloc[0]) - 2 / 3) < 1e-9
