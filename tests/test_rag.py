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


def test_offline_generation_is_grounded():
    idx = build_index()
    g = generate("What is MDE in A/B testing?", idx, top_k=3, force_offline=True)
    assert not g.used_llm
    assert g.answer  # non-empty
    assert faithfulness(g.answer, g.context_doc_ids, idx) == 1.0  # extractive => fully grounded


def test_eval_summary_metrics():
    r = run_eval(force_offline=True, k=5)
    s = r["summary"]
    assert s["n"] == 20
    assert s["retrieval"]["recall@k"] == 1.0     # every question's gold doc in top-5
    assert s["retrieval"]["mrr"] >= 0.9
    assert s["generation"]["faithfulness"] == 1.0
    assert s["generation"]["correctness"] > 0.2  # offline extractive still overlaps gold


def test_correctness_jaccard_sanity():
    assert correctness("CUPED reduces variance", "CUPED reduces variance") == 1.0
    assert correctness("completely different words", "nothing shared here zzz") == 0.0
