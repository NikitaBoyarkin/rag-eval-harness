"""Evaluation: retrieval metrics + generation metrics.

Retrieval: recall@k, MRR, precision@k.
Generation: faithfulness (answer grounded in retrieved context) and answer
correctness (token overlap with the gold answer).
"""
from __future__ import annotations

import json
import re

import pandas as pd

from .index import Index, build_index, search
from .generate import generate


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / (len(a | b) or 1)


def retrieval_metrics(index: Index, eval_set: list[dict], k: int = 5) -> pd.DataFrame:
    rows = []
    for item in eval_set:
        gold = set(item["gold_doc_ids"])
        retrieved = [d for d, _ in search(index, item["question"], top_k=k)]
        hits = [1 if d in gold else 0 for d in retrieved]
        first_hit = next((i + 1 for i, h in enumerate(hits) if h), None)
        rows.append({
            "id": item["id"],
            "recall@k": float(bool(hits) and any(hits)),   # 1 if any gold in top-k
            "mrr": 1.0 / first_hit if first_hit else 0.0,
            "precision@k": sum(hits) / k,
            "gold": list(gold),
            "retrieved": retrieved,
        })
    return pd.DataFrame(rows)


def faithfulness(answer: str, context_doc_ids: list[str], index: Index) -> float:
    """Share of answer tokens that appear in the retrieved context (grounding)."""
    if not answer:
        return 0.0
    atok = _tokens(answer)
    if not atok:
        return 0.0
    ctx_tokens = set()
    for did in context_doc_ids:
        if did in index.doc_ids:
            ctx_tokens |= _tokens(index.texts[index.doc_ids.index(did)])
    return len(atok & ctx_tokens) / len(atok)


def correctness(answer: str, gold: str) -> float:
    return _jaccard(_tokens(answer), _tokens(gold))


def run_eval(force_offline: bool = True, k: int = 5) -> dict:
    from . import config as C
    index = build_index()
    eval_set = [json.loads(l) for l in C.EVAL.read_text().splitlines() if l.strip()]
    retr = retrieval_metrics(index, eval_set, k=k)

    gen_rows = []
    for item in eval_set:
        g = generate(item["question"], index, top_k=k, force_offline=force_offline)
        gen_rows.append({
            "id": item["id"],
            "answer": g.answer,
            "faithfulness": faithfulness(g.answer, g.context_doc_ids, index),
            "correctness": correctness(g.answer, item["gold_answer"]),
            "used_llm": g.used_llm,
        })
    gen = pd.DataFrame(gen_rows)

    summary = {
        "n": len(eval_set),
        "k": k,
        "llm_used": bool(gen["used_llm"].any()),
        "retrieval": {
            "recall@k": round(float(retr["recall@k"].mean()), 3),
            "mrr": round(float(retr["mrr"].mean()), 3),
            "precision@k": round(float(retr["precision@k"].mean()), 3),
        },
        "generation": {
            "faithfulness": round(float(gen["faithfulness"].mean()), 3),
            "correctness": round(float(gen["correctness"].mean()), 3),
        },
    }
    return {"summary": summary, "retrieval": retr, "generation": gen}

