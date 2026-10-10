# RAG Evaluation Harness

A retrieval-augmented generation pipeline with a proper offline evaluation
harness. The point: you can measure a RAG system, not just demo it — retrieval
metrics (recall@k, MRR, precision@k) and generation metrics (faithfulness,
answer correctness) over a labeled question set.

It runs end-to-end with **zero API keys**: TF-IDF retrieval and a deterministic
extractive answerer provide a reproducible offline baseline. Set
`ANTHROPIC_API_KEY` and pass `--llm` to swap in a Claude model for generation —
retrieval and the eval harness stay the same.

## What it shows

- **Indexing**: 12 short analytics concept docs (A/B testing, CUPED, uplift,
  retention, churn, RFM, funnel, LTV, attribution, cohort, DAU/MAU, SQL window
  functions) → chunked and TF-IDF-indexed with 1–2-grams.
- **Retrieval**: cosine top-k over the TF-IDF matrix. Dependency-light, fully
  offline; an embeddings backend can be dropped in without touching eval.
- **Generation**: a Claude model grounded in retrieved context if a key is set;
  otherwise a deterministic extractive fallback returns the most relevant
  sentence from the top doc. Both paths return the same `Generation` record, so
  eval is reproducible offline.
- **Evaluation**: 20 labeled questions (`data/eval.jsonl`) with gold doc ids and
  gold answers. Retrieval: recall@k, MRR, precision@k. Generation: faithfulness
  (share of answer tokens grounded in retrieved context) and answer
  correctness (Jaccard token overlap with the gold answer).

## Results (offline, k=5, no API key)

| Metric | Value |
|---|---|
| recall@5 | 1.00 |
| MRR | 1.00 |
| precision@5 | 0.20 |
| faithfulness | 1.00 |
| correctness | 0.37 |

Retrieval is perfect on this corpus (the gold doc is rank-1 for all 20
questions — TF-IDF is enough at this scale and topic separation). The offline
extractive baseline is fully grounded (faithfulness 1.0) and reaches 0.37
correctness by token overlap. With `--llm` and an API key, generation scores
rise while retrieval stays fixed — the harness separates the two so you can
see which layer to improve.

Per-question breakdown: `reports/retrieval_per_question.csv`,
`reports/generation_per_question.csv`.

## Quick start

```bash
uv run --with scikit-learn --with pandas --with numpy python run.py query "How is the CUPED theta coefficient computed?"
uv run --with scikit-learn --with pandas --with numpy python run.py eval
# with a real LLM (requires ANTHROPIC_API_KEY):
uv run --with scikit-learn --with pandas --with numpy --with anthropic python run.py eval --llm
uv run --with scikit-learn --with pandas --with numpy --with pytest pytest -q
```

## Layout

```
rag-eval-harness/
├── data/
│   ├── docs/*.md          # 12 corpus docs (knowledge base)
│   └── eval.jsonl         # 20 labeled questions
├── src/
│   ├── config.py          # paths, top-k, model id
│   ├── index.py           # TF-IDF index + search
│   ├── retrieve.py        # retrieval facade
│   ├── generate.py        # LLM with offline extractive fallback
│   └── eval.py            # recall@k, MRR, precision@k, faithfulness, correctness
├── tests/                 # index, search, offline grounding, eval summary
├── reports/               # metrics.json + per-question csvs (gitignored)
├── conftest.py
└── run.py
```

## Notes

- Faithfulness here is a grounding proxy (answer tokens in context), not a
  true entailment check. A real system would use an LLM-as-judge or an
  entailment model. The proxy is deliberately simple so it runs offline and
  deterministically.
- Correctness is Jaccard token overlap with the gold answer — a rough but
  reproducible signal, good enough to compare retrieval depths, backends, or
  prompts.
- The corpus is small by design: it makes retrieval trivially perfect and
  isolates the generation layer as the thing to improve. Scaling the corpus
  is where embeddings + recall@k become non-trivial.
- No embeddings backend is included to keep deps minimal. The `Index` class
  is the swap point: replace `build_index`/`search` with an embeddings store
  and every metric still runs unchanged.
