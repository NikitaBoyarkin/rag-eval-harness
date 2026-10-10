"""RAG eval harness CLI.

Usage:
    uv run python run.py query "How is CUPED theta computed?"
    uv run python run.py eval
    uv run python run.py eval --k 5 --llm
"""
from __future__ import annotations

import argparse
import json
import pathlib

from src.index import build_index, search
from src.generate import generate
from src.eval import run_eval
from src import config as C


def cmd_query(args) -> None:
    index = build_index()
    g = generate(args.text, index, top_k=args.k, force_offline=args.offline)
    print(f"\nQ: {args.text}\n")
    print(f"A: {g.answer}\n")
    print(f"sources (used_llm={g.used_llm}):")
    for did, score in search(index, args.text, top_k=args.k):
        print(f"  {did:<14} {score:.3f}")


def cmd_eval(args) -> None:
    C.REPORTS.mkdir(exist_ok=True)
    r = run_eval(force_offline=not args.llm, k=args.k)
    s = r["summary"]
    print(f"\n=== Eval (n={s['n']}, k={s['k']}, llm_used={s['llm_used']}) ===")
    print("Retrieval:")
    for m, v in s["retrieval"].items():
        print(f"  {m:<14}{v}")
    print("Generation:")
    for m, v in s["generation"].items():
        print(f"  {m:<14}{v}")

    out = C.REPORTS / "metrics.json"
    out.write_text(json.dumps(s, indent=2))
    r["generation"].to_csv(C.REPORTS / "generation_per_question.csv", index=False)
    r["retrieval"][["id", "recall@k", "mrr", "precision@k", "gold", "retrieved"]].to_csv(
        C.REPORTS / "retrieval_per_question.csv", index=False)
    print(f"\n[saved {out}]")


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG eval harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("query", help="answer a question")
    q.add_argument("text")
    q.add_argument("--k", type=int, default=C.TOP_K)
    q.add_argument("--offline", action="store_true", help="force extractive fallback")
    q.set_defaults(func=cmd_query)

    e = sub.add_parser("eval", help="run the eval suite")
    e.add_argument("--k", type=int, default=C.EVAL_K)
    e.add_argument("--llm", action="store_true", help="use the LLM if ANTHROPIC_API_KEY is set")
    e.set_defaults(func=cmd_eval)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
