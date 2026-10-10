"""Config for the RAG eval harness."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).parent.parent
DOCS = ROOT / "data" / "docs"
EVAL = ROOT / "data" / "eval.jsonl"
REPORTS = ROOT / "reports"

TOP_K = 5          # default retrieval depth
EVAL_K = 5         # retrieval metrics depth in eval
SEED = 42
MODEL = "claude-haiku-4-5-20251001"   # used only if ANTHROPIC_API_KEY is set
