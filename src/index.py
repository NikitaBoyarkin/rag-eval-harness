"""Indexing: load corpus docs into chunks + a TF-IDF matrix.

Each short doc is one chunk (title + body). TF-IDF retrieval is dependency-light
and fully offline; an embeddings backend can be dropped in later without
changing the eval harness.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config as C


@dataclass
class Index:
    doc_ids: list[str]
    texts: list[str]
    vectorizer: TfidfVectorizer
    matrix: np.ndarray

    def __len__(self):
        return len(self.doc_ids)


def _clean(md: str) -> str:
    lines = [l for l in md.splitlines() if not l.lstrip().startswith("#")]
    md = " ".join(lines)
    md = re.sub(r"\s+", " ", md).strip()
    return md


def build_index() -> Index:
    docs = sorted(C.DOCS.glob("*.md"))
    doc_ids, texts = [], []
    for p in docs:
        doc_ids.append(p.stem)
        texts.append(_clean(p.read_text()))
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english")
    mat = vec.fit_transform(texts).toarray()
    return Index(doc_ids=doc_ids, texts=texts, vectorizer=vec, matrix=mat)


def search(index: Index, query: str, top_k: int = C.TOP_K) -> list[tuple[str, float]]:
    q = index.vectorizer.transform([query]).toarray()
    sims = cosine_similarity(q, index.matrix)[0]
    order = np.argsort(sims)[::-1][:top_k]
    return [(index.doc_ids[i], float(sims[i])) for i in order]
