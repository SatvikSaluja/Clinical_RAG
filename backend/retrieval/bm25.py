"""BM25 lexical retrieval (spec section 6A) - important for exact terminology
(drug names, biomarkers, abbreviations, numeric values) that dense embeddings
can blur. Implemented with `rank_bm25` (pure Python, no external service),
satisfying the spec's "BM25 implementation" option without requiring
Elasticsearch/OpenSearch.
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from backend.models.evidence import EvidenceChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/.]*")


def tokenize(text: str) -> list[str]:
    """Lowercase alnum tokenizer that keeps hyphens/slashes/dots inside a
    token so things like 'HbA1c', 'type-2', 'anti-inflammatory' and
    numeric-unit pairs like '7.4' stay intact - matters for drug names,
    biomarkers and numerical terminology per the spec.
    """
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self, chunks: list[EvidenceChunk]):
        self.chunks = chunks
        self._tokenized = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def search(self, query: str, top_k: int = 20) -> list[tuple[EvidenceChunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"chunks": [c.model_dump() for c in self.chunks]}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with path.open("rb") as f:
            data = pickle.load(f)
        chunks = [EvidenceChunk(**c) for c in data["chunks"]]
        return cls(chunks)

    @classmethod
    def from_jsonl(cls, chunks_path: Path) -> "BM25Index":
        chunks = []
        with chunks_path.open() as f:
            for line in f:
                chunks.append(EvidenceChunk(**json.loads(line)))
        return cls(chunks)
