"""Dense (embedding-based) retrieval (spec section 6B), backed by FAISS and a
configurable biomedical sentence-transformers model. Falls back to a small
general-purpose model if the primary biomedical model can't be loaded (e.g.
no network at query time), so the system degrades gracefully rather than
crashing.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config.settings import settings
from backend.models.evidence import EvidenceChunk

logger = logging.getLogger(__name__)

_model_cache: dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str | None = None) -> tuple[SentenceTransformer, str]:
    """Load (and cache) the configured embedding model, falling back to the
    configured fallback model on failure. Returns (model, actual_model_name).
    """
    primary = model_name or settings.dense_embedding_model
    if primary in _model_cache:
        return _model_cache[primary], primary
    try:
        model = SentenceTransformer(primary)
        _model_cache[primary] = model
        return model, primary
    except Exception as exc:  # pragma: no cover - network/model availability
        fallback = settings.dense_embedding_fallback_model
        logger.warning(
            "Failed to load embedding model '%s' (%s); falling back to '%s'",
            primary, exc, fallback,
        )
        if fallback in _model_cache:
            return _model_cache[fallback], fallback
        model = SentenceTransformer(fallback)
        _model_cache[fallback] = model
        return model, fallback


class DenseIndex:
    def __init__(self, chunks: list[EvidenceChunk], embeddings: np.ndarray, model_name: str):
        self.chunks = chunks
        self.model_name = model_name
        self.dim = embeddings.shape[1]
        # Normalize + inner product = cosine similarity
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

    @classmethod
    def build(cls, chunks: list[EvidenceChunk], model_name: str | None = None, batch_size: int = 64) -> "DenseIndex":
        model, actual_model_name = get_embedding_model(model_name)
        texts = [f"{c.title}. {c.text}" for c in chunks]
        embeddings = model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True,
        ).astype("float32")
        return cls(chunks, embeddings, actual_model_name)

    def search(self, query: str, top_k: int = 20) -> list[tuple[EvidenceChunk, float]]:
        model, _ = get_embedding_model(self.model_name)
        q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / "faiss.index"))
        meta = {
            "model_name": self.model_name,
            "chunks": [c.model_dump() for c in self.chunks],
        }
        (dir_path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, dir_path: Path) -> "DenseIndex":
        meta = json.loads((dir_path / "meta.json").read_text())
        chunks = [EvidenceChunk(**c) for c in meta["chunks"]]
        obj = cls.__new__(cls)
        obj.chunks = chunks
        obj.model_name = meta["model_name"]
        obj.index = faiss.read_index(str(dir_path / "faiss.index"))
        obj.dim = obj.index.d
        return obj
