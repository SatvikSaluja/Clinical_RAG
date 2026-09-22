"""Dense (embedding-based) retrieval (spec section 6B).

Two embedding providers, selected via `settings.embedding_provider`:
- "local" (default): a biomedical sentence-transformers model (PyTorch).
  Falls back to a small general-purpose model if the primary model can't be
  loaded, so the system degrades gracefully rather than crashing.
- "remote": Gemini's embeddings API via the OpenAI-compatible endpoint - no
  PyTorch/transformers import at all. Used for memory-constrained
  deployments (PyTorch alone uses ~350MB+ RSS before loading a single
  model, which doesn't fit a free-tier 512MB host) - see README "Deploying
  on limited RAM".

All heavy imports (sentence-transformers -> torch) are local to the
functions that need them, so importing this module - or using it in
"remote" mode - never pulls PyTorch into memory.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np

from backend.config.settings import settings
from backend.models.evidence import EvidenceChunk

logger = logging.getLogger(__name__)

_model_cache: dict[str, object] = {}


def get_embedding_model(model_name: str | None = None):
    """Load (and cache) the configured LOCAL embedding model, falling back
    to the configured fallback model on failure. Returns (model,
    actual_model_name). Only used when embedding_provider == "local".
    """
    from sentence_transformers import SentenceTransformer  # lazy: pulls in torch

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


def _encode_remote(texts: list[str], batch_size: int = 16) -> np.ndarray:
    """Embed texts via Gemini's OpenAI-compatible embeddings endpoint. No
    local model - just an HTTP call, so this never loads PyTorch."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=settings.remote_embedding_model, input=batch)
        vectors.extend(d.embedding for d in resp.data)
        if i + batch_size < len(texts):
            time.sleep(0.2)  # be gentle with the free-tier rate limit
    return np.array(vectors, dtype="float32")


def encode_texts(texts: list[str]) -> np.ndarray:
    """Embed texts using whichever provider is configured."""
    if settings.embedding_provider == "remote":
        return _encode_remote(texts)
    model, _ = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True).astype("float32")


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
        texts = [f"{c.title}. {c.text}" for c in chunks]
        if settings.embedding_provider == "remote":
            logger.info("Embedding %d chunks via remote provider '%s'", len(chunks), settings.remote_embedding_model)
            embeddings = _encode_remote(texts)
            actual_model_name = settings.remote_embedding_model
        else:
            model, actual_model_name = get_embedding_model(model_name)
            embeddings = model.encode(
                texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True,
            ).astype("float32")
        return cls(chunks, embeddings, actual_model_name)

    def search(self, query: str, top_k: int = 20) -> list[tuple[EvidenceChunk, float]]:
        q_emb = encode_texts([query])
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
