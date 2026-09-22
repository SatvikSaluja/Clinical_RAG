"""Cross-encoder reranking (spec section 8): takes the fused hybrid candidate
pool and produces the final top-k evidence set.

Two implementations, selected via `settings.reranker_provider`:
- CrossEncoderReranker ("local", default): a local PyTorch cross-encoder
  (ncbi/MedCPT-Cross-Encoder, purpose-built for PubMed query/doc relevance;
  falls back to a general MS MARCO cross-encoder if it can't be loaded).
- LLMReranker ("llm"): asks the configured LLM provider to score each
  candidate's relevance directly - no local model, so no PyTorch import at
  all. Used for memory-constrained deployments (see README "Deploying on
  limited RAM"), where loading PyTorch itself (~350MB+ RSS) doesn't fit a
  free-tier 512MB host.

All PyTorch/transformers imports are local to the functions that need
them, so importing this module - or using the LLM reranker - never loads
PyTorch into memory.
"""
from __future__ import annotations

import logging
import re

from backend.config.settings import settings
from backend.retrieval.hybrid import FusionCandidate

logger = logging.getLogger(__name__)

_reranker_cache: dict[str, tuple] = {}


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None):
        primary = model_name or settings.reranker_model
        self.model_name, self.tokenizer, self.model = self._load(primary)

    def _load(self, primary: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # lazy: pulls in torch

        if primary in _reranker_cache:
            tok, mdl = _reranker_cache[primary]
            return primary, tok, mdl
        try:
            tokenizer = AutoTokenizer.from_pretrained(primary)
            model = AutoModelForSequenceClassification.from_pretrained(primary)
            model.eval()
            _reranker_cache[primary] = (tokenizer, model)
            return primary, tokenizer, model
        except Exception as exc:  # pragma: no cover - network/model availability
            fallback = settings.reranker_fallback_model
            logger.warning(
                "Failed to load reranker '%s' (%s); falling back to '%s'",
                primary, exc, fallback,
            )
            if fallback in _reranker_cache:
                tok, mdl = _reranker_cache[fallback]
                return fallback, tok, mdl
            tokenizer = AutoTokenizer.from_pretrained(fallback)
            model = AutoModelForSequenceClassification.from_pretrained(fallback)
            model.eval()
            _reranker_cache[fallback] = (tokenizer, model)
            return fallback, tokenizer, model

    def score(self, query: str, passages: list[str]) -> list[float]:
        import torch  # lazy

        if not passages:
            return []
        inputs = self.tokenizer(
            [query] * len(passages), passages,
            padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
            if logits.shape[-1] == 1:
                scores = logits.squeeze(-1)
            else:
                # Some seq-cls rerankers output 2 logits (irrelevant/relevant);
                # use the softmax probability of the "relevant" class.
                scores = torch.softmax(logits, dim=-1)[:, -1]
        return scores.tolist()

    def rerank(
        self, query: str, candidates: list[FusionCandidate], top_k: int | None = None
    ) -> list[tuple[FusionCandidate, float]]:
        top_k = settings.rerank_top_k if top_k is None else top_k
        if not candidates:
            return []
        passages = [c.chunk.text for c in candidates]
        scores = self.score(query, passages)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


_SCORE_LINE_RE = re.compile(r"^\s*(\d+)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*$")


class LLMReranker:
    """Reranks by asking the LLM to score each candidate 0-10 for relevance
    to the query, one call per rerank (not per passage). No local model."""

    def __init__(self):
        self.model_name = "llm:" + settings.openai_model

    def rerank(
        self, query: str, candidates: list[FusionCandidate], top_k: int | None = None
    ) -> list[tuple[FusionCandidate, float]]:
        top_k = settings.rerank_top_k if top_k is None else top_k
        if not candidates:
            return []

        from backend.generation.generator import get_generator

        passages_block = "\n\n".join(
            f"[{i}] {c.chunk.text[:600]}" for i, c in enumerate(candidates)
        )
        system = (
            "You are a relevance-scoring assistant. Score how relevant each "
            "numbered passage is to the QUERY, from 0 (irrelevant) to 10 "
            "(directly answers it). Respond with exactly one line per "
            "passage, in the form 'N: score' (e.g. '0: 7'), nothing else."
        )
        user = f"QUERY: {query}\n\nPASSAGES:\n{passages_block}"

        scores = [0.0] * len(candidates)
        try:
            generator = get_generator()
            result = generator.generate(system, user)
            for line in result.text.splitlines():
                match = _SCORE_LINE_RE.match(line)
                if match:
                    idx, score = int(match.group(1)), float(match.group(2))
                    if 0 <= idx < len(candidates):
                        scores[idx] = score
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("LLM reranker failed (%s); falling back to hybrid_score order", exc)
            scores = [c.hybrid_score for c in candidates]

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


def get_reranker():
    """Factory: returns the configured reranker implementation."""
    if settings.reranker_provider == "llm":
        return LLMReranker()
    return CrossEncoderReranker()
