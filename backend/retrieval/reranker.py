"""Cross-encoder reranking (spec section 8): takes the fused hybrid candidate
pool and produces the final top-k evidence set using a query-document
cross-encoder, which scores each (query, passage) pair jointly and is
generally more accurate than bi-encoder/BM25 scores alone.

Primary model: ncbi/MedCPT-Cross-Encoder, purpose-built for PubMed query/doc
relevance. Falls back to a general-purpose MS MARCO cross-encoder if the
primary model can't be loaded.
"""
from __future__ import annotations

import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.config.settings import settings
from backend.retrieval.hybrid import FusionCandidate

logger = logging.getLogger(__name__)

_reranker_cache: dict[str, tuple] = {}


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None):
        primary = model_name or settings.reranker_model
        self.model_name, self.tokenizer, self.model, self._mode = self._load(primary)

    def _load(self, primary: str):
        if primary in _reranker_cache:
            tok, mdl, mode = _reranker_cache[primary]
            return primary, tok, mdl, mode
        try:
            tokenizer = AutoTokenizer.from_pretrained(primary)
            model = AutoModelForSequenceClassification.from_pretrained(primary)
            model.eval()
            _reranker_cache[primary] = (tokenizer, model, "seq_cls")
            return primary, tokenizer, model, "seq_cls"
        except Exception as exc:  # pragma: no cover - network/model availability
            fallback = settings.reranker_fallback_model
            logger.warning(
                "Failed to load reranker '%s' (%s); falling back to '%s'",
                primary, exc, fallback,
            )
            if fallback in _reranker_cache:
                tok, mdl, mode = _reranker_cache[fallback]
                return fallback, tok, mdl, mode
            tokenizer = AutoTokenizer.from_pretrained(fallback)
            model = AutoModelForSequenceClassification.from_pretrained(fallback)
            model.eval()
            _reranker_cache[fallback] = (tokenizer, model, "seq_cls")
            return fallback, tokenizer, model, "seq_cls"

    @torch.no_grad()
    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        inputs = self.tokenizer(
            [query] * len(passages), passages,
            padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
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
