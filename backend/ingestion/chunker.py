"""Section-aware chunking (spec section 5): chunk boundaries follow the
document's own structure (labeled abstract sections, then sentence groups)
rather than an arbitrary fixed character window. A section is only further
split when it exceeds max_chunk_chars, and splits happen on sentence
boundaries so no chunk cuts a sentence in half.
"""
from __future__ import annotations

import re

from backend.config.settings import settings
from backend.ingestion.parser import clean_text, detect_sections
from backend.models.evidence import EvidenceChunk, EvidenceDocument

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _pack_sentences(sentences: list[str], max_chars: int, min_chars: int) -> list[str]:
    """Greedily pack consecutive sentences into chunks up to max_chars,
    merging a trailing too-short fragment into the previous chunk."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        if current and current_len + len(sent) + 1 > max_chars:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += len(sent) + 1
    if current:
        chunks.append(" ".join(current))

    if len(chunks) > 1 and len(chunks[-1]) < min_chars:
        last = chunks.pop()
        chunks[-1] = chunks[-1] + " " + last
    return chunks


def chunk_document(doc: EvidenceDocument) -> list[EvidenceChunk]:
    cleaned = clean_text(doc.text)
    sections = detect_sections(cleaned)
    max_chars = settings.max_chunk_chars
    min_chars = settings.min_chunk_chars

    chunks: list[EvidenceChunk] = []
    idx = 0
    for section_name, section_text in sections:
        if len(section_text) <= max_chars:
            pieces = [section_text]
        else:
            pieces = _pack_sentences(split_sentences(section_text), max_chars, min_chars)

        for piece in pieces:
            if len(piece) < min_chars and len(sections) > 1 and len(pieces) == 1:
                # Very short standalone section (e.g. a one-line "CONCLUSIONS:")
                # is still kept - it's real content, just short.
                pass
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"{doc.document_id}_chunk_{idx}",
                    document_id=doc.document_id,
                    text=piece,
                    title=doc.title,
                    section=section_name,
                    publication_year=doc.year,
                    source=doc.source,
                    pmid=doc.pmid,
                    doi=doc.doi,
                    url=doc.url,
                )
            )
            idx += 1

    return chunks


def chunk_documents(docs: list[EvidenceDocument]) -> list[EvidenceChunk]:
    all_chunks: list[EvidenceChunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks
