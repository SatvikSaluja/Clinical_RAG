from backend.models.evidence import EvidenceChunk
from backend.retrieval.bm25 import BM25Index, tokenize


def _chunk(cid, text):
    return EvidenceChunk(chunk_id=cid, document_id=cid, text=text, title="t")


def test_tokenize_preserves_hyphenated_and_dotted_terms():
    tokens = tokenize("HbA1c is 7.4% in type-2 diabetes.")
    assert "hba1c" in tokens
    assert "7.4" in tokens
    assert "type-2" in tokens


def test_bm25_ranks_exact_term_match_first():
    chunks = [
        _chunk("c1", "Metformin reduces hepatic glucose production in type 2 diabetes."),
        _chunk("c2", "Unrelated passage about vitamin D deficiency and bone health."),
        _chunk("c3", "Metformin metformin metformin is the most studied biguanide."),
    ]
    index = BM25Index(chunks)
    results = index.search("metformin", top_k=3)
    assert results[0][0].chunk_id == "c3"  # highest term frequency for the exact query term
    ids = [c.chunk_id for c, _ in results]
    assert "c2" not in ids  # unrelated passage should not match "metformin" at all


def test_bm25_empty_index_returns_empty():
    index = BM25Index([])
    assert index.search("anything") == []
