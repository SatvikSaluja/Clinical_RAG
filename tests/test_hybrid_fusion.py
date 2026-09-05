from backend.models.evidence import EvidenceChunk
from backend.retrieval.hybrid import fuse, merge_candidate_lists


def _chunk(cid):
    return EvidenceChunk(chunk_id=cid, document_id=cid, text="x", title="t")


def test_weighted_fusion_alpha_1_equals_bm25_order():
    bm25 = [(_chunk("a"), 10.0), (_chunk("b"), 5.0)]
    dense = [(_chunk("b"), 0.99), (_chunk("a"), 0.1)]  # dense disagrees with bm25
    fused = fuse(bm25, dense, alpha=1.0, method="weighted")
    assert [c.chunk.chunk_id for c in fused] == ["a", "b"]


def test_weighted_fusion_alpha_0_equals_dense_order():
    bm25 = [(_chunk("a"), 10.0), (_chunk("b"), 5.0)]
    dense = [(_chunk("b"), 0.99), (_chunk("a"), 0.1)]
    fused = fuse(bm25, dense, alpha=0.0, method="weighted")
    assert [c.chunk.chunk_id for c in fused] == ["b", "a"]


def test_fusion_combines_candidates_present_in_only_one_list():
    bm25 = [(_chunk("only_bm25"), 5.0)]
    dense = [(_chunk("only_dense"), 0.8)]
    fused = fuse(bm25, dense, alpha=0.5, method="weighted")
    ids = {c.chunk.chunk_id for c in fused}
    assert ids == {"only_bm25", "only_dense"}


def test_rrf_fusion_rewards_appearing_in_both_lists():
    bm25 = [(_chunk("both"), 10.0), (_chunk("bm25_only"), 9.0)]
    dense = [(_chunk("both"), 0.9), (_chunk("dense_only"), 0.8)]
    fused = fuse(bm25, dense, method="rrf")
    assert fused[0].chunk.chunk_id == "both"  # ranked in both lists -> highest RRF score


def test_merge_candidate_lists_unions_without_displacing_top_hits():
    # Primary (question-only) search's top hit must survive the merge even
    # though the secondary (patient-context) search ranks other things higher.
    primary = [(_chunk("q_top"), 10.0), (_chunk("shared"), 5.0)]
    secondary = [(_chunk("context_extra"), 20.0), (_chunk("shared"), 6.0)]
    merged = merge_candidate_lists(primary, secondary)
    ids = [c.chunk_id for c, _ in merged]
    assert "q_top" in ids  # never dropped just because secondary search scored other things higher
    assert "context_extra" in ids  # secondary search still contributes new candidates


def test_merge_candidate_lists_keeps_best_score_for_duplicate_chunk():
    primary = [(_chunk("dup"), 5.0)]
    secondary = [(_chunk("dup"), 9.0)]
    merged = merge_candidate_lists(primary, secondary)
    assert merged[0] == (merged[0][0], 9.0)


def test_merge_candidate_lists_respects_top_k():
    primary = [(_chunk("a"), 3.0), (_chunk("b"), 2.0)]
    secondary = [(_chunk("c"), 1.0)]
    merged = merge_candidate_lists(primary, secondary, top_k=2)
    assert len(merged) == 2
