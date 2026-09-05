from backend.ingestion.chunker import chunk_document, split_sentences
from backend.models.evidence import EvidenceDocument


def test_split_sentences_basic():
    sents = split_sentences("First sentence. Second sentence! Third one?")
    assert sents == ["First sentence.", "Second sentence!", "Third one?"]


def test_structured_abstract_produces_section_labeled_chunks():
    doc = EvidenceDocument(
        document_id="doc1",
        title="Example",
        text="BACKGROUND: Diabetes is common. METHODS: We did a study. RESULTS: Glucose fell. CONCLUSIONS: It works.",
    )
    chunks = chunk_document(doc)
    sections = {c.section for c in chunks}
    assert "background" in sections
    assert "methods" in sections
    assert "results" in sections
    assert "conclusions" in sections
    for c in chunks:
        assert c.document_id == "doc1"
        assert c.title == "Example"


def test_unstructured_abstract_falls_back_to_single_section():
    doc = EvidenceDocument(document_id="doc2", title="Plain", text="Just a plain unlabeled abstract sentence.")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].section == "abstract"


def test_long_section_is_split_on_sentence_boundaries_not_mid_sentence():
    long_text = "BACKGROUND: " + " ".join([f"This is sentence number {i} in a long passage." for i in range(40)])
    doc = EvidenceDocument(document_id="doc3", title="Long", text=long_text)
    chunks = chunk_document(doc)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.strip().endswith((".", "!", "?"))
