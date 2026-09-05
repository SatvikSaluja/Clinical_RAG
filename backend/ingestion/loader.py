"""Document ingestion: fetch real PubMed abstracts via NCBI E-utilities and
cache them locally. The local cache in `data/raw/pubmed/*.json` IS the
documented offline fallback: `load_cached_documents` never touches the
network, so once a topic has been fetched once, corpus building works fully
offline. No papers/PMIDs/DOIs/URLs are ever fabricated - everything here
comes verbatim from what NCBI actually returns.
"""
from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from backend.config.settings import RAW_DIR, settings
from backend.models.evidence import EvidenceDocument

logger = logging.getLogger(__name__)

PUBMED_DIR = RAW_DIR / "pubmed"

# Topic queries chosen to cover the conditions present in the synthetic
# patients (type 2 diabetes, hypertension, lipids, thyroid, anemia, CKD).
DEFAULT_TOPIC_QUERIES: dict[str, str] = {
    "type2_diabetes_hba1c": "type 2 diabetes mellitus HbA1c glycemic control",
    "metformin": "metformin type 2 diabetes treatment",
    "diabetes_complications": "type 2 diabetes complications nephropathy retinopathy",
    "hypertension_management": "hypertension management blood pressure control guideline",
    "ace_inhibitors": "ACE inhibitor angiotensin blood pressure treatment",
    "lipid_panel_statins": "LDL cholesterol statin therapy cardiovascular risk",
    "hyperlipidemia": "hyperlipidemia dyslipidemia treatment guideline",
    "hypothyroidism": "hypothyroidism TSH levothyroxine treatment",
    "thyroid_function_tests": "thyroid function tests TSH free T4 interpretation",
    "anemia_iron_deficiency": "iron deficiency anemia diagnosis treatment",
    "chronic_kidney_disease": "chronic kidney disease eGFR staging management",
    "ckd_diabetes": "diabetic nephropathy chronic kidney disease diabetes",
    "obesity_metabolic_syndrome": "obesity metabolic syndrome insulin resistance",
    "blood_glucose_monitoring": "self-monitoring blood glucose HbA1c correlation",
    "cardiovascular_risk_diabetes": "cardiovascular risk type 2 diabetes management",
    "smoking_cessation": "smoking cessation cardiovascular risk reduction",
    "physical_activity_diabetes": "physical activity exercise glycemic control diabetes",
    "medication_adherence": "medication adherence chronic disease management outcomes",
}


class PubMedClient:
    """Thin wrapper around NCBI E-utilities (esearch + efetch)."""

    def __init__(self, contact_email: str | None = None, base_url: str | None = None):
        self.contact_email = contact_email or settings.ncbi_contact_email
        self.base_url = base_url or settings.ncbi_eutils_base
        self.session = requests.Session()

    def search(self, query: str, retmax: int = 20) -> list[str]:
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": retmax,
            "retmode": "json",
            "sort": "relevance",
            "email": self.contact_email,
            "tool": "clinical-rag-research-prototype",
        }
        resp = self.session.get(f"{self.base_url}/esearch.fcgi", params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def fetch(self, pmids: list[str]) -> list[EvidenceDocument]:
        if not pmids:
            return []
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
            "email": self.contact_email,
            "tool": "clinical-rag-research-prototype",
        }
        resp = self.session.get(f"{self.base_url}/efetch.fcgi", params=params, timeout=30)
        resp.raise_for_status()
        return _parse_pubmed_xml(resp.text)

    def fetch_topic(self, topic: str, query: str, retmax: int = 20) -> list[EvidenceDocument]:
        pmids = self.search(query, retmax=retmax)
        time.sleep(0.34)  # respect NCBI's ~3 requests/sec rate limit
        docs = self.fetch(pmids)
        for d in docs:
            d.query_topic = topic
        time.sleep(0.34)
        return docs


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _parse_pubmed_xml(xml_text: str) -> list[EvidenceDocument]:
    docs: list[EvidenceDocument] = []
    root = ET.fromstring(xml_text)
    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        if medline is None:
            continue
        pmid = _text(medline.find("PMID"))
        art = medline.find("Article")
        if art is None:
            continue
        title = _text(art.find("ArticleTitle"))

        abstract_parts = []
        abstract_el = art.find("Abstract")
        if abstract_el is not None:
            for ab_text in abstract_el.findall("AbstractText"):
                label = ab_text.get("Label")
                piece = _text(ab_text)
                if not piece:
                    continue
                abstract_parts.append(f"{label}: {piece}" if label else piece)
        text = "\n".join(abstract_parts).strip()
        if not text:
            continue  # skip entries with no usable abstract text

        authors = []
        author_list = art.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last = _text(author.find("LastName"))
                fore = _text(author.find("ForeName"))
                if last:
                    authors.append(f"{fore} {last}".strip())

        year = None
        pub_date = art.find("Journal/JournalIssue/PubDate/Year")
        if pub_date is not None and pub_date.text:
            try:
                year = int(pub_date.text)
            except ValueError:
                year = None

        journal = _text(art.find("Journal/Title"))

        doi = None
        for eid in article.findall(".//ArticleId"):
            if eid.get("IdType") == "doi":
                doi = _text(eid)

        docs.append(
            EvidenceDocument(
                document_id=f"pmid_{pmid}",
                title=title or "(untitled)",
                authors=authors,
                year=year,
                source="PubMed",
                pmid=pmid,
                doi=doi,
                journal=journal or None,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
                text=text,
            )
        )
    return docs


def fetch_and_cache_corpus(
    topics: dict[str, str] | None = None,
    retmax_per_topic: int = 20,
    force_refetch: bool = False,
) -> list[EvidenceDocument]:
    """Fetch each topic query from PubMed (unless already cached) and write
    one JSON file per topic under data/raw/pubmed/. Returns the full combined
    document list, de-duplicated by PMID.
    """
    topics = topics or DEFAULT_TOPIC_QUERIES
    PUBMED_DIR.mkdir(parents=True, exist_ok=True)
    client = PubMedClient()
    all_docs: dict[str, EvidenceDocument] = {}

    for topic, query in topics.items():
        cache_path = PUBMED_DIR / f"{topic}.json"
        if cache_path.exists() and not force_refetch:
            logger.info("Loading cached topic '%s' from %s", topic, cache_path)
            docs = [EvidenceDocument(**d) for d in json.loads(cache_path.read_text())]
        else:
            logger.info("Fetching topic '%s' from PubMed: %s", topic, query)
            try:
                docs = client.fetch_topic(topic, query, retmax=retmax_per_topic)
            except Exception as exc:  # pragma: no cover - network fallback
                logger.warning("PubMed fetch failed for topic '%s' (%s); skipping", topic, exc)
                docs = []
            cache_path.write_text(json.dumps([d.model_dump() for d in docs], indent=2))
        for d in docs:
            all_docs[d.document_id] = d

    return list(all_docs.values())


def load_cached_documents() -> list[EvidenceDocument]:
    """Load whatever is already cached in data/raw/pubmed/ with NO network
    access. This is the offline-fallback path used by corpus building when
    PubMed is unreachable or the caller wants a fully deterministic run.
    """
    docs: dict[str, EvidenceDocument] = {}
    for path in sorted(PUBMED_DIR.glob("*.json")):
        for d in json.loads(path.read_text()):
            doc = EvidenceDocument(**d)
            docs[doc.document_id] = doc
    return list(docs.values())
