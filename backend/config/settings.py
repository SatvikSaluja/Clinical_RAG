"""Central configuration for the Clinical RAG system.

Every tunable parameter referenced throughout the spec (model names, fusion
weights, top-k sizes, reliability-gate behavior, LLM provider) lives here and
is overridable via environment variables / a `.env` file. Nothing is
hard-coded in the retrieval/generation/verification modules themselves -
they all import `settings` from this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = PROCESSED_DIR / "index"
PATIENTS_DIR = DATA_DIR / "patients"
EVALUATION_DIR = DATA_DIR / "evaluation"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ---
    llm_provider: Literal["local", "openai"] = "local"
    local_llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    llm_max_new_tokens: int = 512
    llm_temperature: float = 0.0  # 0.0 = greedy decoding (deterministic, reproducible)

    # --- Retrieval models ---
    dense_embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    dense_embedding_fallback_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "ncbi/MedCPT-Cross-Encoder"
    reranker_fallback_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    # "local" (default, PyTorch models above), "remote" (Gemini/OpenAI-
    # compatible embeddings API - no PyTorch loaded), or "none" (skip dense
    # retrieval and semantic patient-context ranking entirely - BM25 +
    # keyword-overlap only, zero embedding calls). "none" is for hosts
    # where even a free embeddings API's rate limits are too unreliable to
    # depend on at deploy time - see README "Deploying on limited RAM".
    embedding_provider: Literal["local", "remote", "none"] = "local"
    remote_embedding_model: str = "gemini-embedding-001"
    reranker_provider: Literal["local", "llm"] = "local"

    # --- Hybrid retrieval fusion ---
    hybrid_alpha: float = 0.5  # weight on normalized BM25 score; (1-alpha) on dense
    fusion_method: Literal["weighted", "rrf"] = "weighted"
    rrf_k: int = 60
    bm25_candidate_k: int = 20
    dense_candidate_k: int = 20
    rerank_top_k: int = 5

    # --- Chunking ---
    max_chunk_chars: int = 1200
    min_chunk_chars: int = 200

    # --- Patient-aware query construction ---
    max_patient_context_fields: int = 6
    patient_context_similarity_threshold: float = 0.25

    # --- Reliability gate ---
    reliability_gate_mode: Literal["flag", "remove", "rewrite"] = "flag"
    nli_support_threshold: float = 0.55
    nli_contradiction_threshold: float = 0.55
    use_llm_secondary_verifier: bool = True
    # Set false to skip loading the local NLI model (~740MB, the largest
    # model in the pipeline) and use the LLM verifier as the sole claim
    # verifier instead - for memory-constrained deployments.
    use_local_nli_verifier: bool = True

    # --- Ingestion ---
    ncbi_contact_email: str = "research-prototype@example.com"
    ncbi_eutils_base: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


settings = Settings()

for _dir in (RAW_DIR / "pubmed", PROCESSED_DIR, INDEX_DIR, PATIENTS_DIR, EVALUATION_DIR, RESULTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
