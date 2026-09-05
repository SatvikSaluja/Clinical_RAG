"""FastAPI application entry point.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.config.settings import settings

app = FastAPI(
    title="Evidence-Grounded Clinical RAG (Research Prototype)",
    description=(
        "AI research prototype - NOT a medical device. Uses synthetic patient "
        "data and real public biomedical literature. Not for clinical use."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {
        "name": "Evidence-Grounded Clinical RAG System",
        "status": "ok",
        "disclaimer": "Research prototype. Not a medical device. Synthetic patient data only.",
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}
