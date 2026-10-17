from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SearchResultOut(BaseModel):
    doc_id: int
    title: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    query: str
    total: int
    page: int
    page_size: int
    took_ms: float
    mode: Literal["keyword", "semantic", "hybrid"] = "keyword"
    results: list[SearchResultOut]


class DocumentOut(BaseModel):
    doc_id: int
    title: str
    path: str
    text: str


class StatsOut(BaseModel):
    num_docs: int
    vocab_size: int
    uptime_s: float
    index_path: str
    embeddings_loaded: bool = False


class HealthOut(BaseModel):
    status: Literal["ok", "unavailable"] = "ok"
    detail: str | None = None


class ErrorOut(BaseModel):
    detail: str
    request_id: str | None = None


ScorerParam = Literal["bm25", "tfidf"]
ModeParam = Literal["keyword", "semantic", "hybrid"]
