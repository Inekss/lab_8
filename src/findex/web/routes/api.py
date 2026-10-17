from __future__ import annotations

from time import time
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request, status

from findex.web.deps import EmbeddingsDep, IndexDep, SettingsDep
from findex.web.schemas import DocumentOut, HealthOut, SearchResponse, StatsOut
from findex.web.service import get_document, paginated_search

router = APIRouter(tags=["api"])


@router.get("/search", response_model=SearchResponse)
def search_endpoint(
    index: IndexDep,
    embeddings: EmbeddingsDep,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    k: Annotated[int, Query(ge=1, le=100)] = 10,
    scorer: Annotated[Literal["bm25", "tfidf"], Query()] = "bm25",
    page: Annotated[int, Query(ge=1)] = 1,
    mode: Annotated[Literal["keyword", "semantic", "hybrid"], Query()] = "keyword",
) -> SearchResponse:
    # plain ``def``: search is CPU-bound; FastAPI runs this in a thread pool
    if mode in ("semantic", "hybrid") and embeddings is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="embeddings not loaded; set EMBEDDINGS_PATH or run findex embed",
        )
    return paginated_search(
        index,
        q,
        k=k,
        scorer=scorer,
        page=page,
        page_size=k,
        mode=mode,
        embeddings=embeddings,
    )


@router.get("/docs/{doc_id}", response_model=DocumentOut)
def document_endpoint(doc_id: int, index: IndexDep) -> DocumentOut:
    doc = get_document(index, doc_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
    return doc


@router.get("/stats", response_model=StatsOut)
def stats_endpoint(
    request: Request,
    index: IndexDep,
    settings: SettingsDep,
    embeddings: EmbeddingsDep,
) -> StatsOut:
    started = getattr(request.app.state, "started_at", time())
    return StatsOut(
        num_docs=index.num_docs,
        vocab_size=len(index),
        uptime_s=round(time() - started, 3),
        index_path=str(settings.index_path),
        embeddings_loaded=embeddings is not None,
    )


@router.get("/health", response_model=HealthOut)
def health_endpoint(request: Request) -> HealthOut:
    if getattr(request.app.state, "index", None) is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="index not loaded",
        )
    return HealthOut(status="ok")
