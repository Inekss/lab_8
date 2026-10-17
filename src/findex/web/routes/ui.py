from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from findex.web.deps import EmbeddingsDep, IndexDep
from findex.web.service import get_document, highlight_snippet, paginated_search

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.filters["highlight"] = highlight_snippet

router = APIRouter(tags=["ui"], include_in_schema=False)


@router.get("/", response_class=HTMLResponse)
def home(
    request: Request,
    index: IndexDep,
    embeddings: EmbeddingsDep,
    q: Annotated[str | None, Query(max_length=200)] = None,
    k: Annotated[int, Query(ge=1, le=100)] = 10,
    scorer: Annotated[Literal["bm25", "tfidf"], Query()] = "bm25",
    mode: Annotated[Literal["keyword", "semantic", "hybrid"], Query()] = "keyword",
    page: Annotated[int, Query(ge=1)] = 1,
) -> HTMLResponse:
    ctx: dict = {
        "request": request,
        "q": q or "",
        "k": k,
        "scorer": scorer,
        "mode": mode,
        "page": page,
        "response": None,
        "total_pages": 0,
        "embeddings_ready": embeddings is not None,
        "error": None,
    }
    if q:
        if mode in ("semantic", "hybrid") and embeddings is None:
            ctx["error"] = "Embeddings not loaded. Run findex embed and set EMBEDDINGS_PATH."
        else:
            response = paginated_search(
                index,
                q,
                k=k,
                scorer=scorer,
                page=page,
                page_size=k,
                mode=mode,
                embeddings=embeddings,
            )
            ctx["response"] = response
            ctx["total_pages"] = (
                max(1, ceil(response.total / response.page_size)) if response.total else 0
            )
    return templates.TemplateResponse(request, "index.html", ctx)


@router.get("/doc/{doc_id}", response_class=HTMLResponse)
def doc_page(request: Request, doc_id: int, index: IndexDep) -> HTMLResponse:
    doc = get_document(index, doc_id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document not found")
    return templates.TemplateResponse(request, "doc.html", {"request": request, "doc": doc})
