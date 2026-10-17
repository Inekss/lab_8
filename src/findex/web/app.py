from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from time import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from findex.embed import load_embeddings
from findex.models import Index
from findex.store import load_json, load_pickle
from findex.web.routes import api_router, ui_router
from findex.web.settings import Settings, get_settings

logger = logging.getLogger("findex.web")

STATIC_DIR = Path(__file__).resolve().parent / "static"


def load_index(path: Path) -> Index:
    if not path.exists():
        raise FileNotFoundError(f"index not found: {path}")
    if path.suffix.lower() == ".json":
        return load_json(path)
    return load_pickle(path)


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def create_app(settings: Settings | None = None, *, load_on_startup: bool = True) -> FastAPI:
    """Build the FastAPI app.

    When ``settings`` is None, ``get_settings()`` is called at lifespan startup
    so ``INDEX_PATH`` set just before ``uvicorn.run`` is honored.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = settings if settings is not None else get_settings()
        app.state.settings = s
        app.state.started_at = time()
        app.state.index = None
        app.state.embeddings = None
        logging.basicConfig(
            level=getattr(logging, s.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )
        if load_on_startup:
            app.state.index = load_index(s.index_path)
            logger.info(
                "loaded index from %s (%s docs)",
                s.index_path,
                app.state.index.num_docs,
            )
            emb = s.embeddings_path
            if emb is not None and Path(emb).is_file():
                app.state.embeddings = load_embeddings(Path(emb))
                logger.info(
                    "loaded embeddings from %s (%s chunks)",
                    emb,
                    app.state.embeddings.n_chunks,
                )
        yield
        app.state.index = None
        app.state.embeddings = None

    app = FastAPI(
        title="findex",
        version="1.0.0",
        description="Search API and UI for the findex engine (keyword / semantic / hybrid)",
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None)
        logger.exception("unhandled error request_id=%s", request_id)
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error", "request_id": request_id},
        )

    app.include_router(api_router)
    app.include_router(ui_router)
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


app = create_app()
