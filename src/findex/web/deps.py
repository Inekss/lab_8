from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from findex.embed import EmbeddingStore
from findex.models import Index
from findex.web.settings import Settings, get_settings as _load_settings


def get_settings(request: Request) -> Settings:
    return getattr(request.app.state, "settings", None) or _load_settings()


def get_index(request: Request) -> Index:
    index = getattr(request.app.state, "index", None)
    if index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="index not loaded",
        )
    return index


def get_embeddings(request: Request) -> EmbeddingStore | None:
    return getattr(request.app.state, "embeddings", None)


SettingsDep = Annotated[Settings, Depends(get_settings)]
IndexDep = Annotated[Index, Depends(get_index)]
EmbeddingsDep = Annotated[EmbeddingStore | None, Depends(get_embeddings)]
