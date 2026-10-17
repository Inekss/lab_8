# syntax=docker/dockerfile:1
# Multi-stage build with uv (https://docs.astral.sh/uv/guides/integration/docker/)

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --group semantic

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --group semantic

# Index from the bundled corpus. Embeddings are copied from deploy/ (pre-built).
RUN mkdir -p /app/deploy \
    && uv run findex index --corpus src/resources --out-pkl /app/deploy/index.pkl --out-json /app/deploy/index.json \
    && if [ -f /app/embeddings.npy ]; then \
         cp /app/embeddings.npy /app/embeddings.meta.json /app/embeddings_doc_ids.npy /app/deploy/; \
       elif [ -f /app/deploy/embeddings.npy ]; then \
         true; \
       else \
         echo "WARNING: no embeddings.npy — semantic/hybrid will be unavailable"; \
       fi


FROM python:3.12-slim-bookworm AS runtime

RUN groupadd --system --gid 1000 findex \
    && useradd --system --uid 1000 --gid findex --home /app --shell /usr/sbin/nologin findex

WORKDIR /app

COPY --from=builder --chown=findex:findex /app/.venv /app/.venv
COPY --from=builder --chown=findex:findex /app/src /app/src
COPY --from=builder --chown=findex:findex /app/deploy /app/deploy
COPY --from=builder --chown=findex:findex /app/pyproject.toml /app/pyproject.toml

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    INDEX_PATH="/app/deploy/index.pkl" \
    EMBEDDINGS_PATH="/app/deploy/embeddings.npy" \
    LOG_LEVEL="INFO" \
    PORT="8000" \
    HF_HOME="/app/.cache/huggingface" \
    TRANSFORMERS_CACHE="/app/.cache/huggingface"

# Model downloads on first semantic query; cache under /app (writable by findex)
RUN mkdir -p /app/.cache/huggingface && chown -R findex:findex /app/.cache

USER findex
EXPOSE 8000

CMD ["sh", "-c", "uvicorn findex.web:app --host 0.0.0.0 --port ${PORT:-8000}"]
