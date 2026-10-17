from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from findex.documents import read_text
from findex.models import Index


@dataclass(frozen=True)
class EmbeddingStore:
    vectors: np.ndarray  # float32 [N, d], L2-normalized
    chunk_doc_ids: np.ndarray  # int32 [N]
    model_name: str
    path: Path

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0

    @property
    def n_chunks(self) -> int:
        return int(self.vectors.shape[0])


def chunk_text(text: str, *, max_chars: int = 800, overlap: int = 100) -> list[str]:
    """Split document into overlapping character windows (≈200–500 tokens)."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # prefer breaking on paragraph / sentence
        window = text[start:end]
        if end < len(text):
            for sep in ("\n\n", "\n", ". ", " "):
                at = window.rfind(sep)
                if at > max_chars // 3:
                    end = start + at + len(sep)
                    window = text[start:end]
                    break
        piece = window.strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def iter_doc_chunks(index: Index) -> Iterator[tuple[int, str]]:
    for doc_id in sorted(index.doc_meta):
        meta = index.doc_meta[doc_id]
        path = Path(meta.path)
        text = read_text(path) if path.is_file() else ""
        for chunk in chunk_text(text):
            yield doc_id, chunk


def build_embeddings(
    index: Index,
    *,
    model_name: str = "all-MiniLM-L6-v2",
    out: Path,
    batch_size: int = 32,
) -> EmbeddingStore:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    doc_ids: list[int] = []
    texts: list[str] = []
    for doc_id, chunk in iter_doc_chunks(index):
        doc_ids.append(doc_id)
        texts.append(chunk)

    if not texts:
        raise ValueError("no chunks to embed")

    vectors_list: list[np.ndarray] = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("embed", total=len(texts))
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            emb = model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            vectors_list.append(np.asarray(emb, dtype=np.float32))
            progress.advance(task, len(batch))

    vectors = np.vstack(vectors_list).astype(np.float32)
    chunk_doc_ids = np.asarray(doc_ids, dtype=np.int32)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, vectors)
    meta_path = out.with_suffix(".meta.json")
    ids_path = out.with_name(out.stem + "_doc_ids.npy")
    np.save(ids_path, chunk_doc_ids)
    meta_path.write_text(
        json.dumps(
            {
                "model": model_name,
                "n_chunks": int(vectors.shape[0]),
                "dim": int(vectors.shape[1]),
                "vectors": str(out.name),
                "chunk_doc_ids": str(ids_path.name),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return EmbeddingStore(vectors, chunk_doc_ids, model_name, out)


def load_embeddings(path: Path) -> EmbeddingStore:
    path = Path(path)
    meta_path = path.with_suffix(".meta.json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ids_path = path.with_name(meta["chunk_doc_ids"])
        model_name = meta.get("model", "unknown")
    else:
        ids_path = path.with_name(path.stem + "_doc_ids.npy")
        model_name = "unknown"
    # mmap keeps the ~13 MiB matrix on disk until touched (helps low-RAM hosts)
    vectors = np.load(path, mmap_mode="r")
    chunk_doc_ids = np.load(ids_path, mmap_mode="r")
    return EmbeddingStore(vectors, chunk_doc_ids, model_name, path)


_MODEL_CACHE: dict[str, object] = {}


def _limit_torch_threads() -> None:
    import os

    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def encode_query(text: str, model_name: str) -> np.ndarray:
    """Encode one query. Loads MiniLM once (~400+ MiB RSS) — avoid on tiny RAM hosts."""
    _limit_torch_threads()
    from sentence_transformers import SentenceTransformer

    model = _MODEL_CACHE.get(model_name)
    if model is None:
        try:
            import torch

            torch.set_num_threads(1)
        except Exception:
            pass
        model = SentenceTransformer(model_name, device="cpu")
        _MODEL_CACHE[model_name] = model
    vec = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vec, dtype=np.float32)

