from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


def read_text(path: Path) -> str:
    # utf-8 first; a few books are cp1251 and fail on the first byte
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1251")


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: int
    path: Path
    text: str
    title: str


class CorpusReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def iter_documents(self) -> Iterator[str]:
        for _doc_id, _path, text, _title in self._iter():
            yield text

    def iter_docs(self) -> Iterator[tuple[int, Path, str]]:
        # sorted paths keep doc_id stable; ids only increase, so postings stay in order
        for doc_id, path, text, _title in self._iter():
            yield doc_id, path, text

    def iter_documents_meta(self) -> Iterator[Document]:
        for doc_id, path, text, title in self._iter():
            yield Document(doc_id, path, text, title)

    def _iter(self) -> Iterator[tuple[int, Path, str, str]]:
        if self.path.is_file() and self.path.suffix.lower() == ".jsonl":
            yield from self._iter_jsonl()
            return
        for doc_id, file in enumerate(sorted(self.path.glob("*.txt"))):
            yield doc_id, file, read_text(file), file.stem

    def _iter_jsonl(self) -> Iterator[tuple[int, Path, str, str]]:
        with self.path.open(encoding="utf-8") as fh:
            doc_id = 0
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                url = rec.get("url") or f"doc:{doc_id}"
                title = rec.get("title") or url
                text = rec.get("text") or ""
                # Path(url) keeps a stable string identity for DocMeta.path
                yield doc_id, Path(url), text, title
                doc_id += 1
