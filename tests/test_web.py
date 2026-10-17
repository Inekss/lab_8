from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.models import DocMeta, Index, Posting
from findex.web.app import create_app
from findex.web.deps import get_index
from findex.web.settings import Settings


def _tiny_index(tmp_path: Path) -> Index:
    doc = tmp_path / "alpha.txt"
    doc.write_text("harry potter met quirrell in the dungeon", encoding="utf-8")
    postings = {
        "harry": [Posting(0, 1, (0,))],
        "potter": [Posting(0, 1, (1,))],
        "quirrell": [Posting(0, 1, (3,))],
        "dungeon": [Posting(0, 1, (6,))],
    }
    return Index(
        postings,
        {0: 7},
        {0: DocMeta(str(doc), "alpha")},
    )


@pytest.fixture
def client(tmp_path: Path):
    index = _tiny_index(tmp_path)
    app = create_app(
        Settings(index_path=tmp_path / "missing.pkl"),
        load_on_startup=False,
    )
    app.dependency_overrides[get_index] = lambda: index
    with TestClient(app) as c:
        app.state.index = index
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def client_no_index():
    app = create_app(
        Settings(index_path=Path("missing.pkl")),
        load_on_startup=False,
    )
    with TestClient(app) as c:
        yield c


def test_search_ok(client: TestClient) -> None:
    r = client.get("/search", params={"q": "harry", "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "harry"
    assert body["mode"] == "keyword"
    assert body["total"] >= 1
    assert body["results"][0]["doc_id"] == 0
    assert "title" in body["results"][0]


def test_search_semantic_without_embeddings_503(client: TestClient) -> None:
    r = client.get("/search", params={"q": "harry", "mode": "semantic"})
    assert r.status_code == 503


def test_search_validation_422(client: TestClient) -> None:
    r = client.get("/search", params={"q": "harry", "k": "abc"})
    assert r.status_code == 422
    assert "detail" in r.json()


def test_document_404(client: TestClient) -> None:
    r = client.get("/docs/999")
    assert r.status_code == 404


def test_document_ok(client: TestClient) -> None:
    r = client.get("/docs/0")
    assert r.status_code == 200
    assert r.json()["title"] == "alpha"
    assert "harry" in r.json()["text"]


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_503_when_unloaded(client_no_index: TestClient) -> None:
    r = client_no_index.get("/health")
    assert r.status_code == 503


def test_stats(client: TestClient) -> None:
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["num_docs"] == 1
    assert body["vocab_size"] == 4


def test_ui_home(client: TestClient) -> None:
    r = client.get("/", params={"q": "harry"})
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert b"findex" in r.content
    assert b"alpha" in r.content
