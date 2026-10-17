from findex.embed import chunk_text


def test_chunk_text_short() -> None:
    assert chunk_text("hello") == ["hello"]


def test_chunk_text_splits() -> None:
    text = ("word " * 200).strip()
    chunks = chunk_text(text, max_chars=80, overlap=10)
    assert len(chunks) > 1
    assert all(len(c) <= 90 for c in chunks)
