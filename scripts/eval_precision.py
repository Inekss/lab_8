"""Precision@5 for keyword / semantic / hybrid → docs/precision_at_5.md"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from findex.embed import load_embeddings
from findex.score import BM25
from findex.search import EVAL, EVAL_PARAPHRASE, search
from findex.store import load_pickle

K = 5


def p_at_5(hits, title: str) -> float:
    return sum(h.title == title for h in hits) / K


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index.pkl")
    ap.add_argument("--embeddings", default="embeddings.npy")
    args = ap.parse_args()

    index = load_pickle(Path(args.index))
    store = load_embeddings(Path(args.embeddings))
    rows = []
    for kind, pairs in (("original", EVAL), ("paraphrase", EVAL_PARAPHRASE)):
        for query, title in pairs:
            modes = {}
            for mode in ("keyword", "semantic", "hybrid"):
                hits = search(
                    index,
                    query,
                    BM25(),
                    K,
                    mode=mode,
                    embeddings=store,
                    with_snippet=False,
                )
                modes[mode] = p_at_5(hits, title)
            rows.append((query, kind, title, modes))

    lines = [
        "# Precision@5 — keyword / semantic / hybrid",
        "",
        "| query | kind | keyword | semantic | hybrid | relevant |",
        "|---|---|---:|---:|---:|---|",
    ]
    means = {"keyword": 0.0, "semantic": 0.0, "hybrid": 0.0}
    for query, kind, title, modes in rows:
        lines.append(
            f"| `{query}` | {kind} | {modes['keyword']:.2f} | "
            f"{modes['semantic']:.2f} | {modes['hybrid']:.2f} | `{title}` |"
        )
        for m in means:
            means[m] += modes[m]
    n = len(rows)
    lines.append(
        f"| **mean** | | {means['keyword']/n:.2f} | {means['semantic']/n:.2f} | "
        f"{means['hybrid']/n:.2f} | |"
    )
    lines += [
        "",
        "## What this shows",
        "",
        "Original keyword queries (character names) keep high keyword precision. "
        "Paraphrase queries deliberately avoid document vocabulary, so BM25 drops; "
        "semantic recovers meaning via embeddings; hybrid (RRF) stays competitive on both sets.",
        "",
    ]
    out = ROOT / "docs" / "precision_at_5.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
