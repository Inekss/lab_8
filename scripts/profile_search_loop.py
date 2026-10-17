"""Keep search busy so py-spy has time to attach (Windows)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from findex.score import BM25
from findex.search import search
from findex.store import load_pickle


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--index", default="index.pkl")
    p.add_argument("--query", default="harry potter magic")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()

    index = load_pickle(Path(args.index))
    scorer = BM25()
    t0 = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < args.seconds:
        search(index, args.query, scorer, k=args.k)
        n += 1
    print(f"done loops={n} elapsed={time.perf_counter() - t0:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
