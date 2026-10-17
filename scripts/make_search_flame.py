"""cProfile → SVG flame graph (py-spy fallback on Windows).

Usage (from lab_8):
  .\\.venv\\Scripts\\python.exe scripts\\make_search_flame.py
"""
from __future__ import annotations

import cProfile
import pstats
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PROF = DOCS / "search.prof"
SVG = DOCS / "search.svg"
TXT = DOCS / "cprofile_search_hotpath.txt"


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))

    from findex.score import BM25
    from findex.search import search
    from findex.store import load_pickle

    index = load_pickle(ROOT / "index.pkl")
    scorer = BM25()
    query = "harry potter magic"

    # warm one call outside the profile (optional); then profile hot path only
    search(index, query, scorer, k=5)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(200):
        search(index, query, scorer, k=5)
    pr.disable()
    pr.dump_stats(PROF)

    with TXT.open("w", encoding="utf-8") as fh:
        fh.write(
            "# Hot-path cProfile: 200x search after index already loaded\n"
            f"# query={query!r}\n\n"
        )
        stats = pstats.Stats(pr, stream=fh)
        stats.sort_stats("cumtime")
        stats.print_stats(40)

    # flameprof writes SVG to stdout
    r = subprocess.run(
        [sys.executable, "-m", "flameprof", str(PROF)],
        check=False,
        capture_output=True,
    )
    if r.returncode != 0:
        print(r.stderr.decode("utf-8", errors="replace"), file=sys.stderr)
        print(
            "Install flameprof: uv add --dev flameprof",
            file=sys.stderr,
        )
        return 1
    SVG.write_bytes(r.stdout)
    print(f"wrote {SVG}")
    print(f"wrote {TXT}")
    print(f"wrote {PROF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
