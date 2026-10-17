# Top-3 hotspots (BEFORE scorer / NumPy changes)

Measured: 2026-10-17 · corpus `src/resources` (8 docs) · `index.pkl`  
Query: `harry potter magic`

Artifacts:
- Cold CLI: `docs/cprofile_search.txt`
- Hot path (index already in RAM, 200× search): `docs/cprofile_search_hotpath.txt`
- Flame SVG: `docs/search.svg` (cProfile + flameprof; **py-spy** on this Windows fails with `Failed to find python version from target process`)

---

## A. Cold `findex search` (CLI one-shot)

Almost all time is **import + pickle load**, not BM25.

### 1. `load_pickle` / `_pickle.load` + dataclass restore
- ~0.57 s · Hypothesis: ~105k `Posting` objects rebuilt in Python
- Mix Python + native pickle

### 2. Import tax (`importlib`, pulling crawl/rich/httpx)
- ~0.74 s · Hypothesis: cold CLI; irrelevant for API (index already loaded)

### 3. Filesystem during import (`open_code`, `read`, `nt.stat`)
- Native/system I/O · not a NumPy target

---

## B. Hot-path search (what matters for API / Lab 08 speedup) — primary top-3

From `cprofile_search_hotpath.txt` (~1.08 s for 200 searches ≈ **5.4 ms/query**):

### 1. `_snippet` → `documents.read_text` / disk I/O
- **Evidence:** `_snippet` cumtime ≈ 0.95 s of 1.08 s; `BufferedReader.read` + `str.lower` + `decode`
- **Hypothesis:** every hit re-reads the full `.txt` from disk to build a snippet
- **Python vs native:** I/O native + a lot of Python string work
- **Note:** NumPy BM25 won’t fix this; caching text or skipping snippet in benches will

### 2. `search()` orchestration (scoring loop + heap)
- **Evidence:** `search` cumtime ≈ 1.08 s wrapping everything; BM25 `__call__` itself only ~0.014 s
- **Hypothesis:** on this small corpus, postings loops are cheap; wall time is snippet-dominated
- **For Lab 08:** still vectorize BM25 — speedup shows on **long postings lists** / larger corpus; bench table must measure scorer alone or search without snippet I/O

### 3. Path / open overhead inside snippet (`pathlib`, `nt.stat`, `open`)
- Repeated per result document
- Native/system · Hypothesis: `is_file` + open per snippet

---

## Implications (write before NumPy patch)

1. Cold CLI profile ≠ search algorithm cost.  
2. Real per-query cost here is **snippet disk reads**.  
3. NumPy still required by the lab for postings/BM25; measure scorer ms separately so the speedup table is honest.  
4. Optional quick win later: cache document text in memory at index load (not required for M1).

## After NumPy

Re-run `scripts/make_search_flame.py` and compare `docs/search.svg` / hotpath txt.
