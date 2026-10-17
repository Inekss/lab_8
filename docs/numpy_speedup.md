# NumPy scorer speedup (Lab 08)

Corpus: `D:\games\python_laby\lab_8\src\resources` · docs=8 · terms=69525

| Query | Term(s) | Docs matched | Lab 3 loop (ms) | NumPy (ms) | Speedup |
|---|---|---:|---:|---:|---:|
| common term (8-doc corpus) | `book` | 6 | 0.010 | 0.039 | 0.3x |
| rare term (8-doc corpus) | `aaaaaaaaaargh` | 1 | 0.007 | 0.073 | 0.1x |
| 3-term query (8-doc corpus) | `harry potter magic` | 3 | 0.024 | 0.331 | 0.1x |
| common term (50k synthetic docs) | `common` | 50000 | 176.312 | 18.364 | 9.6x |

Largest speedup in this table: **9.6x**.

## Notes

- Measured scorer + top-k only (no snippet disk I/O).
- On the real 8-document corpus, posting lists are tiny: NumPy setup cost dominates and speedup can be **below 1x**. That is expected.
- On a 50k-doc synthetic common term, the Lab-3 Python loop over `Posting` objects is the bottleneck and NumPy wins clearly.
