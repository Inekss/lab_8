# 5 paraphrase queries (Lab 08 M3)

Same books as lab-3 `EVAL`, but **no shared keywords** with the character/object names
used in the original 10 queries. Expected label = document `title` (stem of the `.txt`).

| # | Query (paraphrase) | Relevant | Maps to (lab-3 idea) |
|---|---|---|---|
| 1 | `turbaned teacher who shared a body with the dark lord` | `harry_potter_1` | Quirrell |
| 2 | `alchemist who created a stone that grants endless life` | `harry_potter_1` | Flamel |
| 3 | `enormous serpent hidden in the school's underground pipes` | `harry_potter_2` | Basilisk |
| 4 | `tiny magical servant who stole letters and warned of a plot` | `harry_potter_2` | Dobby |
| 5 | `innocent half-eagle creature sentenced to be killed by officials` | `harry_potter_3` | Buckbeak |

## As Python tuples (for later `EVAL_PARAPHRASE`)

```python
EVAL_PARAPHRASE = (
    ("turbaned teacher who shared a body with the dark lord", "harry_potter_1"),
    ("alchemist who created a stone that grants endless life", "harry_potter_1"),
    ("enormous serpent hidden in the school's underground pipes", "harry_potter_2"),
    ("tiny magical servant who stole letters and warned of a plot", "harry_potter_2"),
    ("innocent half-eagle creature sentenced to be killed by officials", "harry_potter_3"),
)
```

## How we will use them

Together with the original 10 `EVAL` queries → **15 rows** in precision@5 for
`keyword` / `semantic` / `hybrid`. Expect keyword weak on these five; semantic stronger.
