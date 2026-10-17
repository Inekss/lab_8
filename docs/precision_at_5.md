# Precision@5 — keyword / semantic / hybrid

| query | kind | keyword | semantic | hybrid | relevant |
|---|---|---:|---:|---:|---|
| `quirrell` | original | 0.20 | 0.20 | 0.20 | `harry_potter_1` |
| `flamel` | original | 0.20 | 0.20 | 0.20 | `harry_potter_1` |
| `firenze` | original | 0.20 | 0.20 | 0.20 | `harry_potter_1` |
| `basilisk` | original | 0.20 | 0.20 | 0.20 | `harry_potter_2` |
| `lockhart` | original | 0.20 | 0.20 | 0.20 | `harry_potter_2` |
| `dobby` | original | 0.20 | 0.20 | 0.20 | `harry_potter_2` |
| `"chamber of secrets"` | original | 0.20 | 0.20 | 0.20 | `harry_potter_2` |
| `buckbeak` | original | 0.20 | 0.20 | 0.20 | `harry_potter_3` |
| `lupin` | original | 0.20 | 0.20 | 0.20 | `harry_potter_3` |
| `pettigrew` | original | 0.20 | 0.20 | 0.20 | `harry_potter_3` |
| `turbaned teacher who shared a body with the dark lord` | paraphrase | 0.00 | 0.20 | 0.20 | `harry_potter_1` |
| `alchemist who created a stone that grants endless life` | paraphrase | 0.00 | 0.20 | 0.20 | `harry_potter_1` |
| `enormous serpent hidden in the school's underground pipes` | paraphrase | 0.20 | 0.20 | 0.20 | `harry_potter_2` |
| `tiny magical servant who stole letters and warned of a plot` | paraphrase | 0.20 | 0.20 | 0.20 | `harry_potter_2` |
| `innocent half-eagle creature sentenced to be killed by officials` | paraphrase | 0.00 | 0.20 | 0.20 | `harry_potter_3` |
| **mean** | | 0.16 | 0.20 | 0.20 | |

## What this shows

P@5 = 0.20 means the labeled book is in the top 5 once (1/5) — normal with three Potter titles among eight docs. On **original** name queries keyword already finds the right book. On **paraphrases** without shared vocabulary, keyword drops to 0.00 for three queries while semantic (and hybrid via RRF) still place the right book in the top 5. Mean: keyword 0.16 vs semantic/hybrid 0.20 — the gap is exactly those paraphrase misses.
