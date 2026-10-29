---
title: Novelty Classes
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md, raw/refs/dataset-creation-readme.md]
tags: [competition, classes, strategy]
---

# Novelty Classes

Every test molecule belongs to exactly one of three hidden difficulty categories, in increasing order of difficulty:

| Class | Definition | What a solution needs |
|---|---|---|
| **1 — publicly measured before** | Reference spectra of the molecule exist publicly (and therefore in `train.parquet`) | Comparison against reference spectra (spectral similarity) |
| **2 — known structure, never measured** | No public spectra exist, but the structure is listed in [[entities/PubChem|PubChem]] or [[entities/COCONUT|COCONUT]] | Retrieval from structural databases, ranked by how plausibly each candidate would produce the observed fragments |
| **3 — unknown structure** | The structure is not in PubChem at all | Generating a structure directly from the spectrum (de novo) |

Which molecule belongs to which class — and the class distribution overall — is **hidden** for the whole competition. The repo's simulator assumes a **20% / 60% / 20%** mix (class 1 / 2 / 3).

**What this implies for the solution:**

1. **Class 1 (assumed ~20% of molecules)** is solvable by direct spectrum comparison — but the reference spectra may come from a *different measurement setting*: another instrument, adduct, or collision energy. The repo measures exactly this in `max_class1_exp/` ([[concepts/solution/Retrieval]]): changed adducts are the hardest case (MRR 0.393), changed collision energies the easiest (0.933).
2. **Class 2 (assumed ~60%)** carries the largest share of the score. The correct structure exists in a database but has *never been measured*, so spectrum comparison alone cannot rank it. The solution must judge **how well a candidate structure would fragment into the observed peaks** ([[concepts/solution/Candidate Pools]]; the in-silico fragmentation channel) or use learned spectrum embeddings.
3. **Class 3 (assumed ~20%)** requires generating structures from the spectrum ([[concepts/solution/De Novo Channel]]) — even a modest MRR of 0.1 here contributes about 0.02 to the overall score.
4. Because the classes are hidden, the final ranked list must **blend** all three approaches behind a calibrated confidence, so that strong reference-spectrum matches come first and database and generated candidates fill the remaining positions ([[concepts/solution/Submission Architecture]]). The local analogue is reporting scores by similarity band (`max_visible_tc`) and calibrating match confidence on difficulty-1 versus difficulty-2 evaluation sets.
