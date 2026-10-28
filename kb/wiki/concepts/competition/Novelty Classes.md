---
title: Novelty Classes
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md, raw/refs/dataset-creation-readme.md]
tags: [competition, classes, strategy]
---

# Novelty Classes

Every scored test molecule falls into exactly one of three hidden classes, in increasing difficulty:

| Class | Definition | What can find it |
|---|---|---|
| **1 — in public spectral libraries** | Reference MS/MS spectra exist publicly (hence in `train.parquet`) | Library search / spectral similarity |
| **2 — known structure, no public spectra** | No public spectra, but the structure is in [[entities/PubChem|PubChem]] or [[entities/COCONUT|COCONUT]] | Database retrieval + spectral reasoning (fragmentation plausibility, formula/CSIF-like scoring) |
| **3 — novel structure** | Not in PubChem at all | De novo structure generation only |

The class of each test molecule — and the class distribution — is **hidden** for the whole competition. The repo's simulator uses a **20% / 60% / 20%** mix (level 1 / 2 / 3) as the working assumption.

**Strategic implications:**

1. **Class 1 (≤ 20% of molecules)** is winnable by pure spectral matching, but the reference spectra may come from a *different setting* — another instrument, adduct, or collision energy. The repo measures exactly this in `max_class1_exp/` ([[concepts/solution/Retrieval]]): adduct changes are the hard case (MRR 0.393), CE changes easy (0.933).
2. **Class 2 (~60%)** is the bulk of the score. The candidate is in PubChem/COCONUT but has *no reference spectrum*, so similarity alone cannot rank it. Retrieval must reason about **how well a candidate structure would fragment** ([[concepts/solution/Candidate Pools]], in-silico fragmentation channels) or use embeddings.
3. **Class 3 (~20%)** requires de novo generation ([[concepts/solution/De Novo Channel]]) — even modest MRR here (e.g. 0.1) is worth ~0.02 overall.
4. Because classes are hidden, the final ranked list must **blend** all three channels behind a calibrated confidence so that strong library hits sit on top and database/de novo guesses fill the rest ([[concepts/solution/Submission Architecture]]); the local analogue is `max_visible_tc`-banded reporting and the difficulty-1-vs-2 confidence calibration.
