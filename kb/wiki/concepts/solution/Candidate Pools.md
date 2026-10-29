---
title: Candidate Pools
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md, raw/refs/scripts-overview.md, raw/refs/dataset-creation-readme.md]
tags: [pools, coconut, pubchem, class-2]
---

# Candidate Pools

Retrieval and re-ranking ([[concepts/solution/Retrieval]], [[concepts/solution/Re-ranking]]) need a set of structures to rank. The pool's composition decides **which difficulty classes are reachable at all**:

| Class | Requirement on the pool |
|---|---|
| 1 — publicly measured | Training structures suffice (they *are* the reference library) |
| 2 — in PubChem/COCONUT, never measured | The structure must come from an **external database extract** |
| 3 — not in PubChem | Unreachable by any pool — generation only ([[concepts/solution/De Novo Channel]]) |

**Repo assets:**

- **Training structures** — all `normalized_smiles` from visible training rows; the base pool (~275k structures, deduplicated by tautomer-standardised key).
- **COCONUT pool** (`scripts/build_coconut_pool.py`) — standardised SMILES from COCONUT (the natural-products database) with verified keys and neutral masses, written as a compact table (`inchikey14, smiles, mass`) for use at scoring time **and prepared for upload as a Kaggle dataset** (the scoring notebook runs without internet; [[concepts/solution/Submission Architecture]]). `build_pool_fps.py` precomputes pool fingerprints.
- **`pool_extra.parquet`** (from `make_split.py`) — held-out training structures playing external-database candidates in local validation, standing in for COCONUT in class-2 evaluation.
- **PubChem** — implied by the class definitions but no build script exists yet; the class-2/class-3 boundary runs *through* PubChem, so its snapshot date and coverage matter (open question).

**Local-evaluation discipline** (simulator §7.3): a pool built from visible training rows alone must be extended with `split.library_additions()` (the level-2 structures) — otherwise level 2 silently turns into level 3 — and every pool, external ones included, must drop `split.excluded_library_keys` (the level-3 structures). Pools are compared **by key**, never by SMILES string.

**Ranking structures that were never measured:** pool members absent from the training spectra have no reference spectra, so the similarity channel is silent for them. They are ranked by the mass channel (the ±ppm window), the in-silico fragmentation channel (would this structure plausibly produce the observed peaks?), the learned scorer, and fingerprint-space proximity from the de novo encoder ([[concepts/solution/index|architecture]]). This is why the in-silico fragmenter (`fragmentation.py`) is central to class 2.
