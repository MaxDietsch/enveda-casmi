---
title: Candidate Pools
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md, raw/refs/scripts-overview.md, raw/refs/dataset-creation-readme.md]
tags: [pools, coconut, pubchem, class-2]
---

# Candidate Pools

Retrieval and re-ranking ([[concepts/solution/Retrieval]], [[concepts/solution/Re-ranking]]) need a set of structures to rank. The pool composition decides **which classes are reachable at all**:

| Class | Requirement on the pool |
|---|---|
| 1 — public spectra | Train structures suffice (they *are* the library) |
| 2 — in PubChem/COCONUT, no spectra | Structure must come from an **external database pool** |
| 3 — not in PubChem | Unreachable by any pool — de novo only ([[concepts/solution/De Novo Channel]]) |

**Repo assets:**

- **Train structures** — all `normalized_smiles` from visible train rows; the base pool (~275k structures, deduplicated by tautomer-canonical key).
- **COCONUT pool** (`scripts/build_coconut_pool.py`) — canonicalised COCONUT SMILES with verified `inchikey14` and neutral masses, written as a compact parquet (`inchikey14, smiles, mass`) for kernel runtime use **and prepared for Kaggle dataset upload** (the kernel runs offline, [[concepts/solution/Submission Architecture]]). `build_pool_fps.py` precomputes pool fingerprints.
- **`pool_extra.parquet`** (from `make_split.py`) — held-out train structures playing the role of external-DB candidates in local validation, "exactly like COCONUT entries for Class 2".
- **PubChem** — implied by the class definitions but no build script yet; the class-2/class-3 boundary runs *through* PubChem, so its snapshot date and coverage matter (open question).

**Local-evaluation discipline** (simulator §7.3): a pool built from visible train rows alone must be extended with `split.library_additions()` (level-2 structures) — otherwise level 2 silently becomes level 3 — and every pool, external included, must drop `split.excluded_library_keys` (level 3). Pools are compared **by key**, never by SMILES string.

**Ranking without reference spectra:** pool members that never appeared in train have no spectra, so the `sim` channel is silent for them; they are ranked by the `mass` channel (±ppm window), the `eif` channel (in-silico fragmentation explained-intensity fraction — does this structure *plausibly produce* the observed peaks?), the `learned` scorer, and fingerprint-space proximity from the de novo encoder ([[concepts/solution/index|architecture]]). This is why the in-silico fragmenter (`fragmentation.py`) is load-bearing for class 2.
