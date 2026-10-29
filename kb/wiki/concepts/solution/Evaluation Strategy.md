---
title: Evaluation Strategy
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/dataset-creation-readme.md, raw/refs/dataset-creation-improvements.md, raw/refs/scripts-overview.md]
tags: [evaluation, validation, leakage]
---

# Evaluation Strategy

Kaggle offers only ~400 scored molecules and 5 submissions per day, so all development runs on **local evaluation splits that cannot leak** — built by `dataset-creation/`, a simulator of the hidden test with known answers and known difficulty. Leakage means information about the correct answer reaching the model through the training data, which inflates measured scores.

**Two complementary split mechanisms:**

1. **`dataset-creation/` simulator** (primary): constructs evaluation molecules that imitate the hidden test's three difficulty classes. (Definitions of "visible"/"hidden" below refer to which training rows a solution is allowed to see.)
    - Level 1 (class 1): the molecule's evaluation spectra differ from its still-visible training spectra in declared dimensions — `instrument`, `adduct`, `ce` (collision energy), or `mixed` (2–3 dimensions at once).
    - Level 2 (class 2): all of the molecule's rows are removed from training; optionally (10%) every structurally similar molecule (Tanimoto similarity ≥ 0.5 — a fingerprint-similarity measure) is removed too, simulating "nothing similar in the training data". Level-2 structures must be added to the candidate pool via `split.library_additions()`.
    - Level 3 (class 3): identical selection, but flagged — the evaluation code must remove these structures from every candidate pool, external ones included.
    - Default set: 1,000 molecules at 20% / 60% / 20%; several fixed random seeds (2026–2030) allow paired comparisons (score noise ±0.013 at n=1000).
2. **`scripts/make_split.py`** (simpler): removes entire structures from train → `train_cut.parquet` (drop-in train replacement), `test_cut.parquet` (mirrors the test schema, plus the correct answers), `pool_extra.parquet` (the removed structures play external-database candidates, as COCONUT does for class 2).

**The molecule key equals the scorer's key:** the tautomer-standardised InChIKey14. All exclusion operates on the key, so tautomers recorded under different `inchikey14` values cannot leak the answer. Compare predictions by key, never by SMILES string; `rescore_tautomer.py` additionally reports formula-level and tautomer-canonical matches, separating retrieval failures from mere notation mismatches.

**Fair-evaluation rules** (simulator §7.3):
1. Train only where `train_mask` is True — including every statistic, prior and lookup table derived from train.
2. Add level-2 structures to the candidate pool (otherwise level 2 silently turns into level 3).
3. Remove level-3 keys from every pool.
4. Never tune on the correct answers — one fixed seed for development, different seeds for the final comparison.

**Diagnostics beyond the headline MRR:** per difficulty, per changed dimension, per source library (natural-products versus synthetic difference), and per structural-similarity band via `max_visible_tc` (0–0.4 / 0.4–0.6 / 0.6–1.0). `check_split.py` enforces 9 leak/reproducibility checks; the class-1 experiment adds unit tests plus `--leak-eval` / `--leak-train` self-tests (deliberately leaking the answers to verify the pipeline reaches MRR ≈ 1).

Known simulator limitations (fewer spectra per molecule than the test, synthetic chemistry, remaining analogues) are tracked in `POTENTIAL_IMPROVEMENTS.md` — read local scores with these in mind.
