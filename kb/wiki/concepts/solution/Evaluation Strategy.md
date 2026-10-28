---
title: Evaluation Strategy
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/dataset-creation-readme.md, raw/refs/dataset-creation-improvements.md, raw/refs/scripts-overview.md]
tags: [evaluation, validation, leakage]
---

# Evaluation Strategy

Kaggle gives only ~400 scored molecules behind 5 submissions/day, so all development runs on **local, leak-free evaluation splits** built by `dataset-creation/` — a test-set simulator with known answers and known difficulty.

**Two complementary split mechanisms:**

1. **`dataset-creation/` simulator** (primary): constructs eval molecules that imitate the hidden test's three novelty classes.
    - Level 1 (class 1): molecule's eval spectra differ from its visible train spectra in declared dimensions — `instrument`, `adduct`, `ce`, or `mixed` (2–3 dims at once).
    - Level 2 (class 2): all rows hidden; optionally (10%) all Tanimoto ≥ 0.5 neighbours hidden too. Level-2 structures must be added to the candidate pool via `split.library_additions()`.
    - Level 3 (class 3): flag only — the evaluation code must drop `excluded_library_keys` from every pool, external ones included.
    - Default overall set: 1,000 molecules at 20/60/20; multiple seeds (2026–2030) enable paired comparisons (score noise ±0.013 at n=1000).
2. **`scripts/make_split.py`** (coarser): holds out entire structures → `train_cut.parquet` (drop-in train replacement), `test_cut.parquet` (mirrors test schema + truth), `pool_extra.parquet` (held-out structures play external-DB candidates, like COCONUT for class 2).

**The molecule key is the scorer's key:** tautomer-canonical InChIKey14. Hiding always operates on the key, so tautomers stored under different `inchikey14` cannot leak. Compare predictions by key, never by SMILES; `rescore_tautomer.py` additionally reports formula-level and tautomer-canonical matches to separate retrieval failures from representation mismatches.

**Fair-evaluation rules** (from the simulator's §7.3):
1. Train only where `train_mask` is True — every statistic, prior, and lookup table built from train included.
2. Add level-2 structures to the pool (otherwise level 2 silently becomes level 3).
3. Remove level-3 keys from every pool.
4. Never tune on the truth — one dev seed, other seeds for final comparison.

**Diagnostics beyond the headline MRR:** per-difficulty, per-`change`, per-`ingest_lib` (NP vs synthetic gap), and per-similarity-band via `max_visible_tc` (0–0.4 / 0.4–0.6 / 0.6–1.0). `check_split.py` enforces 9 leak/reproducibility checks; the class-1 experiment adds unit tests plus `--leak-eval`/`--leak-train` self-tests that must approach MRR 1.0.

Known simulator limits (fewer spectra per molecule, synthetic chemistry, residual analogues) are tracked in `POTENTIAL_IMPROVEMENTS.md` — read local scores with these in mind.
