---
title: Dataset-Creation README (Eval-Set Simulator)
type: summary
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/dataset-creation-readme.md, raw/refs/dataset-creation-plan.md, raw/refs/dataset-creation-improvements.md]
tags: [validation, evaluation, simulator]
---

# Summary — Eval-Set Simulator (`dataset-creation/`)

The repo's core evaluation tool: builds **local eval sets that mimic the hidden Kaggle test** so solutions can be scored with MRR@25 locally, with known answers *and* known difficulty. Split files never copy spectra — they reference row ids of `train.parquet` (~250 KB per split); `dataloader.py` is the only place where splits meet the real data.

**Molecule identity = scorer identity:** tautomer-canonical InChIKey14 (`TautomerEnumerator().Canonicalize(mol)` → `MolToInchiKey()[:14]`). Train has 275,810 distinct `inchikey14` but only 274,195 keys (1,615 tautomer duplicates merged). Hiding works on the key, so tautomers cannot leak.

**Eligible eval spectra** imitate the test cleaning: timsTOF, one of the ten test adducts, CE present, base peak ≥ 1,000, |precursor error| ≤ 10 ppm, ≥ 1 peak. ~180k molecules qualify (mass 157–1,159 Da); nearly all from `enveda-180` (see Known Limits).

**Three difficulty levels** mirror the hidden novelty classes:
- **Level 1** — molecule stays in train but its visible spectra differ from the eval spectra in one (or 2–3, `mixed`) dimensions: `instrument`, `adduct`, `ce`, `mixed` (default mix 30/20/20/30).
- **Level 2** — all rows of the molecule hidden (known structure, no spectra); for 10% also every neighbour at Tanimoto ≥ 0.5 is hidden ("nothing similar in train").
- **Level 3** — like 2 but flagged `exclude_from_library` (must be removed from candidate pools by the evaluation code).

**Overall set:** 20% level 1 / 60% level 2 / 20% level 3. Default 1,000 molecules (score noise ±0.013; use several seeds, paired comparisons). Sampling is weighted (mass bins toward test shares, 70% positive mode) and diversified (no two eval molecules at Tanimoto ≥ 0.6; weight ÷ neighbour count).

**Fair-evaluation rules:** train only where `train_mask` is True; add level-2 structures to the candidate pool via `library_additions()`; remove level-3 keys from *every* pool; compare by key, never by SMILES; never tune on the truth.

**Validation:** `check_split.py` runs 9 leak/reproducibility checks; all default sets pass. `max_visible_tc` column enables score-by-similarity-band analysis.

Known gaps (fewer spectra per molecule than the test, synthetic chemistry, remaining analogues, fingerprint collisions) are catalogued with ideas in `POTENTIAL_IMPROVEMENTS.md`.
