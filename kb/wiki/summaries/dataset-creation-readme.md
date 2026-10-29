---
title: Dataset-Creation README (Eval-Set Simulator)
type: summary
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/dataset-creation-readme.md, raw/refs/dataset-creation-plan.md, raw/refs/dataset-creation-improvements.md]
tags: [validation, evaluation, simulator]
---

# Summary — Eval-Set Simulator (`dataset-creation/`)

The repo's core evaluation tool: builds **local evaluation sets that imitate the hidden Kaggle test**, so solutions can be scored with MRR@25 locally — with known correct answers *and* known difficulty. Split files never copy spectra; they reference row numbers of `train.parquet` (~250 KB per split), and `dataloader.py` is the only place where splits meet the real data.

**Molecule identity equals scorer identity:** the tautomer-standardised InChIKey14 (`TautomerEnumerator().Canonicalize(mol)` → `MolToInchiKey()[:14]`). Train contains 275,810 distinct `inchikey14` values but only 274,195 standardised keys (1,615 tautomer duplicates merged). All exclusion operates on the key, so tautomers recorded under different keys cannot leak the answer.

**Eligible evaluation spectra** imitate the test's cleaning: timsTOF, one of the ten test adducts, collision energy present, strongest peak ≥ 1,000 counts, |precursor mass error| ≤ 10 ppm, at least one peak. About 180k molecules qualify (mass 157–1,159 Da); almost all come from `enveda-180` (see limitations below).

**Three difficulty levels** mirror the hidden novelty classes:
- **Level 1** — the molecule stays in the training data, but its visible spectra differ from its evaluation spectra in one (or 2–3, `mixed`) dimensions: `instrument`, `adduct`, `ce`, `mixed` (default mix 30/20/20/30).
- **Level 2** — every row of the molecule is removed from training (known structure, never measured); for 10% of cases every structurally similar molecule (fingerprint similarity ≥ 0.5) is removed too, simulating "nothing similar in the training data".
- **Level 3** — like level 2, but flagged `exclude_from_library`: the evaluation code must remove these structures from every candidate pool.

**Overall set:** 20% level 1 / 60% level 2 / 20% level 3. Default 1,000 molecules (score noise ±0.013; use several seeds for paired comparisons). Sampling is weighted (mass ranges toward test shares; 70% positive mode) and diversified (no two evaluation molecules at fingerprint similarity ≥ 0.6; sampling weight divided by neighbourhood size).

**Fair-evaluation rules:** train only where `train_mask` is True; add level-2 structures to the candidate pool via `library_additions()`; remove level-3 keys from *every* pool; compare by key, never by SMILES string; never tune on the correct answers.

**Validation:** `check_split.py` runs 9 leak and reproducibility checks; all default sets pass. The `max_visible_tc` column (similarity of the most similar still-visible molecule) enables score-by-similarity-band analysis.

Known limitations (fewer spectra per molecule than the test, synthetic chemistry, remaining structural analogues, fingerprint collisions) are catalogued with proposed fixes in `POTENTIAL_IMPROVEMENTS.md`.
