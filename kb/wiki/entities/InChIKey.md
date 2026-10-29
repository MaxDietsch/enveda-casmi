---
title: InChIKey
type: entity
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-competition-overview.md, raw/refs/dataset-creation-readme.md]
tags: [entity, identifier, key]
---

# InChIKey

A fixed-length identifier computed from a molecular structure. The competition's matching key is its **first block (InChIKey14)** — 14 characters identifying the molecule's flat skeleton, ignoring stereochemistry — computed **after tautomer standardisation** (RDKit pinned at 2026.03.3). Two predictions count as correct exactly when their standardised first blocks are equal, so tautomer and stereo-variants of one molecule are interchangeable. In the training data, 275,810 distinct `inchikey14` values collapse to 274,195 standardised keys (1,615 tautomer duplicates). The repo computes the key with `TautomerEnumerator().Canonicalize(mol)` → `MolToInchiKey()[:14]` (`common.molecule_key`) and uses it exclusively for exclusion and scoring ([[concepts/solution/Evaluation Strategy]]). Structures sharing a molecular formula do not automatically share a key — but same-formula structures are the most convincing wrong candidates.
