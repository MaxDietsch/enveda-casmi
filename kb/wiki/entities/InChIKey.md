---
title: InChIKey
type: entity
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-competition-overview.md, raw/refs/dataset-creation-readme.md]
tags: [entity, identifier, key]
---

# InChIKey

A fixed-length hash of the InChI structure identifier. The competition's matching key is the **first block (InChIKey14)** — the 14-character skeleton block that encodes atom connectivity independent of stereochemistry — computed **after RDKit tautomer canonicalization** (pinned 2026.03.3). Two molecules match iff their tautomer-canonical InChIKey14s are equal, so tautomer and stereo variants are free. In train, 275,810 distinct `inchikey14` values collapse to 274,195 canonical keys (1,615 tautomer duplicates). The repo computes the key with `TautomerEnumerator().Canonicalize(mol)` → `MolToInchiKey()[:14]` (`common.molecule_key`) and hides/scores exclusively by it ([[concepts/solution/Evaluation Strategy]]). Same-formula ≠ same key, but same-formula structures are the hardest decoys.
