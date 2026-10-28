---
title: COCONUT
type: entity
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md, raw/refs/scripts-overview.md]
tags: [entity, database, natural-products]
---

# COCONUT

**COlleCtion of Open Natural prodUcTs** — the natural-products database. Class-2 test molecules are, by definition, structures present in PubChem or COCONUT with no public spectra ([[concepts/competition/Novelty Classes]]). The repo builds a canonicalised COCONUT pool (`build_coconut_pool.py`: `inchikey14, smiles, mass`, verified masses) as the class-2 candidate pool for kernel runtime, uploaded as a Kaggle dataset because the kernel runs offline. COCONUT membership is also a proposed proxy for natural-product-likeness weighting in the eval simulator ([[concepts/solution/Candidate Pools]], [[concepts/solution/Roadmap]]).
