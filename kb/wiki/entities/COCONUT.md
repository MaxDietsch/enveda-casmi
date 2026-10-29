---
title: COCONUT
type: entity
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md, raw/refs/scripts-overview.md]
tags: [entity, database, natural-products]
---

# COCONUT

**COlleCtion of Open Natural prodUcTs** — an open database of natural products (molecules produced by living organisms). Class-2 test molecules are, by definition, structures present in PubChem or COCONUT that have never been measured ([[concepts/competition/Novelty Classes]]). The repo builds a standardised COCONUT extract (`build_coconut_pool.py`: `inchikey14, smiles, mass`, with verified masses) as the class-2 candidate pool for scoring time, uploaded as a Kaggle dataset because the scoring notebook runs without internet. COCONUT membership is also a proposed indicator of natural-product-likeness for weighting the evaluation simulator ([[concepts/solution/Candidate Pools]], [[concepts/solution/Roadmap]]).
