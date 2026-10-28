---
title: Competition
type: concept-index
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-competition-overview.md, raw/articles/kaggle-data-description.md]
tags: [competition]
---

# Competition — Enveda CASMI 2026

Kaggle competition `enveda-CASMI26-molecule-id-mass-spectra`, hosted by [[entities/Enveda Therapeutics|Enveda Therapeutics]] reviving the [[entities/CASMI|CASMI]] tradition. Predict up to 25 ranked SMILES per molecule from LC-MS/MS spectra; scored by MRR@25 on the tautomer-canonical InChIKey14. $50k prize pool; final deadline **Dec 14, 2026**; code competition (notebook-only, ≤ 9 h, offline, external pretrained models allowed as datasets).

Sub-pages:

- [[concepts/competition/Task and Evaluation]] — MRR@25, InChIKey14 matching, submission format, submission quota
- [[concepts/competition/Test Set]] — size, instrument, adducts, cleaning, per-molecule aggregation
- [[concepts/competition/Novelty Classes]] — the three hidden difficulty classes and what each demands of a solution

The three classes drive the solution architecture: library search wins class 1, database retrieval with spectral reasoning covers class 2, and only de novo generation can score on class 3 — see [[concepts/solution/index|Solution Architecture]].

Sources: [[summaries/kaggle-competition-overview]], [[summaries/kaggle-data-description]], [[summaries/kaggle-rules-timeline]].
