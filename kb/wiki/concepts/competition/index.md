---
title: Competition
type: concept-index
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-competition-overview.md, raw/articles/kaggle-data-description.md]
tags: [competition]
---

# Competition — Enveda CASMI 2026

Kaggle competition `enveda-CASMI26-molecule-id-mass-spectra`, hosted by [[entities/Enveda Therapeutics|Enveda Therapeutics]], reviving the [[entities/CASMI|CASMI]] benchmark series. The task: for each unknown molecule (measured as a group of tandem-mass-spectrometry spectra), output up to 25 candidate structures as SMILES strings, ranked best first. Scoring uses MRR@25 on the tautomer-canonical InChIKey14 (see [[concepts/Glossary]]). Prize pool $50,000; final deadline **Dec 14, 2026**; submissions run as Kaggle notebooks with a 9-hour limit, no internet access, and externally pretrained models permitted as attached datasets.

Sub-pages:

- [[concepts/competition/Task and Evaluation]] — the metric, the matching rule, the submission format, the daily submission limit
- [[concepts/competition/Test Set]] — size, instrument, adducts, applied cleaning, per-molecule aggregation
- [[concepts/competition/Novelty Classes]] — the three hidden difficulty categories and what each requires from a solution

The three classes drive the solution architecture: comparing against reference spectra solves class 1; ranking database entries by how plausibly they would fragment covers class 2; only structure generation from the spectrum itself can score on class 3 — see [[concepts/solution/index|Solution Architecture]].

Sources: [[summaries/kaggle-competition-overview]], [[summaries/kaggle-data-description]], [[summaries/kaggle-rules-timeline]].
