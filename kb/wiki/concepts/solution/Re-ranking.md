---
title: Re-ranking
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/max-class1-readme.md, raw/refs/stage2-plan.md, raw/refs/stage2-training-explained.md, raw/refs/src-enveda-casmi26.md]
tags: [reranking, features, leakage, scorer]
---

# Re-ranking (Stage 2 — Learned Shortlist Scorer)

Retrieval hands over near-perfect recall; the re-ranker turns those shortlists into the right order. Two implementations share one idea.

**`max_class1_exp/stage2.py`** — re-ranks the stage-1 shortlists with a small tabular model over ~16 deterministic features per (query molecule, candidate):

| Feature group | Features |
|---|---|
| Stage-1 scores | `entropy_best`, `cosine_best`, `entropy_composite` |
| Neutral-loss space | `nl_entropy_best`, `nl_cosine_best` — the signal that survives adduct changes ([[concepts/solution/Retrieval]]) |
| Match statistics | number of matched peaks, share of intensity explained, against the candidate's best-matching library spectrum |
| Mass | absolute mass error in ppm |
| Sizes | number of the candidate's library spectra (`n_candidates` is context only, never a model feature) |
| Setting flags | shares of the query's spectra measured with the same adduct / same polarity / same instrument family |
| Acquisition | smallest collision-energy gap to the candidate's spectra (missing → imputed 100); number of source libraries containing the molecule (a "well-documented compound" prior) |

Continuous features are **standardised (z-scored) within each shortlist**: the model sees how a candidate compares to its competitors for the same molecule, not absolute values. The label is "this candidate is the correct structure", exactly one positive per shortlist. Models: logistic regression and/or `HistGradientBoostingClassifier`, chosen on an internal 10% validation subset.

**Training data is generated during training, in memory**, from the regular training set, using the simulator's building blocks (`make_difficulty1.qualifying` / `choose_rows`): each sampled molecule holds out one measurement setting as a pseudo-query. Leakage rules:

- **R1** no evaluation molecule ever becomes a training query (no positive label for a molecule that will later be evaluated); rows the evaluation split hides are invisible. Evaluation molecules appear only in the roles they also play at evaluation time (negative candidates; visible library rows) — enforced by assertions.
- **R2** each training example's pseudo-query rows are excluded from its library — the correct structure is reachable only through its other measurement settings. Asserted.
- **R3** simulator metadata (`change`, `difficulty`) is never a feature.
- **R4** one shared `featurize` function for training and evaluation.
- **R5** a single seeded random generator.

**`src/enveda_casmi26/scoring.py` (Tier 4)** generalises this beyond library matches: individual match features barely separate the true structure from coincidental same-formula candidates, so features are **aggregated per candidate** (how much of the spectrum the candidate explains, and the chemical plausibility of those explanations) and a logistic model learns P(candidate is correct | features). Trained by `train_scorer.py`; exported for the Kaggle notebook by `export_scorer_json.py` ([[concepts/solution/Submission Architecture]]).

**Success criteria** (STAGE2_PLAN §6): stage-2 MRR@25 ≥ 0.765 overall (the entropy_best baseline); the changed-adduct case clearly above its baseline 0.393 (0.5+ is a clear improvement); changed-collision-energy and changed-instrument cases not degraded; neutral-loss and/or entropy features prominent in the feature importances.
