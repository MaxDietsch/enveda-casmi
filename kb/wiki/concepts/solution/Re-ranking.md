---
title: Re-ranking
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/max-class1-readme.md, raw/refs/stage2-plan.md, raw/refs/stage2-training-explained.md, raw/refs/src-enveda-casmi26.md]
tags: [reranking, features, leakage, scorer]
---

# Re-ranking (Stage 2 — Learned Shortlist Scorer)

Retrieval hands over near-perfect recall; the re-ranker turns shortlists into the right order. Two implementations share the same idea:

**`max_class1_exp/stage2.py`** — re-ranks stage-1 shortlists with a tabular model over ~16 deterministic features per (query molecule, candidate):

| Feature group | Features |
|---|---|
| Stage-1 scores | `entropy_best`, `cosine_best`, `entropy_composite` |
| Neutral-loss space | `nl_entropy_best`, `nl_cosine_best` — the signal that survives adduct changes |
| Match statistics | `matched_peaks`, `explained_intensity` vs the candidate's best-entropy library spectrum |
| Mass | `mass_error_ppm` (abs) |
| Sizes | `n_library_spectra` (context only: `n_candidates` never a feature) |
| Setting flags | `has_same_adduct`, `has_same_polarity`, `same_instrument_family` (shares over query spectra) |
| Acquisition | `min_ce_gap` (imputed 100 when NaN), `n_source_libraries` ("well-known compound" prior) |

Continuous features are **z-scored within each shortlist** (contrast trick, as in `CandidateScorer`): the model learns *how a candidate compares to its competitors*, not absolute values. Labels: $y = (\text{candidate} = \text{truth})$, exactly one positive per shortlist. Models: logistic regression and/or `HistGradientBoostingClassifier`, selected on an internal 10% holdout.

**Training data is generated on the fly** from the regular train set with the simulator building blocks (`make_difficulty1.qualifying` / `choose_rows`): each sampled molecule holds out one setting as a pseudo-query. Leakage rules:

- **R1** no eval molecule ever becomes a training query (no positive label for evaluated molecules); split-hidden rows invisible. Eval molecules appear only in their evaluation-time roles (label-0 decoys, visible library rows) — asserted.
- **R2** the pseudo-query's rows are held out of the library — asserted.
- **R3** simulator metadata (`change`, `difficulty`) is never a feature.
- **R4** one shared `featurize` for training and evaluation.
- **R5** single seeded RNG.

**`src/enveda_casmi26/scoring.py` (Tier 4)** generalizes this beyond library matches: per-match features barely separate truth from decoys, so features are **aggregated per candidate** (how much of the spectrum it explains, and the chemical quality of the explanations) and a logistic model learns P(true | features). Trained by `train_scorer.py`; exported for the kernel by `export_scorer_json.py` ([[concepts/solution/Submission Architecture]]).

**Success criteria** (STAGE2_PLAN §6): overall stage-2 MRR@25 ≥ 0.765 (entropy_best baseline); `adduct` clearly above 0.393 (0.5+ = clear win); `ce`/`instrument` not degraded; neutral-loss and/or entropy features on top of feature importances.
