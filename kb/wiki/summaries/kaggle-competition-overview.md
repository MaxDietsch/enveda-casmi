---
title: Kaggle Competition Overview
type: summary
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-competition-overview.md]
tags: [competition, metric, submission]
---

# Summary — Kaggle Competition Overview

The competition asks for ML models that predict **2D chemical structures (SMILES) from LC-MS/MS spectra**, including for molecules never seen before — motivated by the fact that most molecules detected in biological samples are unknown to reference libraries. Applications: drug discovery and disease biomarkers.

**Metric: MRR@25.** For each molecule $u$, only the **first** correct guess in up to 25 ranked SMILES counts:

$$\text{MRR@25} = \frac{1}{U} \sum_{u=1}^{U} \frac{1}{rank_u}$$

Position 1 scores 1.0, position 2 scores 0.5, position 25 scores 0.04; no correct guess scores 0.

**Matching is on atom connectivity only.** Prediction and answer both go through RDKit tautomer canonicalization (pinned at 2026.03.3) and are reduced to the **first block of the InChIKey (InChIKey14)**; stereo and tautomer form do not matter (glucose example: two different SMILES reduce to the same key `WQZGKKKJIJFFOK`). A [metric notebook](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank) is public.

**Submission format:** `molecule_id,smiles`; up to 25 semicolon-joined SMILES, best first; every `molecule_id` exactly once; no nulls, no duplicates, no >25 lists. Fewer than 25 guesses is allowed; wrong guesses cost nothing beyond their rank.

Key operational consequences for the solution:

- Because only the *first* hit counts and wrong guesses are free, one should always emit 25 diverse, plausible candidates ([[concepts/competition/Novelty Classes|Novelty Classes]]).
- Matching is on tautomer-canonical InChIKey14, so the local scorer must use exactly that key ([[entities/RDKit]], [[concepts/solution/Evaluation Strategy|Evaluation Strategy]]).
- Submissions are made through Kaggle Notebooks with runtime and offline constraints ([[concepts/solution/Submission Architecture|Submission Architecture]]), with 5 submissions/day ([[summaries/kaggle-rules-timeline|rules summary]]).

See also: [[concepts/competition/Task and Evaluation]], [[concepts/competition/Test Set]].
