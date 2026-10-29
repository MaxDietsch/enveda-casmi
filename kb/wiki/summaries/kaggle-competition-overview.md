---
title: Kaggle Competition Overview
type: summary
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-competition-overview.md]
tags: [competition, metric, submission]
---

# Summary — Kaggle Competition Overview

The competition asks for machine-learning models that predict **2D molecular structures (as SMILES text) from tandem-mass spectra**, including for molecules never measured before — motivated by the fact that most molecules detected in biological samples are absent from reference libraries. Applications: drug discovery and disease biomarkers.

**Metric: MRR@25.** For each molecule, only the **first** correct guess among up to 25 ranked SMILES counts: position 1 scores 1.0, position 2 scores 0.5, position 25 scores 0.04, no correct guess scores 0. Averaged over all molecules:

$$\text{MRR@25} = \frac{1}{U} \sum_{u=1}^{U} \frac{1}{rank_u}$$

**Matching is on the flat skeleton only.** Prediction and answer both undergo RDKit tautomer standardisation (pinned at 2026.03.3) and are reduced to the **first block of the InChIKey (InChIKey14)**; stereochemistry and tautomer form do not matter (two different-looking SMILES of glucose reduce to the same key). A [metric notebook](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank) is public.

**Submission format:** `molecule_id,smiles`; up to 25 semicolon-joined SMILES, best guess first; every `molecule_id` exactly once; no empty fields, no duplicate ids, no lists longer than 25. Fewer than 25 guesses is allowed; a wrong guess costs nothing beyond the position it occupies.

Key operational consequences for the solution:

- Only the *first* correct answer counts and wrong guesses are free, so one should always output 25 distinct, plausible candidates ([[concepts/competition/Novelty Classes|Novelty Classes]]).
- Matching uses the tautomer-standardised InChIKey14, so the local scorer must use exactly that key ([[entities/RDKit]], [[concepts/solution/Evaluation Strategy|Evaluation Strategy]]).
- Submissions run as Kaggle notebooks under strict runtime and offline constraints ([[concepts/solution/Submission Architecture|Submission Architecture]]), with 5 submissions per day ([[summaries/kaggle-rules-timeline|rules summary]]).

See also: [[concepts/competition/Task and Evaluation]], [[concepts/competition/Test Set]].
