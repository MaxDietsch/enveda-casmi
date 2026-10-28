---
title: Task and Evaluation
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-competition-overview.md, raw/articles/kaggle-rules.md]
tags: [competition, metric, mrr, submission-format]
---

# Task and Evaluation

**Task.** For each `molecule_id` in the test set (a group of MS/MS spectra of one unknown molecule), output up to 25 candidate structures as SMILES, best guess first, semicolon-joined in a single field. Predictions are **per molecule**, so evidence from all spectra of a molecule must be aggregated ([[concepts/competition/Test Set]]).

**Metric — MRR@25.**

$$\text{MRR@25} = \frac{1}{U} \sum_{u=1}^{U} \frac{1}{rank_u}$$

where $rank_u$ is the position of the **first** correct structure for molecule $u$; a molecule with no correct guess scores 0. Only the first hit counts — later duplicates of the truth add nothing.

**Matching key.** A prediction is correct when it has the same **atom connectivity** as the answer: both SMILES pass through RDKit's tautomer canonicalization (pinned at **2026.03.3**) and are reduced to the **first block of the InChIKey (InChIKey14)**, then compared. Stereo and tautomer form are free. Practical consequences:

- Never emit multiple tautomers/stereoisomers of the same candidate — they waste slots (identical InChIKey14).
- Local scoring must use the identical key: `TautomerEnumerator().Canonicalize(mol)` → `MolToInchiKey()[:14]` ([[concepts/solution/Evaluation Strategy]]).
- A formula-level match is *not* a hit, but same-formula structures are the strongest decoys ([[concepts/solution/Re-ranking]]).

**Submission format.** `molecule_id,smiles` header; every `molecule_id` exactly once; no nulls, no repeated ids, ≤ 25 guesses per row. Fewer than 25 guesses is allowed and wrong guesses are free beyond the rank they occupy → **always fill all 25 slots** with diverse, plausible candidates, ordered by confidence.

**Quota.** 5 submissions/day; winning requires open-sourcing under an OSI license. Metric reference: Kaggle [metric notebook](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank).
