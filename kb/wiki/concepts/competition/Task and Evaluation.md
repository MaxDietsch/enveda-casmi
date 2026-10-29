---
title: Task and Evaluation
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-competition-overview.md, raw/articles/kaggle-rules.md]
tags: [competition, metric, mrr, submission-format]
---

# Task and Evaluation

**Task.** For each `molecule_id` in the test set — a group of tandem-mass-spectrometry spectra of one unknown molecule — output up to 25 candidate structures as **SMILES** strings (a text encoding of a molecular structure), best guess first, joined by semicolons in a single field. Predictions are **per molecule**, so the evidence from all of a molecule's spectra must be combined into one ranked list ([[concepts/competition/Test Set]]).

**Metric — MRR@25 (mean reciprocal rank, top 25).** A molecule earns the reciprocal of the position of the **first** correct guess: 1.0 for position 1, 0.5 for position 2, 0.04 for position 25, and 0 if no guess is correct.

$$\text{MRR@25} = \frac{1}{U} \sum_{u=1}^{U} \frac{1}{rank_u}$$

Only the first correct guess counts; further copies of the correct answer add nothing.

**Matching rule.** A prediction is correct when it describes the same flat molecular skeleton as the answer: both the predicted SMILES and the answer are passed through RDKit's tautomer standardisation (RDKit version pinned at 2026.03.3) and reduced to the **first block of the InChIKey (InChIKey14)**, then compared. Stereochemistry and tautomer form do not matter (two different-looking SMILES of glucose can reduce to the same key). Practical consequences:

- Never submit several tautomers or stereo-variants of one candidate — they have identical InChIKey14s and waste list positions.
- Local scoring must use exactly this key: `TautomerEnumerator().Canonicalize(mol)` → `MolToInchiKey()[:14]` ([[concepts/solution/Evaluation Strategy]]).
- A structure with the same molecular formula is *not* automatically correct, but same-formula structures are the most convincing wrong candidates ("decoys"; [[concepts/solution/Re-ranking]]).

**Submission format.** Header `molecule_id,smiles`; every `molecule_id` exactly once; no empty fields, no repeated ids, at most 25 guesses per row. Submitting fewer than 25 guesses is allowed, and a wrong guess costs nothing beyond the position it occupies — so **always fill all 25 positions** with distinct, plausible candidates, ordered by confidence.

**Submission limit.** 5 per day; winners must publish their code under an open-source licence. Reference implementation: Kaggle [metric notebook](https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank).
