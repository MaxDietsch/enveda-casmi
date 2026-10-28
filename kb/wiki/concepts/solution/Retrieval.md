---
title: Retrieval
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/max-class1-readme.md, raw/refs/stage2-plan.md, raw/refs/src-enveda-casmi26.md]
tags: [retrieval, similarity, entropy, library-search]
---

# Retrieval (Stage 1 — Library Search)

Retrieval finds the class-1 answer among training structures. The implemented stage 1 (`max_class1_exp/stage1.py`, mirrored in `pipeline.py`'s `sim` channel):

1. **Query mass:** neutral mass from precursor m/z + adduct per spectrum (`pipeline.neutral_mass`); the molecule's query mass is the **median** over its spectra.
2. **Candidates:** every molecule with a visible train row within **±10 ppm** of the query mass — neutral-mass matching, so the answer is found even when train only has it in another adduct. Median 52 candidates, max ~470; **candidate recall 1.0** on the difficulty-1 split.
3. **Library:** all visible spectra of all candidates (all instruments/adducts/CEs; ~467k spectra), cached.
4. **Cleaning** after Li et al. 2021: precursor-region removal, 0.02 Da greedy centroiding, 1% noise floor, sum-normalisation ([[concepts/data/Spectrum Curation]]).
5. **Similarity:** each query spectrum vs every library spectrum of every candidate; peaks match within tolerance, one match per query peak. **Entropy similarity** (weighted): spectra with entropy $S < 3$ get intensities raised to $0.25 + 0.25S$, then

$$\text{sim} = \frac{\sum_{\text{matched}} [(a+b)\ln(a+b) - a\ln a - b\ln b]}{\ln 4}$$

   equivalent to $1 - (2S(AB) - S(A) - S(B))/\ln 4$. **Cosine** on the same peaks is the classic baseline.
6. **Aggregation:** candidate score = mean over the query's spectra; ties broken by smaller mass error, then key.

**Four scoring variants:** `cosine_best` (best over candidate's spectra), `entropy_best`, `entropy_composite` (vs one merged spectrum per candidate+polarity — robust to CE differences, similar to the test's merged-CE spectra), `entropy_combined` (mean of best + composite).

**Measured baseline** (1000 difficulty-1 molecules): entropy_best **MRR@25 0.765** overall — but `adduct` change only **0.393** vs `ce` 0.933, `instrument` 0.918. Recall is perfect; **all error is ranking error, concentrated in adduct changes**, because an adduct shift moves fragment m/z so literal peak matching misses true matches. The fix is [[concepts/solution/Re-ranking|re-ranking]] with **neutral-loss-space** similarity: $nl = \text{precursor} - \text{fragment}$, algebraically the "modified cosine" with library peaks shifted by the precursor difference — the signal that survives an adduct change.

**Leakage guards:** library restricted to `train_mask == True` rows, asserted no eval row inside; `--leak-eval` self-test must give MRR ≈ 1.
