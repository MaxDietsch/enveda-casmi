---
title: Retrieval
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/max-class1-readme.md, raw/refs/stage2-plan.md, raw/refs/src-enveda-casmi26.md]
tags: [retrieval, similarity, entropy, library-search]
---

# Retrieval (Stage 1 — Library Search)

Retrieval finds the class-1 answer: the correct structure *does* have reference spectra in the training data. The implemented stage 1 (`max_class1_exp/stage1.py`, mirrored in the `sim` channel of `pipeline.py`):

1. **Query mass:** compute the molecule's neutral mass from the precursor mass and adduct of each spectrum (`pipeline.neutral_mass`); the molecule's query mass is the **median** over its spectra.
2. **Candidates:** every structure with a visible training row within **±10 ppm** of the query mass — matching on the *neutral* mass, so the answer is found even when the training data measured it only in a different adduct. Median 52 candidates, at most ~470; **the correct structure is among the candidates in 100% of cases** on the difficulty-1 split.
3. **Library:** all visible spectra of all candidates (every instrument, adduct and collision energy; ~467k spectra), cached.
4. **Cleaning** after Li et al. 2021: precursor-region removal, merging of peaks closer than 0.02 Da, 1% noise floor, scaling to sum 1 ([[concepts/data/Spectrum Curation]]).
5. **Similarity:** each query spectrum is compared with every library spectrum of every candidate; peaks match within tolerance, each query peak at most once per library spectrum. **Entropy similarity** (weighted): spectra with entropy $S < 3$ have their intensities raised to $0.25 + 0.25S$, then

$$\text{sim} = \frac{\sum_{\text{matched}} [(a+b)\ln(a+b) - a\ln a - b\ln b]}{\ln 4}$$

   which equals $1 - (2S(AB) - S(A) - S(B))/\ln 4$: 1 for identical spectra, 0 for spectra with no peak in common. **Cosine** similarity on the same peaks is the classic baseline.
6. **Aggregation:** a candidate's score is the mean over the query's spectra; ties break by smaller mass error, then by key.

**Four scoring variants:** `cosine_best` (best over the candidate's spectra), `entropy_best`, `entropy_composite` (against one merged spectrum per candidate and polarity — robust to collision-energy differences, and close to the test's merged spectra), `entropy_combined` (mean of best + composite).

**Measured baseline** (1000 difficulty-1 molecules): entropy_best reaches **MRR@25 0.765** overall — but when the *adduct* changed, only **0.393**, versus 0.933 for changed collision energy and 0.918 for changed instrument. Candidate recall is perfect, so **all remaining error is ranking error, concentrated in changed adducts**: an adduct shift moves every fragment mass, so literal peak-to-peak comparison misses true matches. The remedy is [[concepts/solution/Re-ranking|re-ranking]] with **neutral-loss-space** similarity: comparing $nl = \text{precursor} - \text{fragment}$ (equivalently, shifting library peaks by the precursor difference) cancels the adduct shift.

**Leakage guards:** the library contains only rows with `train_mask == True`, with an assertion that no evaluation row is included; the `--leak-eval` self-test (evaluation spectra deliberately placed in the library) must give MRR ≈ 1.
