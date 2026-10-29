---
title: Max Class-1 Experiment README
type: summary
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/max-class1-readme.md, raw/refs/stage2-plan.md, raw/refs/stage2-training-explained.md]
tags: [class-1, retrieval, reranking, experiment]
---

# Summary — Class-1 Experiment (`max_class1_exp/`)

Measures how well molecules of the easiest class can be identified: reference spectra exist in training, but measured in a *different setting* — another instrument, adduct, or collision energy. Two implemented stages over the simulator's difficulty-1 split; a third stage is planned.

**Stage 1 — library search.** Per molecule: neutral query mass (median over spectra, adduct-corrected); candidates = visible training structures within ±10 ppm (median 52; the correct structure is among the candidates 100% of the time); library = all visible spectra of those candidates (~467k). Cleaning after Li et al. 2021 (precursor-region removal, 0.02 Da peak merging, 1% noise floor, scaling to sum 1). Four scoring variants: `cosine_best`, `entropy_best` (entropy similarity with low-entropy weighting), `entropy_composite` (against one merged spectrum per candidate and polarity — robust to collision-energy differences), `entropy_combined` (mean of best + composite). A candidate's score is the mean over the query's spectra. Baseline (1000 molecules, mix 0.4/0.3/0.3): **entropy_best MRR@25 0.765 overall — changed adduct only 0.393, changed collision energy 0.933, changed instrument 0.918**. Candidate recall is perfect, so all error is ranking error, concentrated in changed adducts (an adduct shift moves every fragment mass, so literal peak comparison misses true matches).

**Stage 2 — learned re-ranking of the stage-1 shortlists.** ~16 deterministic features per (molecule, candidate) pair: the stage-1 scores, **neutral-loss-space** similarities (which cancel out adduct shifts), match statistics, |mass error in ppm|, setting flags (same adduct / polarity / instrument family), collision-energy gap, number of source libraries containing the molecule. Features are standardised **within each shortlist** (the model learns how candidates compare to their competitors); exactly one positive label per shortlist. Models: logistic regression and/or gradient-boosted trees; training examples are generated **at training time, in memory**, from the regular training set using the simulator's building blocks, under leakage rules R1–R5 (never a positive label for an evaluated molecule; evaluation-hidden rows invisible; shared feature function; seeded randomness). Success criteria: ≥ 0.765 overall, ≥ 0.5 on the changed-adduct case.

**Validation culture:** unit tests (22 + 24 checks), `--leak-eval` / `--leak-train` self-tests (deliberate leakage must yield MRR ≈ 1), `--train-on-eval` as a diagnostic of how much label leakage would inflate scores.

**Stage 3 (planned):** a Siamese spectrum-embedding network trained on cross-setting pairs (~2,500 molecules measured on timsTOF and other instruments, ~69,000 with several adducts), compared against pretrained DreaMS embeddings. Further ideas: polarity handling, query merging, parameter sweeps, per-source-library reporting, **confidence gating** calibrated on difficulty-1 versus difficulty-2 splits for the final combined submission.
