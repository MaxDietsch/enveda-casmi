---
title: Max Class-1 Experiment README
type: summary
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/max-class1-readme.md, raw/refs/stage2-plan.md, raw/refs/stage2-training-explained.md]
tags: [class-1, retrieval, reranking, experiment]
---

# Summary — Class-1 Experiment (`max_class1_exp/`)

Measures how well class-1-style molecules (reference spectra exist, but in a different instrument/adduct/CE setting) can be identified. Two implemented stages over the difficulty-1 split from the simulator; stage 3 planned.

**Stage 1 — library search.** Per molecule: neutral query mass (median over spectra, adduct-corrected); candidates = visible train molecules within ±10 ppm (median 52, recall 1.0); library = all visible spectra of candidates (~467k). Cleaning after Li et al. 2021 (precursor-region removal, 0.02 Da greedy centroiding, 1% noise floor, sum-normalisation). Four scoring variants: `cosine_best`, `entropy_best` (entropy similarity with low-entropy weighting), `entropy_composite` (against a per-candidate-and-polarity merged spectrum — CE-robust), `entropy_combined` (mean of best + composite). Candidate score = mean over query spectra. Baseline (1000 molecules, mix 0.4/0.3/0.3): **entropy_best MRR@25 0.765 overall — adduct 0.393, ce 0.933, instrument 0.918**. All error is ranking error, concentrated in `adduct` (adduct shifts fragment m/z, so literal peak matching misses).

**Stage 2 — learned re-ranker over stage-1 shortlists.** ~16 deterministic features per (molecule, candidate) pair: stage-1 scores, **neutral-loss-space** similarities (survive adduct changes), match statistics, |ppm|, setting flags (same adduct/polarity/instrument family), CE gap, source-library count. Features are z-scored **within each shortlist** (contrast trick); labels y = (candidate == truth), exactly one positive per shortlist. Models: logistic regression and/or HistGradientBoosting; trained **on the fly** from the regular train set using the simulator building blocks, with leakage rules R1–R5 (never a positive label for an eval molecule; split-hidden rows invisible; shared featurizer; seeded). Success criteria: ≥ 0.765 overall, adduct ≥ 0.5.

**Validation culture:** unit tests (22 + 24 checks), `--leak-eval`/`--leak-train` self-tests (MRR ≈ 1), `--train-on-eval` as a leakage-delta diagnostic.

**Stage 3 (planned):** MS2DeepScore-style Siamese spectrum embedding trained on cross-setting pairs (~2.5k bridge molecules, ~69k multi-adduct) as an additional feature; DreaMS pretrained embeddings as a shortcut. Other ideas: polarity handling, query merging, parameter sweeps, per-source-library reporting, **confidence gating** calibrated on difficulty 1 vs 2 for the merged submission.
