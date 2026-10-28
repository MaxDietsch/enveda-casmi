---
title: Repo Solution Code (src/ and scripts/)
type: summary
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/src-enveda-casmi26.md, raw/refs/scripts-overview.md, raw/refs/repo-readme.md]
tags: [solution, pipeline, code-map]
---

# Summary — Repo Solution Code

The repo contains a **two-branch solution**: (A) the `src/enveda_casmi26/` multi-channel retrieval pipeline with in-silico fragmentation, and (B) the `max_class1_exp/` class-1 experiment (see [[summaries/max-class1-readme]]), plus `dataset-creation/` for evaluation ([[summaries/dataset-creation-readme]]).

**`src/enveda_casmi26/` modules:**
- `cleaning.py` — 7-step curation: precursor-region cut, relative-intensity floor, deisotoping (C-13 companions ~1.0033 Da), mass-defect junk filter, windowed top-K / global cap, sqrt/log transforms, renormalisation; never silently drops spectra.
- `fragmentation.py` — in-silico MS2: acyclic single cuts, ring double cuts (fissions), H rearrangements, neutral-loss cascades, adduct-aware charge states; matches theoretical fragments to observed peaks at ppm tolerance.
- `pipeline.py` — the end-to-end task: group spectra into molecules, retrieve candidates by **neutral-mass window**, rank with evidence channels (**mass**, **sim** spectral similarity, **eif** explained-intensity fraction from the fragmenter, **learned** CandidateScorer probability), z-scored per channel within each molecule's candidate set and summed per configuration; metric MRR@25 on inchikey14.
- `scoring.py` — Tier-4 candidate-level scorer: per-match features barely separate truth from decoys, so features aggregate per candidate and a logistic model learns P(true | features).
- `fp_model.py` — de novo transformer encoder + Morgan fingerprint head; **single source of truth mirrored into the Kaggle dataset** as `code_fp.py`.
- `tracking.py` — MLflow helpers (local sqlite default).

**`scripts/` roles (36):** data (`download_data`, `clean_data`, `make_split` — leak-free train_cut/test_cut/pool_extra split holding out whole structures), baselines (`similarity_baseline`, `make_submission`), retrieval infrastructure (`build_coconut_pool` — COCONUT canonicalization for kernel + Kaggle dataset upload; `build_pool_fps`), fragmenter tuning (`mine_ce_priors`, `mine_losses`, `mine_unexplained`, `collect_matches`, `benchmark_decoys`, `rescore_tautomer` — strict vs formula vs tautomer scoring), learned channels (`train_denovo`, `eval_denovo`, `train_fp_head`, `train_scorer`, `train_fp_channel`, `analyze_fusion`), kernel path (`export_scorer_json`, `evaluate_kernel_output`, `smoke_test_kernel`, `validate_kernel`, `simulate_holdout`) and diagnostics (`explain_spectrum`, `analyze_failures`, `profile_spectra`).

Experiment tracking via MLflow; runs logged locally to `mlflow.db` by default.
