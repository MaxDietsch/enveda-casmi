---
title: Repo Solution Code (src/ and scripts/)
type: summary
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/src-enveda-casmi26.md, raw/refs/scripts-overview.md, raw/refs/repo-readme.md]
tags: [solution, pipeline, code-map]
---

# Summary — Repo Solution Code

The repo contains a **two-branch solution**: (A) the `src/enveda_casmi26/` multi-channel retrieval pipeline with in-silico fragmentation, and (B) the `max_class1_exp/` class-1 experiment ([[summaries/max-class1-readme]]), plus `dataset-creation/` for evaluation ([[summaries/dataset-creation-readme]]).

**`src/enveda_casmi26/` modules:**
- `cleaning.py` — 7-step spectrum curation: precursor-region cut, relative-intensity floor, removal of carbon-13 satellite peaks, a mass-defect artefact filter (singly-charged organic fragment ions below ~350 Da never show certain decimal ranges), top-K peaks per mass window and/or a global cap, square-root/log intensity transforms, rescaling to strongest peak = 1. Spectra are flagged rather than silently removed.
- `fragmentation.py` — in-silico fragmentation: software that predicts which fragments a candidate structure would produce (single cuts of non-ring bonds, double cuts of ring bonds, hydrogen rearrangements, neutral-loss cascades, adduct-aware charge handling) and checks them against observed peaks at ppm tolerance.
- `pipeline.py` — the end-to-end task: group spectra into molecules, retrieve candidates by **neutral-mass window**, rank them with evidence channels (**mass**, **sim** — spectral similarity, **eif** — the share of observed peak intensity a candidate's predicted fragments explain, **learned** — a trained scorer's probability), each standardised within the molecule's candidate set and summed with configurable weights; metric MRR@25 on the InChIKey14.
- `scoring.py` — the candidate-level scorer: individual match features barely separate the true structure from coincidental candidates, so features are aggregated per candidate and a logistic model learns P(candidate is correct | features).
- `fp_model.py` — the spectrum→SMILES transformer encoder plus a fingerprint-prediction head; **single source of truth mirrored into the Kaggle dataset** as `code_fp.py`.
- `tracking.py` — MLflow experiment-tracking helpers (local SQLite by default).

**`scripts/` roles (36):** data (`download_data`, `clean_data`, `make_split` — a leak-free validation split that removes whole structures: `train_cut`/`test_cut`/`pool_extra`), baselines (`similarity_baseline`, `make_submission`), retrieval infrastructure (`build_coconut_pool` — standardised COCONUT extract for scoring time and Kaggle-dataset upload; `build_pool_fps`), fragmenter tuning (`mine_ce_priors`, `mine_losses`, `mine_unexplained`, `collect_matches`, `benchmark_decoys`, `rescore_tautomer` — strict versus formula versus tautomer-canonical scoring), learned channels (`train_denovo`, `eval_denovo`, `train_fp_head`, `train_scorer`, `train_fp_channel`, `analyze_fusion`), the notebook path (`export_scorer_json`, `evaluate_kernel_output`, `smoke_test_kernel`, `validate_kernel`, `simulate_holdout`) and diagnostics (`explain_spectrum`, `analyze_failures`, `profile_spectra`).

Experiment tracking via MLflow; runs are logged locally to `mlflow.db` by default.
