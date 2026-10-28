---
title: Submission Architecture
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-rules.md, raw/refs/src-enveda-casmi26.md, raw/refs/scripts-overview.md, raw/refs/max-class1-readme.md]
tags: [kernel, submission, kaggle, constraints]
---

# Submission Architecture

The competition is a **code competition**: scoring happens when Kaggle re-runs a notebook against the hidden test. Hard constraints from the rules:

- submit via Kaggle Notebooks; ≤ **9 hours** run-time (CPU or GPU);
- **internet disabled** at inference — every external asset (COCONUT pool, pretrained weights, fingerprints) must be attached as a Kaggle dataset;
- freely & publicly available external data and pretrained models are allowed;
- 5 submissions/day; final deadline Dec 14, 2026.

**How the repo is shaped by this:**

1. **Single-source-of-truth modules.** `fp_model.py` is mirrored into the Kaggle dataset as `code_fp.py` so kernel and local training share definitions; `export_scorer_json.py` serialises the learned scorer to JSON for kernel consumption (no sklearn dependency surprises offline).
2. **Kernel-path scripts:** `smoke_test_kernel.py` and `validate_kernel.py` exercise the notebook pipeline locally; `evaluate_kernel_output.py` scores a kernel's output against local splits; `simulate_holdout.py` mimics the re-run behaviour (visible `test.parquet` placeholder replaced by a hidden set of ~same size).
3. **Versioned pipelines.** Dump scripts (`_dump_v17.py`, `_dump_tutorial.py`) snapshot pipeline versions; the fusion analysis (`analyze_fusion.py`) measures channel combinations, and `make_submission.py` is the minimal fallback (cosine library search + nearest-neutral-mass filler up to 25 slots).
4. **Channel fusion** ([[concepts/solution/index|architecture]]): per-molecule candidate sets are z-scored per channel (`mass`, `sim`, `eif`, `learned`) and summed per configuration; `pipeline.py` is the same code locally and in the kernel.
5. **Class-agnostic ordering.** Classes are hidden, so the final 25-slot list must blend channels behind calibrated confidence: strong library hits on top, database/de novo candidates filling the rest. The class-1 experiment's `top_score_*` confidence gating — calibrated on difficulty 1 vs difficulty 2 splits — is the planned mechanism ([[concepts/solution/Roadmap]]).

**Slot hygiene:** every `molecule_id` exactly once; ≤ 25 unique-InChIKey14 candidates (tautomer duplicates waste slots); invalid generated SMILES filtered before writing the CSV ([[concepts/competition/Task and Evaluation]]).
