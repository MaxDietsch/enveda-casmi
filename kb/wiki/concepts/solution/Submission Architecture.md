---
title: Submission Architecture
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-rules.md, raw/refs/src-enveda-casmi26.md, raw/refs/scripts-overview.md, raw/refs/max-class1-readme.md]
tags: [kernel, submission, kaggle, constraints]
---

# Submission Architecture

This is a **code competition**: Kaggle re-runs a notebook against the hidden test and scores its output. Hard constraints from the rules:

- submissions run as Kaggle **notebooks** with at most **9 hours** of compute (CPU or GPU);
- **no internet access** at scoring time — every external asset (the COCONUT extract, pretrained model weights, precomputed fingerprints) must be uploaded as a Kaggle dataset and attached;
- freely and publicly available external data and pretrained models are allowed;
- 5 submissions per day; final deadline Dec 14, 2026.

**How the repo is shaped by this:**

1. **Single-source-of-truth modules.** `fp_model.py` is mirrored into the Kaggle dataset as `code_fp.py` so the notebook and local training share identical definitions; `export_scorer_json.py` serialises the trained scorer to JSON so the notebook needs no Python library beyond its environment.
2. **Notebook-path scripts** (the repo calls the notebook "the kernel"): `smoke_test_kernel.py` and `validate_kernel.py` exercise the notebook pipeline locally; `evaluate_kernel_output.py` scores a notebook's output against local splits; `simulate_holdout.py` mimics the re-run behaviour (the visible `test.parquet` placeholder is replaced by a hidden set of roughly the same size).
3. **Versioned pipelines.** Snapshot scripts (`_dump_v17.py`, `_dump_tutorial.py`) record pipeline versions; `analyze_fusion.py` measures channel combinations; `make_submission.py` is the minimal fallback (similarity ranking plus nearest-neutral-mass candidates to fill all 25 positions).
4. **Channel fusion** ([[concepts/solution/index|architecture]]): each molecule's candidate scores are standardised per channel (`mass`, `sim`, `eif` — explained intensity from the in-silico fragmenter — and `learned`) and summed with configurable weights; `pipeline.py` is the same code locally and in the notebook.
5. **Class-agnostic ordering.** The difficulty classes are hidden, so the final 25-position list must blend all channels behind calibrated confidence: strong reference-spectrum matches first, database and generated candidates filling the rest. The class-1 experiment's score-based confidence gating — calibrated on difficulty-1 versus difficulty-2 evaluation sets — is the planned mechanism ([[concepts/solution/Roadmap]]).

**Submission-list hygiene:** every `molecule_id` exactly once; at most 25 candidates with distinct InChIKey14s (tautomer duplicates waste positions); invalid generated SMILES filtered out before writing the CSV ([[concepts/competition/Task and Evaluation]]).
