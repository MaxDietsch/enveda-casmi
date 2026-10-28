---
title: Roadmap
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/max-class1-readme.md, raw/refs/dataset-creation-improvements.md, raw/refs/analysis-stats.md, raw/articles/kaggle-data-description.md]
tags: [roadmap, ideas, open-questions]
---

# Roadmap

Collected, not-yet-implemented ideas from the repo's own docs, ordered by branch.

**Class-1 experiment — stage 3 and beyond:**
- **Learned cross-instrument spectrum embedding** (stage 3): a Siamese network in the MS2DeepScore style trained on positive pairs = same molecule, different instrument/adduct/CE. Train has ~2,500 bridge molecules (timsTOF + other instruments), ~69,000 with ≥ 2 adducts, ~178,000 with ≥ 2 CEs in one adduct. The embedding similarity becomes a stage-2 feature. Compare against pretrained **DreaMS** embeddings as a shortcut ([[entities/DreaMS]], [[entities/MS2DeepScore]]).
- **Polarity handling:** compare only same-polarity library spectra, or learn the `[M+H]+` → `[M-H]-` mapping.
- **Query merging:** merge the query's own spectra into one composite (mirror of `entropy_composite`).
- **Parameter sweeps:** `--tol` (0.01/0.02/0.05), `--noise`, top-N peaks, `--ppm`.
- **Confidence gating for the merged test:** calibrate `top_score_*` on difficulty 1 vs difficulty 2 (where the answer is never in the library) so trustworthy library hits rank on top and database/de novo candidates follow — the class-agnostic ordering for the final submission.
- **Error bars:** stage 1 on several difficulty-1 seeds.

**Simulator gaps** (`POTENTIAL_IMPROVEMENTS.md`): spectra-per-molecule below test rate (fix: draw `k` first, then sample among molecules with ≥ k candidate rows); synthetic chemistry (ideas: NP-likeness-weighted sampling, COCONUT membership proxy, accepting Q-TOF spectra from `riken`/`gnps`/`mona`/`msdial` as eval spectra since timsTOF is a Q-TOF); residual analogues (report scores by `max_visible_tc` band); fingerprint collisions (count-based Morgan or radius 3); rare adducts; multi-seed score noise.

**Data-side ideas:** report every metric **per source library** (`ingest_lib`) to expose the NP-vs-synthetic gap; mine MassIVE/GeMS unannotated spectra for self-supervised pretraining (official description points at GeMS/DreaMS); GNPS "suspect" annotations are excluded from train but the technique (network propagation) is fair game as an idea.

**Fragmenter channel:** `mine_ce_priors.py` (CE-bucketed cleavage/neutral-loss priors), `mine_losses.py` / `mine_unexplained.py` (discovering fragmentation rules from unexplained peaks), `benchmark_decoys.py` (true + 9 same-formula decoys) — all feed the `eif` channel's discrimination tuning.

**Open questions** tracked in CLAUDE.md: adduct-shift matching headroom; best class-3 strategy per engineering hour; hit-confidence calibration; closing the chemistry gap. Sources to ingest next: Li et al. 2021 (entropy similarity), MS2DeepScore, DreaMS, CASMI 2026 official pages.
