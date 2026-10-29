---
title: Roadmap
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/max-class1-readme.md, raw/refs/dataset-creation-improvements.md, raw/refs/analysis-stats.md, raw/articles/kaggle-data-description.md]
tags: [roadmap, ideas, open-questions]
---

# Roadmap

Collected but not-yet-implemented ideas from the repo's own documents, ordered by development branch.

**Class-1 experiment — stage 3 and beyond:**
- **Learned cross-instrument spectrum embedding** (stage 3): a two-tower ("Siamese") neural network, in the style of MS2DeepScore, trained on positive pairs = same molecule, different measurement setting. Training pairs are available: ~2,500 molecules measured on timsTOF *and* other instruments, ~69,000 molecules in ≥ 2 adducts, ~178,000 with ≥ 2 collision energies in one adduct. The embedding similarity becomes a re-ranking feature. Compare against pretrained **DreaMS** embeddings as the pretrained alternative ([[entities/DreaMS]], [[entities/MS2DeepScore]]).
- **Polarity handling:** compare only same-polarity library spectra, or learn the mapping between positive- and negative-mode spectra of the same molecule.
- **Query merging:** merge the query's own spectra into one composite spectrum (the mirror of `entropy_composite`).
- **Parameter sweeps:** peak-match tolerance (0.01 / 0.02 / 0.05 Da), noise floor, peak caps, mass window in ppm.
- **Confidence gating for the merged test:** calibrate the top match score on difficulty-1 versus difficulty-2 splits (where the answer is provably not in the library), so that trustworthy reference-spectrum matches rank first and database and generated candidates follow — the class-agnostic ordering for the final submission.
- **Uncertainty estimates:** run stage 1 on several evaluation seeds.

**Simulator limitations** (`POTENTIAL_IMPROVEMENTS.md`): fewer spectra per molecule than the test (proposed fix: draw the desired count first, then sample among molecules that have that many candidate rows); synthetic chemistry (ideas: weight sampling toward natural-product-likeness, using COCONUT membership as an indicator; accept Q-TOF spectra from `riken`/`gnps`/`mona`/`msdial` as evaluation spectra, since timsTOF is a Q-TOF and fragments similarly); remaining structurally similar training molecules (report scores by `max_visible_tc` band); fingerprint collisions between distinct molecules (use count-based fingerprints or a larger radius); rare adducts; score noise across seeds.

**Data-side ideas:** report every metric **per source library** (`ingest_lib`) to expose the natural-products-versus-synthetic difference; use the unannotated public spectra corpora (MassIVE / GeMS) for self-supervised pretraining (the official description points at GeMS/DreaMS); network-propagation of annotations (the GNPS "suspect" technique) is worth exploring even though such inferred labels were excluded from train.

**Fragmenter channel:** `mine_ce_priors.py` (collision-energy-dependent cleavage statistics), `mine_losses.py` / `mine_unexplained.py` (discovering new fragmentation rules from unexplained peaks), `benchmark_decoys.py` (the correct structure plus 9 same-formula wrong candidates) — all feed the tuning of the in-silico fragmentation channel.

**Open questions** tracked in CLAUDE.md: how much ranking error the changed-adduct case still holds after neutral-loss re-ranking; the best class-3 approach per unit of development effort; confidence calibration for the merged ordering; closing the chemistry gap. Sources to read next: Li et al. 2021 (entropy similarity), MS2DeepScore, DreaMS, and the CASMI 2026 pages.
