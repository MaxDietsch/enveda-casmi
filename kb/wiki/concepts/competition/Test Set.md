---
title: Test Set
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md]
tags: [competition, test-set, timstof, adducts]
---

# Test Set

The hidden test set (the public `test.parquet` is a placeholder drawn from training examples and is replaced at scoring time):

- **Size:** ~1,500 spectra of **~400 molecules**, 1–16 spectra per molecule (median 3).
- **Instrument:** every spectrum comes from a Bruker timsTOF (`instrument_type` is always `timsTOF`) — one consistent, high-mass-accuracy platform ([[entities/Bruker timsTOF]]).
- **Masses:** neutral monoisotopic mass 157–1,159 Da, median 348.
- **Adducts:** exactly ten forms — `[M+H]+, [M+NH4]+, [M-H2O+H]+, [M-2H2O+H]+, [M+Na]+, [M+K]+, [M-H]-, [M-H2O-H]-, [M+CH2O2-H]-, [M+Cl]-`. An adduct is the small charged attachment that lets the instrument measure the molecule; it shifts all recorded masses.
- **Chemistry:** molecules resembling **natural products** — compounds produced by living organisms (plants, microbes, mammals), or synthetic analogues of such compounds. This differs from the synthetic, pharmaceutical-style chemistry that dominates the instrument-matched part of the training data ([[concepts/data/Source Libraries]]).
- **Collision energy:** always recorded in electron volts; a spectrum may be **merged** from several acquisition energies (stored as a list of values).
- **Cleaning applied by the organisers** (deliberately minimal — they did not want to discard information):
  1. spectra whose measured precursor mass disagreed with the known structure and adduct were removed;
  2. spectra whose strongest peak was below 1,000 raw counts were removed as likely noise;
  3. peaks above the precursor mass + 2 Da were removed (a fragment cannot be heavier than the molecule it came from);
  4. intensities were rescaled so the strongest peak equals 1.0.

**Per-molecule aggregation.** The same molecule can appear in several spectra (measured with different adducts and/or collision energies). The submission is a single ranked list per `molecule_id`, so a robust pipeline must combine the evidence across spectra (for example by averaging per-spectrum match scores, as in [[concepts/solution/Retrieval]]).

**The difficulty mix is hidden** — see [[concepts/competition/Novelty Classes]]; the repo's simulator assumes a 20% / 60% / 20% split ([[concepts/solution/Evaluation Strategy]]).
