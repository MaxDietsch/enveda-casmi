---
title: Test Set
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md]
tags: [competition, test-set, timstof, adducts]
---

# Test Set

The hidden test (the public `test.parquet` is a placeholder drawn from training examples, replaced at re-run):

- **Size:** ~1,500 spectra of **~400 molecules**, 1–16 spectra per molecule (median 3).
- **Instrument:** 100% Bruker timsTOF (`instrument_type` always `timsTOF`) — a Q-TOF platform ([[entities/Bruker timsTOF]]).
- **Masses:** neutral monoisotopic mass 157–1,159 Da, median 348.
- **Adducts:** exactly ten: `[M+H]+, [M+NH4]+, [M-H2O+H]+, [M-2H2O+H]+, [M+Na]+, [M+K]+, [M-H]-, [M-H2O-H]-, [M+CH2O2-H]-, [M+Cl]-`.
- **Chemistry:** natural-product-like — confirmed/hypothesised natural products, NP analogs, plausibly-natural synthetics. This contrasts with the synthetic drug-like slant of the biggest timsTOF training library ([[concepts/data/Source Libraries]]).
- **Collision energy:** always in eV (`collision_energy_orig_units` = `eV`); `collision_energy_ev` is a list — a single value is one acquisition, several values mean the spectrum is **merged** from multiple energies.
- **Cleaning applied by the organizers (minimal, deliberately lossless):**
  1. spectra whose measured precursor mass was inconsistent with the known structure and adduct were dropped;
  2. base peak < 1,000 raw counts dropped as likely noise;
  3. peaks above precursor + 2 Da removed (a fragment cannot outweigh its singly-charged precursor);
  4. intensities normalised so the max is 1.0.

**Per-molecule aggregation.** The same molecule can appear as several spectra (different adducts and/or CEs). The submission is one ranked list per `molecule_id`; a robust pipeline combines the per-spectrum evidence (e.g. mean match score, as in [[concepts/solution/Retrieval]]).

**Class mix is hidden** — see [[concepts/competition/Novelty Classes]]; the repo's simulator approximates it as 20/60/20 ([[concepts/solution/Evaluation Strategy]]).
