---
title: Kaggle Data Description
type: summary
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md]
tags: [competition, data, test-set]
---

# Summary — Kaggle Data Description

Task framing: identify an unknown molecule from its MS/MS spectra; molecules resemble natural products (confirmed or hypothesised NPs, NP analogs, plausibly-natural synthetics). **Predictions are per molecule, not per spectrum** — evidence from all of a molecule's spectra must be aggregated into one ranked list.

**Files:** `train.parquet` (~2.5M spectra, ~275k structures, one row per spectrum, label given); `test.parquet` (visible version is drawn from training examples; replaced by the hidden test at re-run, ~same size); `sample_submission.csv`.

**Test set:** ~1,500 spectra of ~400 molecules, 1–16 spectra per molecule (median 3), all on a Bruker timsTOF; masses 157–1,159 Da (median 348); ten adducts. Minimal cleaning only: precursor-vs-structure consistency, base peak ≥ 1,000 raw counts, peaks above precursor + 2 Da removed. Intensities normalised to base peak = 1.0.

**Three novelty classes** (hidden during the competition): 1 = public reference spectra exist; 2 = no spectra but structure in PubChem/COCONUT; 3 = not in PubChem at all → de novo. See [[concepts/competition/Novelty Classes]].

**Schema:** test has 12 columns (`molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities, base_peak_intensity, adduct, ionization_mode, instrument_type, precursor_mz, collision_energy_ev, collision_energy_orig, collision_energy_orig_units`); train adds `normalized_smiles` (the label), `inchikey`/`inchikey14`, `molecular_formula`, `ingest_lib`, `adduct_orig`, `precursor_error_ppm` (label-quality signal, ships uncleaned), `num_peaks`. Train `instrument_type` is free text; `base_peak_intensity` is null for pre-normalised libraries. Details: [[concepts/data/Train Schema]].

**Training libraries:** 11 sources with very different upstream preprocessing; exact duplicate spectra were removed but compounds recur across libraries. `enveda-180` (timsTOF, synthetic, instrument-matched) and `enveda-np-examples` (250 common NPs, closest to the test) are the only uniformly-timsTOF libraries. See [[concepts/data/Source Libraries]].

**Collision energy:** three columns; `collision_energy_ev` is the best-effort eV conversion designed to line up with the test (NCE converted via the nominal Thermo formula); use it if the model consumes CE. Multi-value entries = merged acquisitions.

**Suggested curation:** intensity floors (2%/1%/0.1%), drop peaks above precursor, min/max peak counts (top-N, or ~6 peaks per 50 Da window), deisotoping, sqrt/log transforms. See [[concepts/data/Spectrum Curation]].

**Pointed-to resources:** SIRIUS and MIST-CF (formula annotation), matchms, GNPS suspect annotations (excluded from train), MassIVE (unannotated corpus), DreaMS/GeMS (pretrained embeddings + corpus), FragHub (harmonised libraries).
