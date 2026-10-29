---
title: Kaggle Data Description
type: summary
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md]
tags: [competition, data, test-set]
---

# Summary — Kaggle Data Description

Task framing: identify an unknown molecule from its tandem-mass spectra; the molecules resemble natural products (compounds produced by living organisms, or synthetic analogues). **Predictions are per molecule, not per spectrum** — the evidence from all of a molecule's spectra must be combined into one ranked list.

**Files:** `train.parquet` (~2.5M spectra, ~275k structures, one row per spectrum, structure given); `test.parquet` (the visible version is drawn from training examples; it is replaced by the hidden test at scoring time, at roughly the same size); `sample_submission.csv`.

**Test set:** ~1,500 spectra of ~400 molecules, 1–16 spectra per molecule (median 3), all measured on a Bruker timsTOF; masses 157–1,159 Da (median 348); ten adduct forms. Only minimal cleaning was applied: precursor-structure consistency, strongest peak ≥ 1,000 raw counts, no peaks above precursor + 2 Da; intensities scaled so the strongest peak equals 1.0.

**Three novelty classes** (hidden during the competition): 1 = public reference spectra exist; 2 = no spectra but the structure is listed in PubChem/COCONUT; 3 = not listed in PubChem at all → must be generated. See [[concepts/competition/Novelty Classes]].

**Schema:** the test file has 12 columns (`molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities, base_peak_intensity, adduct, ionization_mode, instrument_type, precursor_mz, collision_energy_ev, collision_energy_orig, collision_energy_orig_units`); train adds `normalized_smiles` (the target), `inchikey`/`inchikey14`, `molecular_formula`, `ingest_lib`, `adduct_orig`, `precursor_error_ppm` (a label-quality signal, delivered unfiltered), `num_peaks`. Train `instrument_type` is free text; `base_peak_intensity` is missing for libraries that delivered pre-scaled intensities. Details: [[concepts/data/Train Schema]].

**Training libraries:** 11 sources with very different upstream preprocessing; exactly duplicated spectra were removed, but the same molecule measured by different libraries remains. `enveda-180` (timsTOF, synthetic, instrument-matched) and `enveda-np-examples` (250 common natural products, closest to the test) are the only consistently timsTOF-labelled libraries. See [[concepts/data/Source Libraries]].

**Collision energy:** three columns; `collision_energy_ev` is the best-effort conversion designed to match the test (NCE converted via the nominal Thermo formula); models that use collision energy should use this column. Multiple values mean the spectrum was merged from several acquisition energies.

**Suggested curation:** intensity floors (2%/1%/0.1%), removing peaks above the precursor, minimum/maximum peak counts (top-N, or ~6 peaks per 50 Da window), removing carbon-13 satellite peaks, square-root/log intensity transforms. See [[concepts/data/Spectrum Curation]].

**Pointed-to resources:** SIRIUS and MIST-CF (formula prediction), matchms, GNPS propagated annotations (excluded from train because the labels are inferred), MassIVE (public repository of spectra without identities), DreaMS/GeMS (pretrained embeddings + corpus), FragHub (harmonised library aggregation).
