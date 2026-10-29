---
title: Glossary
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md, raw/articles/kaggle-competition-overview.md]
tags: [glossary, reference]
---

# Glossary

Plain-language definitions of the recurring terms in this knowledge base. Chemistry terms are explained; basic machine-learning terminology (features, logistic regression, embeddings) is used without gloss.

- **MS/MS spectrum (plural: spectra)** — the record a mass spectrometer produces of one measurement: a list of fragment masses paired with their relative intensities (signal strengths), with the strongest peak scaled to 1.0.
- **Fragment** — a piece of a molecule after the instrument has broken it apart.
- **Precursor** — the intact molecule, charged and prepared for measurement. Its **precursor m/z** is the measured mass-to-charge ratio of that intact ion.
- **Adduct** — the small charged attachment added to a molecule so the instrument can measure it (a proton `[M+H]+`, a sodium ion `[M+Na]+`, a loss of water, etc.). The adduct shifts all measured masses, so the same molecule in different adducts produces shifted spectra.
- **Collision energy** — how forcefully the instrument breaks the molecule apart. Higher energy produces more, smaller fragments. Reported in electron volts (**eV**) or as a vendor-specific relative scale (**NCE**, "normalised collision energy").
- **Merged spectrum** — a spectrum combining acquisitions at several collision energies into one list of peaks.
- **timsTOF** — a specific commercial mass-spectrometer platform (Bruker). All hidden-test spectra come from it.
- **Orbitrap / Q-TOF / ion trap** — families of mass-spectrometer designs, differing in mass accuracy and how they fragment molecules.
- **Polarity (ionisation mode)** — whether ions carry a positive or negative charge during measurement.
- **ppm (parts per million)** — relative error unit for mass comparisons: $10\,\text{ppm}$ of 300 Da is 0.003 Da.
- **SMILES** — a standard text encoding of a molecule's structure, for example `CCO` for ethanol.
- **InChIKey / InChIKey14** — a canonical identifier computed from a molecular structure. Its **first block (InChIKey14)** identifies the two-dimensional skeleton and ignores stereochemistry. The competition scores on InChIKey14 equality.
- **Tautomer** — an alternative arrangement of hydrogens and double bonds within the same molecule (the keto/enol pair is the classic example). Scoring treats tautomers of one molecule as identical.
- **Stereochemistry** — the three-dimensional arrangement of a molecule's atoms. Irrelevant to scoring here; only the flat skeleton matters.
- **Natural product** — a molecule produced by a living organism (plants, microbes, animals), as opposed to a synthetic compound.
- **Reference (spectral) library** — a curated collection of spectra of known molecules, used for comparison. In this competition, `train.parquet` plays that role.
- **In-silico fragmentation** — software that predicts which fragments a candidate structure would produce, and checks them against the observed peaks.
- **Neutral loss** — the mass difference between the precursor and a fragment. Comparing these differences cancels out adduct shifts.
- **Candidate (structure)** — one molecule that might be the unknown; a **candidate pool** is the set of all candidates being ranked.
- **Decoy** — a deliberately wrong but plausible candidate, used in evaluation to test how well a scoring method separates the true structure from convincing alternatives.
- **MRR@25** — mean reciprocal rank over the top 25 predictions; see [[concepts/competition/Task and Evaluation]].
- **Leakage** — an evaluation flaw where information about the correct answer reaches the model through the training data, inflating measured scores.
- **Novelty class (1 / 2 / 3)** — the hidden difficulty category of each test molecule; see [[concepts/competition/Novelty Classes]].
- **Kaggle notebook (formerly "kernel")** — the notebook environment in which submissions must run; this repo's scripts refer to it as "the kernel".
