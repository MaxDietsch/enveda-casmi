---
title: Bruker timsTOF
type: entity
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md, raw/refs/dataset-creation-improvements.md]
tags: [entity, instrument]
---

# Bruker timsTOF

A commercial mass-spectrometer platform (a Q-TOF design) used for **100% of the hidden test spectra** (`instrument_type == "timsTOF"`). The training data's timsTOF spectra come almost entirely from `enveda-180` and `enveda-np-examples`. Because it is a Q-TOF, its fragmentation resembles that of other Q-TOF/TOF instruments — which is why a proposed simulator improvement accepts Q-TOF spectra from `riken`, `gnps`, `mona`, `msdial` and `massbank` as evaluation spectra for a natural-product variant. Instrument-matched training data is the repo's main advantage on class 1 ([[concepts/solution/Retrieval]]).
