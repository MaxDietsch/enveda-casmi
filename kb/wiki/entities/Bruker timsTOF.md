---
title: Bruker timsTOF
type: entity
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md, raw/refs/dataset-creation-improvements.md]
tags: [entity, instrument]
---

# Bruker timsTOF

The Q-TOF mass spectrometer platform used for **100% of the hidden test spectra** (`instrument_type == "timsTOF"`). Train's timsTOF data lives almost entirely in `enveda-180` plus `enveda-np-examples`. As a Q-TOF, its fragmentation is close to other Q-TOF/TOF spectra, which is why the simulator's proposed improvement accepts Q-TOF spectra from `riken`/`gnps`/`mona`/`msdial`/`massbank` as eval spectra for a natural-product variant. Instrument-matched training data is the repo's main advantage on class 1 ([[concepts/solution/Retrieval]]).
