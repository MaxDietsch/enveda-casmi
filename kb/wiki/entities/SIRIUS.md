---
title: SIRIUS
type: entity
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md]
tags: [entity, tool, formula-annotation]
---

# SIRIUS

An established academic toolbox for identifying molecules from tandem-mass spectra: it predicts the **molecular formula** from the spectrum (and, via its CSI:FingerID component, probable substructures). The official data description names it — together with [[entities/MIST-CF|MIST-CF]] — as the reference for the **formula-first strategy**: predict the molecular formula first, then restrict structure search or generation to candidates with that formula, which sharply reduces the search space. Relevant to class 2 (database retrieval) and class 3 (generation) as discussed in [[concepts/solution/De Novo Channel]].
