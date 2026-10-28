---
title: PubChem
type: entity
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md]
tags: [entity, database]
---

# PubChem

The comprehensive open molecular database. The class boundary runs through it: class 2 = "known structure, no public spectra" — the structure **is in PubChem** (or COCONUT); class 3 = "novel structure" — **not in PubChem at all**, reachable only by de novo generation. No PubChem pool build exists in the repo yet (open question: snapshot date and coverage matter for the class-2/3 split). Level-3 evaluation keys must be removed from every pool, external ones included ([[concepts/solution/Candidate Pools]]).
