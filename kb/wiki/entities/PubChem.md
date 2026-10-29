---
title: PubChem
type: entity
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md]
tags: [entity, database]
---

# PubChem

The comprehensive open database of chemical structures. The difficulty-class boundary runs through it: class 2 = "known structure, never measured publicly" — the structure **is listed in PubChem** (or COCONUT); class 3 = "unknown structure" — **not listed in PubChem at all**, reachable only by generating a structure from the spectrum. No PubChem extract exists in the repo yet (open question: which snapshot date and coverage define the class-2/3 boundary). Level-3 evaluation keys must be removed from every candidate pool, external ones included ([[concepts/solution/Candidate Pools]]).
