---
title: RDKit
type: entity
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-competition-overview.md, raw/refs/src-enveda-casmi26.md, raw/refs/dataset-creation-readme.md]
tags: [entity, toolkit, cheminformatics]
---

# RDKit

The open-source cheminformatics toolkit underlying everything: SMILES parsing/validation, standardisation, InChIKey computation, tautomer canonicalization (`rdMolStandardize.TautomerEnumerator`), exact masses (`ExactMolWt`), Morgan fingerprints (radius 2, 2048 bits — the repo's similarity/diversity currency), Bemis–Murcko scaffolds, and the in-silico fragmenter's bond/ring manipulation. The competition metric pins **RDKit 2026.03.3** for tautomer canonicalization; the repo develops against RDKit 2026.3.6. Validation loop in the simulator (canonicalizing 277k SMILES takes ~6 min, cached); keys computed once per structure and reused everywhere ([[entities/InChIKey]]).
