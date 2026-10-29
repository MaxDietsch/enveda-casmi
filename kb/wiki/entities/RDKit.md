---
title: RDKit
type: entity
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-competition-overview.md, raw/refs/src-enveda-casmi26.md, raw/refs/dataset-creation-readme.md]
tags: [entity, toolkit, cheminformatics]
---

# RDKit

The open-source cheminformatics library underlying the entire solution: reading and validating SMILES strings, structure standardisation, InChIKey computation, tautomer standardisation (`rdMolStandardize.TautomerEnumerator`), exact molecular masses (`ExactMolWt`), Morgan fingerprints (a standard 2048-bit structural descriptor used here for all similarity and diversity computations), scaffold extraction, and the bond/ring operations of the in-silico fragmenter. The competition metric pins **RDKit 2026.03.3** for tautomer standardisation; the repo develops against RDKit 2026.3.6. Standardising all ~277k training SMILES takes ~6 minutes and is cached; keys are computed once per structure and reused everywhere ([[entities/InChIKey]]).
