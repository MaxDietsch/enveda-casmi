---
title: DreaMS
type: entity
created: 2026-10-28
updated: 2026-10-28
sources: [raw/articles/kaggle-data-description.md, raw/refs/max-class1-readme.md]
tags: [entity, model, embeddings, pretraining]
---

# DreaMS

A project that released **pretrained spectrum-embedding models** (neural networks mapping a spectrum to a learned vector space where similar spectra lie close together) together with **GeMS**, a large curated corpus of spectra without known identities, harvested from the public MassIVE repository — enabling self-supervised pretraining at repository scale. The official description lists it as a pretrained alternative to training embeddings from scratch; the repo's stage-3 plan uses DreaMS embeddings as the comparison baseline for its own cross-setting embedding model ([[concepts/solution/Roadmap]]). Pretrained and public, so usable in the offline scoring notebook as an attached dataset ([[concepts/solution/Submission Architecture]]).
