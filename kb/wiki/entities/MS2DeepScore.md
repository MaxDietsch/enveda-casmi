---
title: MS2DeepScore
type: entity
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/max-class1-readme.md]
tags: [entity, model, embeddings, similarity]
---

# MS2DeepScore

A Siamese neural network that learns **structural** similarity from spectrum pairs — the architectural style planned for stage 3 of the class-1 experiment: train on positive pairs of the *same molecule measured in different settings* (instrument/adduct/CE), so the embedding is robust to exactly the changes class 1 presents. Train supplies ~2,500 bridge molecules, ~69,000 multi-adduct and ~178,000 multi-CE molecules as pair sources. The embedding similarity would become a stage-2 feature; compare with pretrained [[entities/DreaMS|DreaMS]] ([[concepts/solution/Roadmap]]).
