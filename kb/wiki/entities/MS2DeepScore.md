---
title: MS2DeepScore
type: entity
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/max-class1-readme.md]
tags: [entity, model, embeddings, similarity]
---

# MS2DeepScore

A two-tower ("Siamese") neural network that learns **structural** similarity from pairs of tandem-mass spectra. It is the architectural template for the class-1 experiment's planned stage 3: train on positive pairs of the *same molecule measured under different conditions* (instrument, adduct, collision energy), so the learned embedding is robust to exactly the changes class 1 presents. The training data supplies ~2,500 molecules measured on timsTOF *and* other instruments, ~69,000 with several adducts and ~178,000 with several collision energies as pair sources. The embedding similarity would become a re-ranking feature, compared against pretrained [[entities/DreaMS|DreaMS]] embeddings ([[concepts/solution/Roadmap]]).
