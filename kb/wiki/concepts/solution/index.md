---
title: Solution Architecture
type: concept-index
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/src-enveda-casmi26.md, raw/refs/scripts-overview.md, raw/refs/max-class1-readme.md]
tags: [solution, architecture]
---

# Solution Architecture

The repo attacks the problem with a **multi-channel retrieve-then-rank pipeline**, informed by a strict local evaluation methodology. The three hidden novelty classes ([[concepts/competition/Novelty Classes]]) decompose the problem:

```mermaid
flowchart LR
    S["test molecule<br/>(1-16 MS2 spectra)"] --> Q["neutral query mass<br/>(median over spectra)"]
    Q --> C["candidate pool<br/>train structures +<br/>COCONUT / PubChem"]
    C --> CH1["channel: sim<br/>spectral similarity"]
    C --> CH2["channel: eif<br/>in-silico fragmentation"]
    C --> CH3["channel: mass<br/>ppm proximity"]
    C --> CH4["channel: learned<br/>CandidateScorer P(true)"]
    CH1 & CH2 & CH3 & CH4 --> Z["z-score within candidate set,<br/>weighted sum per config"]
    Z --> DN["de novo channel<br/>(transformer / fp head)"]
    Z & DN --> R["25 ranked SMILES<br/>(calibrated blend)"]
```

Two parallel development branches in the repo:

- **`src/enveda_casmi26/` + `scripts/`** — the full pipeline for all classes: candidate retrieval by neutral-mass window, in-silico fragmentation explained-intensity channel, contrastive candidate scorer, de novo transformer + fingerprint head, COCONUT pool, kernel export path ([[summaries/repo-solution-code]]).
- **`max_class1_exp/`** — a focused, deeply-validated experiment on class-1 retrieval and re-ranking ([[summaries/max-class1-readme]]).

Sub-pages:

- [[concepts/solution/Evaluation Strategy]] — leak-free local splits, the simulator, honest measurement
- [[concepts/solution/Retrieval]] — stage 1: mass window, entropy similarity, neutral losses
- [[concepts/solution/Re-ranking]] — stage 2: learned shortlist re-ranker, features, leakage rules
- [[concepts/solution/De Novo Channel]] — spectrum→SMILES transformer and fingerprint head
- [[concepts/solution/Candidate Pools]] — train + COCONUT/PubChem pools and class implications
- [[concepts/solution/Submission Architecture]] — offline kernel constraints, scorer export, channel fusion
- [[concepts/solution/Roadmap]] — stage 3 embeddings and open ideas

Everything is validated locally before any Kaggle submission ([[concepts/solution/Evaluation Strategy]]); experiment tracking via MLflow.
