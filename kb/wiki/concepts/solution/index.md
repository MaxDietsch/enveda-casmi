---
title: Solution Architecture
type: concept-index
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/src-enveda-casmi26.md, raw/refs/scripts-overview.md, raw/refs/max-class1-readme.md]
tags: [solution, architecture]
---

# Solution Architecture

The repo solves the problem with a **two-stage pipeline — first retrieve candidates, then rank them** — guided by a strict local evaluation methodology. The three hidden novelty classes ([[concepts/competition/Novelty Classes]]) decompose the work:

```mermaid
flowchart LR
    S["test molecule<br/>(1-16 MS/MS spectra)"] --> Q["neutral query mass<br/>(median over spectra)"]
    Q --> C["candidate pool<br/>train structures +<br/>COCONUT / PubChem"]
    C --> CH1["similarity channel:<br/>compare against reference spectra"]
    C --> CH2["fragmentation channel:<br/>would this structure<br/>produce these peaks?"]
    C --> CH3["mass channel:<br/>how close is the mass?"]
    C --> CH4["learned channel:<br/>trained scorer P(candidate correct)"]
    CH1 & CH2 & CH3 & CH4 --> Z["standardise per channel within<br/>the molecule's candidates,<br/>weighted sum per configuration"]
    Z --> DN["de novo channel<br/>(structure generation / fingerprint prediction)"]
    Z & DN --> R["25 ranked SMILES<br/>(calibrated blend)"]
```

Two parallel development branches live in the repo:

- **`src/enveda_casmi26/` + `scripts/`** — the full pipeline for all classes: candidate retrieval by mass window, in-silico fragmentation scoring, a trained candidate scorer, structure generation plus fingerprint prediction, the COCONUT database extract, and the Kaggle-notebook export path ([[summaries/repo-solution-code]]).
- **`max_class1_exp/`** — a focused, carefully validated experiment on class-1 retrieval and re-ranking ([[summaries/max-class1-readme]]).

Sub-pages:

- [[concepts/solution/Evaluation Strategy]] — leak-free local splits, the simulator, honest measurement
- [[concepts/solution/Retrieval]] — stage 1: mass window, entropy similarity, neutral losses
- [[concepts/solution/Re-ranking]] — stage 2: learned re-ranking of the retrieved shortlists
- [[concepts/solution/De Novo Channel]] — spectrum→SMILES generation and fingerprint prediction
- [[concepts/solution/Candidate Pools]] — train structures + COCONUT/PubChem, and which classes each makes reachable
- [[concepts/solution/Submission Architecture]] — offline Kaggle-notebook constraints, scorer export, channel fusion
- [[concepts/solution/Roadmap]] — planned embedding stage and open ideas

Vocabulary: [[concepts/Glossary]].

Everything is validated locally before any Kaggle submission ([[concepts/solution/Evaluation Strategy]]); experiment tracking via MLflow.
