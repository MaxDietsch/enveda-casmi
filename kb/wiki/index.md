# Index — Enveda CASMI 2026 — Molecule ID from Mass Spectra

> Knowledge base for the Kaggle competition `enveda-CASMI26-molecule-id-mass-spectra` and the local solution repo: the task and its rules, the data, the evaluation methodology, and the solution architecture being built.

## 🔖 Navigation
- [[#Concepts]] · [[#Entities]] · [[#Summaries]] · [[#Open Questions]]

## Concepts
### Competition
- [[concepts/competition/index|Competition]] — the challenge, metric and rules at a glance
    - [[concepts/competition/Task and Evaluation]] — MRR@25, tautomer-canonical InChIKey14 matching, submission format, quota
    - [[concepts/competition/Test Set]] — ~1,500 spectra / ~400 molecules, all timsTOF, ten adducts, minimal cleaning
    - [[concepts/competition/Novelty Classes]] — the three hidden classes and what each demands

### Training Data
- [[concepts/data/index|Training Data]] — 2.54M spectra, 275k structures, measured profile
    - [[concepts/data/Train Schema]] — the 18 columns, CE semantics, ppm error, row-id convention
    - [[concepts/data/Source Libraries]] — the 11 ingest libraries, quality and overlap
    - [[concepts/data/Spectrum Curation]] — common cleaning steps and the repo's 7-step cleaner

### Solution Architecture
- [[concepts/solution/index|Solution Architecture]] — multi-channel retrieve-then-rank pipeline
    - [[concepts/solution/Evaluation Strategy]] — simulator, leak-free splits, fair-evaluation rules
    - [[concepts/solution/Retrieval]] — stage 1: mass window, entropy similarity, measured baseline
    - [[concepts/solution/Re-ranking]] — stage 2: features, contrast z-scoring, leakage rules
    - [[concepts/solution/De Novo Channel]] — transformer, fingerprint head, formula-first alternatives
    - [[concepts/solution/Candidate Pools]] — train + COCONUT/PubChem, class reachability
    - [[concepts/solution/Submission Architecture]] — offline kernel, scorer export, channel fusion
    - [[concepts/solution/Roadmap]] — stage 3 embeddings, simulator fixes, open ideas

## Entities
- [[entities/CASMI]] — Critical Assessment of Small Molecule Identification, revived for 2026
- [[entities/Enveda Therapeutics]] — competition host; enveda-180 + NP-examples data
- [[entities/Bruker timsTOF]] — the test-set instrument platform
- [[entities/InChIKey]] — the matching key (tautomer-canonical first block)
- [[entities/RDKit]] — cheminformatics toolkit, pinned at 2026.03.3 by the metric
- [[entities/COCONUT]] — natural-products database; class-2 candidate pool
- [[entities/PubChem]] — molecular database; the class-2/3 boundary
- [[entities/SIRIUS]] — formula annotation toolbox
- [[entities/MIST-CF]] — learned formula prediction
- [[entities/DreaMS]] — pretrained spectrum embeddings + GeMS corpus
- [[entities/MS2DeepScore]] — Siamese spectral-embedding architecture (stage 3 model)
- [[entities/matchms]] — standard spectra-processing library

## Summaries (chronological)
- 2026-10-28 — [[summaries/kaggle-competition-overview]] — official overview and evaluation pages
- 2026-10-28 — [[summaries/kaggle-data-description]] — official data and test-set description
- 2026-10-28 — [[summaries/kaggle-rules-timeline]] — rules, code requirements, timeline, prizes
- 2026-10-28 — [[summaries/dataset-creation-readme]] — the local eval-set simulator
- 2026-10-28 — [[summaries/max-class1-readme]] — class-1 retrieval and re-ranker experiment
- 2026-10-28 — [[summaries/repo-solution-code]] — src/ pipeline, scripts and their roles

## Open Questions
- Q1: How much class-1 headroom is adduct-shifted matching vs CE difference, and does neutral-loss re-ranking close the `adduct` gap (0.393 → 0.5+)?
- Q2: Best class-3 strategy per engineering hour: de novo SMILES transformer, formula-conditioned retrieval (SIRIUS/MIST-CF style), or embedding kNN (DreaMS)?
- Q3: How to calibrate library-hit confidence (difficulty-1 vs difficulty-2 splits) for the class-agnostic final ordering?
- Q4: How to close the chemistry gap (synthetic timsTOF train vs natural-product-like test) — per-library reporting, NP-weighted eval, Q-TOF NP spectra?
- Q5: Does the hidden class mix match the assumed 20/60/20, and is the strategy robust to a different mix?
