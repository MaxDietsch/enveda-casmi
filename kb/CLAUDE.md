# Enveda CASMI 2026 — Molecule ID from Mass Spectra Knowledge Base

> Schema document — read at the start of every session together with `wiki/index.md`.
> Update after every major compile, ingest batch, or structural change.

## Scope

What this wiki covers:
- The Kaggle competition **Enveda CASMI 2026** (`enveda-CASMI26-molecule-id-mass-spectra`): task, metric, rules, test-set specification, data description.
- The training data: 2.5M MS/MS spectra over 11 source libraries, their schema, quality signals and curation.
- The local solution repo (`enveda-casmi/`): evaluation methodology (leak-free local splits), retrieval, re-ranking, de novo and fragmentation channels, submission architecture.
- Domain concepts: spectral similarity, entropy similarity, neutral losses, in-silico fragmentation, candidate pools (PubChem/COCONUT), spectrum embeddings.

What this wiki deliberately excludes:
- Generic ML engineering not tied to this competition (MLflow admin, uv tricks, …).
- The raw source data files themselves (multi-GB parquet; see `raw/refs/` pointers).

## Operations

This wiki follows the llm-wiki skill's five operations: `compile`, `ingest`, `query`, `lint`, `audit`.
Every operation appends an entry to `log/YYYYMMDD.md`.

## Naming conventions

- **Concept pages** (`wiki/concepts/`): Title Case noun phrases; folder-split topics (`competition/`, `data/`, `solution/`) with `index.md` + one file per aspect.
- **Entity pages** (`wiki/entities/`): Proper names (tools, databases, organizations).
- **Summary pages** (`wiki/summaries/`): kebab-case source slug.

All pages carry YAML frontmatter: `title`, `type`, `created`, `updated`, `sources`, `tags`.

### Diagrams and formulas
- All diagrams are **mermaid**. No ASCII art.
- All formulas are **KaTeX** (inline `$...$` or block `$$...$$`).

### Raw file policy
- Kaggle competition pages and web articles → full copies in `raw/articles/`.
- Repo-internal documents (plans, READMEs, analysis) live in the same repository as this wiki, so they get **pointer files** in `raw/refs/` with `kind: ref` + `repo_path` — one source of truth, no duplication.
- Large binaries → pointer files only, never copied.

## Current articles

### Concepts
- [[concepts/competition/index|Competition]] — the challenge, metric and rules
    - [[concepts/competition/Task and Evaluation]]
    - [[concepts/competition/Test Set]]
    - [[concepts/competition/Novelty Classes]]
- [[concepts/data/index|Training Data]] — schema, libraries, curation
    - [[concepts/data/Train Schema]]
    - [[concepts/data/Source Libraries]]
    - [[concepts/data/Spectrum Curation]]
- [[concepts/solution/index|Solution Architecture]] — how the repo attacks the problem
    - [[concepts/solution/Evaluation Strategy]]
    - [[concepts/solution/Retrieval]]
    - [[concepts/solution/Re-ranking]]
    - [[concepts/solution/De Novo Channel]]
    - [[concepts/solution/Candidate Pools]]
    - [[concepts/solution/Submission Architecture]]
    - [[concepts/solution/Roadmap]]

### Entities
- [[entities/CASMI]] — Critical Assessment of Small Molecule Identification
- [[entities/Enveda Therapeutics]] — competition host
- [[entities/Bruker timsTOF]] — test-set instrument platform
- [[entities/InChIKey]] — the matching key (first block, tautomer-canonical)
- [[entities/RDKit]] — cheminformatics toolkit pinned at 2026.03.3
- [[entities/COCONUT]] — natural-products database (class-2 reachability)
- [[entities/PubChem]] — molecular database (class-2/3 boundary)
- [[entities/SIRIUS]] — formula/structure annotation toolbox
- [[entities/MIST-CF]] — learned formula prediction from MS/MS
- [[entities/DreaMS]] — pretrained spectrum embeddings + GeMS corpus

### Summaries (chronological)
- 2026-10-28 — [[summaries/kaggle-competition-overview]] — official overview/evaluation pages
- 2026-10-28 — [[summaries/kaggle-data-description]] — official data + test-set description
- 2026-10-28 — [[summaries/kaggle-rules-timeline]] — rules, code requirements, timeline, prizes
- 2026-10-28 — [[summaries/dataset-creation-readme]] — the local eval-set simulator
- 2026-10-28 — [[summaries/max-class1-readme]] — class-1 retrieval + re-ranker experiment
- 2026-10-28 — [[summaries/repo-solution-code]] — src/ pipeline, scripts and their roles

## Open research questions

- How much of the class-1 (library-match) headroom is adduct-shifted peak matching vs. CE differences? (stage-1 shows `adduct` change is the bottleneck: MRR 0.393 vs 0.933 for `ce`)
- Best de novo strategy for class 3: transformer SMILES generation vs. formula-conditioned retrieval (SIRIUS/MIST-CF style) vs. embedding nearest-neighbour (DreaMS)?
- How to calibrate class-1 library-hit confidence so confident hits rank above database/de novo candidates in the merged, class-agnostic submission?
- How to close the chemistry gap: train timsTOF data is synthetic drug-like (`enveda-180`), test is natural-product-like?

## Research gaps

Sources to ingest:
- [ ] Li et al., *Nature Methods* 2021 — entropy similarity paper (the stage-1 scoring foundation)
- [ ] MS2DeepScore / MS2Deepscore paper — learned spectral embeddings (stage 3)
- [ ] DreaMS paper + GeMS corpus details — pretraining shortcut
- [ ] CASMI 2026 official challenge pages (casmi.org) — category definitions as published for the original contest
- [ ] Winning writeups once the competition closes

## Audit backlog

*(none — run `python3 scripts/audit_review.py <wiki-root> --open` to refresh)*

## Notes for the LLM

- Language: en
- Tone: neutral, deep technical; concise pages, heavy cross-linking
- Depth: deep technical (this wiki drives an active competition solution)
- Handling contradictions: state both, cite each, add to Open Research Questions.
- Competition facts come from the official Kaggle pages in `raw/articles/`; where the repo's own docs add measured numbers (`analysis/stats.json`, simulators), cite the pointer in `raw/refs/`.
