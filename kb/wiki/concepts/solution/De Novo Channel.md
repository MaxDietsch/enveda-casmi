---
title: De Novo Channel
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/src-enveda-casmi26.md, raw/refs/scripts-overview.md, raw/articles/kaggle-data-description.md]
tags: [denovo, transformer, fingerprints, class-3]
---

# De Novo Channel

"De novo" means **from scratch**: generating a molecular structure directly from the spectrum, without any candidate list. Class-3 molecules are **not listed in PubChem at all** ([[concepts/competition/Novelty Classes]]), so no candidate pool contains them and generation is the only way to score. Even partial success matters: at an assumed ~20% class-3 share, an MRR of 0.1 on this class contributes about 0.02 to the overall score.

**What the repo has:**

- **`scripts/train_denovo.py`** — the de novo **spectrum → SMILES transformer** (a sequence model that outputs the structure as text), based on the organisers' tutorial architecture, scaled to a 6 GB GPU. Split-aware: trains only on `train_cut` spectra, held-out structures never seen, and — like the organisers' tutorial — excludes rows from `ingest_lib == 'enveda-180'`.
- **`scripts/eval_denovo.py`** — honest evaluation on held-out `test_cut` molecules: fraction of generated SMILES that are chemically **valid**, whether the exact InChIKey14 appears at rank 1 or anywhere among the generated candidates, and a **complementarity analysis** measuring where generation succeeds on molecules that retrieval misses.
- **`src/enveda_casmi26/fp_model.py` + `scripts/train_fp_head.py`** — a **fingerprint head** on the frozen transformer encoder: spectrum → encoder → mean pooling → small MLP → a 2048-bit Morgan fingerprint (a standard binary structural descriptor). Instead of (or in addition to) generating SMILES, the model predicts this fingerprint, enabling **nearest-neighbour search in fingerprint space** over any candidate pool — a middle path between generation and pool ranking. Uses the same 500-structure validation holdout as the transformer; `test_cut` molecules are never touched. `fp_model.py` is the **single source of truth mirrored into the Kaggle dataset** as `code_fp.py` ([[concepts/solution/Submission Architecture]]).
- **`train_fp_channel.py` / `analyze_fusion.py`** — integrate fingerprint predictions into the combined channel scoring and measure the gain.

**Design options** (from the official data description and the roadmap):

- **Formula-first:** predict the molecular formula, then restrict search/generation to structures with that formula — the approach of [[entities/SIRIUS|SIRIUS]] and [[entities/MIST-CF|MIST-CF]]; a common way to shrink the search space before choosing structures.
- **Embedding nearest-neighbour:** pretrained spectrum embeddings ([[entities/DreaMS]]) place the query in a learned space where structurally similar candidates can be found without reference spectra — the planned stage 3 ([[concepts/solution/Roadmap]]).
- **Validity matters:** generated SMILES must parse with RDKit and reduce to a valid InChIKey14; invalid strings waste positions in the ranked list.

Open question (CLAUDE.md): which approach — direct SMILES generation, formula-conditioned retrieval, or embedding nearest-neighbour — gives the best class-3 MRR per unit of development effort.
