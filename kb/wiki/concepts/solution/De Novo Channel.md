---
title: De Novo Channel
type: concept
created: 2026-10-28
updated: 2026-10-28
sources: [raw/refs/src-enveda-casmi26.md, raw/refs/scripts-overview.md, raw/articles/kaggle-data-description.md]
tags: [denovo, transformer, fingerprints, class-3]
---

# De Novo Channel

Class 3 molecules are **not in PubChem at all** ([[concepts/competition/Novelty Classes]]) — no candidate pool contains them, so the only way to score is to **generate** structures from the spectrum. Even partial success pays: at ~20% class-3 share, MRR 0.1 on this class ≈ +0.02 overall.

**What the repo has:**

- **`scripts/train_denovo.py`** — the de novo **spectrum → SMILES transformer** (tutorial architecture scaled to a 6 GB GPU). Honest-split aware: trains only on `train_cut` spectra, held-out structures never seen, and **excludes `ingest_lib == 'enveda-180'`** rows like the organizers' tutorial.
- **`scripts/eval_denovo.py`** — honest evaluation on held-out `test_cut` molecules: SMILES **validity**, exact **inchikey14 top-1** / any-hit among generated candidates, and a **rescue analysis** measuring how de novo generation would complement the v8 retrieval channel (i.e. where it finds molecules retrieval misses).
- **`src/enveda_casmi26/fp_model.py`** + **`scripts/train_fp_head.py`** — a **fingerprint head** on the frozen transformer encoder: spectrum → encoder (frozen, `models/denovo_v1.pt`) → masked-mean pooling → MLP → 2048-bit Morgan radius-2 fingerprint (BCE loss). The predicted fingerprint enables **nearest-neighbour retrieval in fingerprint space** over any candidate pool — a bridge between pure generation and pool ranking. Same 500-structure validation holdout as the transformer; `test_cut` never touched. `fp_model.py` is the **single source of truth mirrored into the Kaggle dataset** as `code_fp.py` ([[concepts/solution/Submission Architecture]]).
- **`train_fp_channel.py` / `analyze_fusion.py`** — wire the fingerprint predictions into the channel fusion and measure the gain.

**Design space (from the official data description and roadmap):**

- **Formula-first:** predict the molecular formula, then condition generation/retrieval on it — the SIRIUS / MIST-CF approach ([[entities/SIRIUS]], [[entities/MIST-CF]]); a common way to narrow the search space before structures are chosen.
- **Embedding nearest-neighbour:** pretrained spectrum embeddings ([[entities/DreaMS]]) map the query into a learned space where structural neighbours can be retrieved without reference spectra — the planned stage 3 ([[concepts/solution/Roadmap]]).
- **Validity matters:** generated SMILES must be RDKit-parseable and reduce to a valid InChIKey14; invalid strings waste rank slots.

Open question (see CLAUDE.md): which de novo strategy — direct SMILES generation, formula-conditioned retrieval, or embedding kNN — gives the best class-3 MRR per engineering hour.
