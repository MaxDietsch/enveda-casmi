---
kind: ref
repo_path: src/enveda_casmi26/
size: ~2.4 KB across 8 modules
---
The shared solution library, mirrored into the Kaggle kernel/dataset: `cleaning.py` (7-step curation: precursor cut, intensity floor, deisotoping, mass-defect junk filter, windowed top-K, sqrt/log transform, renormalize), `fragmentation.py` (in-silico MS2: side-chain cleavages, ring fissions, H rearrangements, neutral-loss cascades, adduct-aware charge states), `pipeline.py` (candidate retrieval by neutral-mass window + multi-channel ranking: mass / sim / eif / learned, z-scored within molecule, MRR@25 on inchikey14), `scoring.py` (Tier-4 candidate-level contrastive likelihood scorer, logistic), `fp_model.py` (de novo transformer encoder + Morgan fingerprint head, single source of truth shared with the kernel), `tracking.py` (MLflow helpers).
