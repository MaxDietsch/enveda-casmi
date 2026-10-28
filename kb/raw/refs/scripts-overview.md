---
kind: ref
repo_path: scripts/
size: 36 scripts
---
Script inventory (roles from module docstrings): data ops — `download_data.py` (kagglehub), `clean_data.py` (cleaned parquet copies), `make_split.py` (leak-free train_cut/test_cut/pool_extra validation split holding out whole structures); baselines — `similarity_baseline.py` (library search validate/test/decoys), `make_submission.py` (cosine + mass-slot filler); retrieval channels — `build_coconut_pool.py` (COCONUT canonicalization for kernel + dataset upload), `build_pool_fps.py`, `collect_matches.py`, `mine_ce_priors.py` / `mine_losses.py` / `mine_unexplained.py` (fragmenter tuning), `benchmark_decoys.py`, `rescore_tautomer.py` (strict vs formula vs tautomer scoring); learned — `train_denovo.py` (spectrum→SMILES transformer), `eval_denovo.py`, `train_fp_head.py` (Morgan head on frozen encoder), `train_scorer.py` (candidate-level contrastive scorer), `train_fp_channel.py`, `analyze_fusion.py`, `export_scorer_json.py` (kernel-ready scorer), `evaluate_kernel_output.py`, `smoke_test_kernel.py`, `validate_kernel.py`, `simulate_holdout.py`, `explain_spectrum.py`.
