# Class-2 experiment (`max_class2_exp/`) — experiment plan

> **Forum evidence:** see [`FORUM_FINDINGS.md`](FORUM_FINDINGS.md) for the
> 2026-11-05 synthesis of all 150 competition discussion threads and the
> concrete P0/P1/P2 changes it implies for this experiment.
>
> **What is trained, in plain words:** see
> [`TRAINING_EXPLAINED.md`](TRAINING_EXPLAINED.md) — every model and
> channel, its inputs, which molecules it sees, and a vocabulary table.
>
> **Code-level walkthrough:** see [`PIPELINE.md`](PIPELINE.md) for what
> happens to one spectrum at inference and how the training script works,
> step by step.

**Class 2 is assumed to be ~60 % of the hidden test.** The correct structure
is known to science (in PubChem / COCONUT) but has **never been measured** —
no reference spectrum of the answer exists anywhere in the data. The local
analogue is `dataset-creation/out/difficulty2.parquet`: every train row of
each eval molecule is hidden, its structure only enters through
`split.library_additions()` (simulator §7.3). The task: given the query
spectra and a candidate pool of **structures**, rank them by MRR@25.

This README is the plan for the experiment. It describes the problem, what
we deliberately do **not** do, the proposed algorithm, the training/leakage
protocol, the evaluation protocol, and the implementation status.

---

## 1. The problem, precisely

For each eval molecule we get 1–16 timsTOF spectra (ten adducts, CEs given).
The candidate pool is built from:

- every molecule with ≥ 1 *visible* train row (the measured structures),
- **plus the library additions** (the class-2 answers, ~1,000 structures),
- optionally an external pool (`--external-pool`, e.g. the COCONUT extract
  from `scripts/build_coconut_pool.py`),
- **minus the level-3 exclusions** (`split.excluded_library_keys`).

Retrieval is trivial: the pool is sorted by exact mass, candidates = the
± ppm window around the query's median implied neutral mass (~50–300
candidates at ±10 ppm). **All difficulty is in ranking.** The answer has no
spectra; most decoys *do* (they are visible train molecules). The question
the algorithm must answer is:

> *How plausibly would this candidate **structure** produce the observed peaks?*

## 2. What we deliberately do NOT do

- **No reference-spectrum search.** The class-1 pipeline (`max_class1_exp/
  stage1.py` / `stage2.py`) ranks candidates by similarity against their own
  library spectra. For class 2 that is not just weak, it is *backwards*: the
  truth has no library spectra, so any decoy with measured spectra outranks
  it. The class-1 experiment is a different problem; **its code is not used
  here** — not its stage logic, and not its modules.
- **No class-1 imports.** `max_class2_exp/` is self-contained: its own
  spectrum cleaning and similarity maths (`spectra_utils.py`, small numpy
  functions), its own data handling. It shares only the repo-wide packages
  (`src/enveda_casmi26`, `dataset-creation`) that every experiment uses.
- **No eval-molecule spectra in training, ever.** Neither as examples nor as
  lookups (asserted at runtime; §5).

## 3. Proposed algorithm: rank structures, not spectra

Evidence channels score each (query, candidate) pair — all answering the
question above from opposite directions — fused by a learned re-ranker:

```
channel P (built):    spectrum --[learned model]--> predicted fingerprint
                      vs. the candidate's actual fingerprint
channel F (built):    structure --[in-silico fragmentation]--> theoretical peaks
                      vs. the observed peaks
channel F' (built):   structure --[MetFrag-lite cuts]--> simple theoretical peaks
                      with cut weights; linear / rarity-weighted explained
                      intensity, gap-to-best (FORUM_FINDINGS.md P1.1)
channel F2 (built):   structure --[ICEBERG, zero-shot]--> predicted spectrum
                      vs. the observed spectrum (matched cosine)
channel A (flagged):  structure --[nearest visible relatives]--> proxy spectra
                      vs. the observed peaks (OFF by default, --enable-analogues)
channel M (built):    mass / formula window (retrieval), signed ppm error,
                      worst-over-spectra mass error, adduct agreement, source
                      flags, same-formula competition
                      |
                      v
   learned re-ranker (logistic / HGB over channel scores + context)
   trained on on-the-fly pseudo-class-2 examples
```

### Channel P — learned spectrum→fingerprint prediction (the centerpiece)

The CSI:FingerID idea, adapted to this dataset: learn to predict a molecular
fingerprint *from the spectrum*, then rank candidates by agreement between
the predicted fingerprint and each candidate's actual fingerprint. This is a
genuinely different algorithm from library search: it never compares the
query with reference spectra; it learns the *association* between peaks and
substructures from the visible train data and applies it to structures that
were never measured.

- **Model (v1, default, always on):** a multi-label *linear* model —
  500 intensity-weighted linear m/z bins + adduct one-hot + scaled
  precursor m/z → 2048 sigmoid outputs, BCE — trained **inside every
  stage2 run** on rows visible under the split mask only (and in clean
  mode minus the sampled pseudo-truths' rows): leak-free by construction,
  asserted (R1/R2 extended to channel P). CPU-only, a couple of minutes,
  deliberately never cached to disk (a stale cached model would be a
  silent leak). This is a FingerID-lite.
- **Model (v2, auto when the artifacts exist):** the repo's de novo
  transformer encoder + Morgan head (`src/enveda_casmi26/fp_model.py`) —
  no flag: presence of `out/denovo_v1.pt` + `out/denovo_fp_head.pt`
  decides. The encoder is trained by **`train_denovo.py` in this
  folder**, the head by **`train_fp_head.py`**: CUDA with bf16
  autocast + TF32 when a GPU is visible, CPU fallback otherwise; every run
  logs config, losses, generation metrics, and the checkpoint + tokenizer
  to MLflow (`mlflow.db`, experiment `enveda-casmi26`). Run from the repo
  root with `uv run python max_class2_exp/train_denovo.py`; a smoke test
  needs `--spectra 3 --epochs 1 --max-rows 400 --val-structures 10` plus
  small `--embed-dim/--n-layers/--ff-dim`. Defaults are documented in
  `--help` (§7.1).
- **Leak guard (v2, default, no stage2 flag):** both trainers exclude
  every molecule of every `dataset-creation/out/*.parquet` split (plus
  `data/test_cut.parquet` when present) *by default* and record the
  excluded keys in the checkpoint `meta`; `fpchannel.build_fp_model`
  **refuses** an artifact whose guard does not cover the split being
  evaluated (retrain it, or move it away to fall back to v1).
  `--include-eval-molecules` (trainer flag) opts out for the final Kaggle
  model only — stage2 will then refuse that artifact locally, by design.
- **Candidate score:** FingerID-style log-likelihood rather than plain
  Tanimoto: for each bit b with predicted probability p_b and candidate bit
  c_b ∈ {0,1}, score = Σ_b [c_b log(p_b/prior_b) + (1−c_b)
  log((1−p_b)/(1−prior_b))], normalised by bit priors so that "everything
  predicted 1" cannot win. A candidate's score is the mean over the
  query's spectra (predict per spectrum, average the probability vector,
  then score).
- **Training data:** visible train rows only (spectrum → fingerprint of its
  own molecule). The eval molecules' rows are all hidden, so they are never
  seen; in clean mode the sampled pseudo-truths' rows are hidden from the
  FP model too, so the channel can never recognise a pseudo-truth from its
  own spectra (R2 holds for every learned channel, asserted).

### Channel F — in-silico fragmentation (built, `insilico.py`)

The repo's Fragmentor (`src/enveda_casmi26/fragmentation.py`) enumerates
theoretical fragment m/z values of the candidate *structure* for the query's
adduct: side-chain cuts, ring fissions, neutral-loss cascades with
functional-group gating, even-electron rule (~50 ms, ~700 fragments per
molecule, cached per process). The observed peaks are matched within
`--frag-ppm`; matched peak count, explained intensity share, and mean ppm
closeness become features. This channel needs no training data and no
reference spectra — it is pure chemistry simulation — and it is the only
channel that can discriminate *within* a structural family (which isomer is
the right one).

### Channel F′ — MetFrag-lite cuts (`metfrag.py`, built; forum P1.1)

The forum's best-evidenced class-2 feature block is deliberately *dumber*
than the Fragmentor — 1-bond and 2-bond cuts only (2-bond down-weighted
×0.6), H shifts −2..+3 covering every H-transfer convention, `--mtf-tol`
0.005–0.01 Da. Four scoring choices that the forum validated on a
241-molecule isomer panel (+0.065 MRR) and on the no-public-spectra split
(0.502 → 0.627):

- **linear intensity weighting** of explained intensity (not sqrt);
- **rarity weighting**: a peak every candidate explains is weak evidence
  (`1 / (1 + #candidates matching)`);
- **gap to the best candidate** in the same shortlist (explained intensity
  *relative* to the competition, which is what ranking needs);
- cut parsimony as the signal — no cleavage-propensity bookkeeping
  (propensity weights and reachability penalties were both *rejected* on
  the forum).

Per-candidate features: `mtf_matched_peaks`, `mtf_explained_linear`,
`mtf_rarity_explained`, `mtf_gap_best`, `mtf_mean_ppm`, `mtf_available`.
Enumerated sets are cached per (SMILES, polarity); the channel benchmarks
at ~0.1–0.3 s per structure.

### Channel M — mass and formula context

Signed mass error (ppm) of the candidate vs. the query mass; the precursor
implies the neutral mass and constrains candidate formulas already via the
ppm window. Cheap but far from useless: `mass_error` alone is the baseline.

### Fusion — learned re-ranker over pseudo-class-2 examples

A small tabular model (logistic and/or HistGradientBoosting, as in the
class-1 recipe but reimplemented here) maps the channel scores (+ context:
candidate visibility counts, shortlist size) to P(candidate is the truth).
Training data is generated **on the fly**: sample molecules that are visible
under the split, hide *all* their rows (exactly what the simulator does to
eval molecules), use their rows as pseudo-queries, and add them to the
training pool as additions. The re-ranker thus learns under the same
asymmetry as the eval set — pseudo-truths have no spectra, decoys do.

### Optional channel F2 — neural spectrum prediction (`nnspec.py`)

A *learned* upgrade of channel F: instead of matching observed peaks against
an unweighted theoretical m/z list, a pretrained neural model predicts a full
peak list **with intensities** for each candidate structure, conditioned on
adduct, collision energy and instrument; the predicted spectrum is compared
to the observed one (matched cosine, ppm-tolerant) and the similarity becomes
a re-ranker feature. This directly answers the Fragmentor's known weakness
("a peak that matches *some* theoretical fragment is weak evidence" — the
docstring TODO of "likelihood-weighted scoring").

**Wired into the fusion** (`f2_bridge.py` + `f2_batch.py`, no flag): stage2
collects the unique (smiles, adduct, 5 eV CE bucket) triples of every
shortlist, predicts the missing ones in ONE ms-pred subprocess and caches
them in `out/cache/f2_predictions.parquet` (the first pass over a fresh
pool is a one-time cost; later runs are instant). The features
`f2_cosine_best` / `f2_cosine_mean` / `f2_available` are appended to every
shortlist — training, eval and blind control through the same code path
(R4). No env / no weights → the channel degrades to all-zero features.

- **Model:** ICEBERG 2.1 (Wang et al., bioRxiv 2025.05.28.656653; code
  MIT, `extern/ms-pred`, vendored + git-ignored). Zero-shot — no training
  here, frozen weights, inference only.
- **Checkpoint:** the public **MassSpecGym** `msg_all` weights
  (`out/iceberg/{gen,inten_contr}/best.ckpt`). Trained on public
  experimental MS/MS only (GNPS/MassBank/MoNA-type sources) — no NIST (which
  would be rule-ineligible: the competition allows only freely/publicly
  available external data), no Enveda data.
- **Coverage:** all ten competition adducts are in ICEBERG's adduct
  vocabulary (`ICEBERG_ADDUCT` map); collision energy in eV (the test's
  20–60 eV range) and instrument (`QTOF` for timsTOF) are model inputs.
- **Environment:** runs in the ms-pred uv env (its own rdkit/torch pins —
  NOT the repo venv):

  ```bash
  # one-time setup
  git clone https://github.com/coleygroup/ms-pred extern/ms-pred
  (cd extern/ms-pred && uv sync --extra cpu)   # CUDA: --extra cu124
  # weights: see out/iceberg/ (downloaded from the ms-pred README's
  # MassSpecGym msg_all Dropbox link)
  ```

  ```bash
  # demo / sanity check (predict + compare to measured train spectra)
  extern/ms-pred/.venv/bin/python max_class2_exp/nnspec.py \
      --smiles "Cn1cnc2c1c(=O)n(C)c(=O)n2C" --adduct "[M+H]+" --ce 40 --compare-train

  # discrimination sanity check: truth vs ~30 mass-window decoys, timsTOF queries
  extern/ms-pred/.venv/bin/python max_class2_exp/f2_discrimination_check.py
  ```

  Note: invoke the env's python directly (`extern/ms-pred/.venv/bin/python`)
  — `uv run` without `--extra cpu` re-resolves and installs the CUDA torch
  wheels on top of the lean CPU env.

- **Leakage protocol (extend R1/R2 to F2):** the class-2 answers are
  never-measured molecules, so their spectra are in no corpus — no answer
  leakage. But *measured decoys* may occur in the public training data of
  the checkpoint (soft memorization), and pseudo-class-2 truths drawn from
  visible train molecules would enjoy the same memorization. Before trusting
  local numbers, score pseudo-queries only with a checkpoint that did not
  see their spectra, or (self-trained variant) train on a molecule-disjoint
  half of the visible pool.
- **Self-trained upgrade (later, optional):** the ms-pred codebase can
  retrain ICEBERG on our own `train.parquet` (2.5M spectra, instrument- and
  adduct-matched, `collision_energy_ev`) — the "exactly our use case"
  version. Zero-shot comes first to measure the channel's value.

### Optional channel A — analogue transfer (flagged, `--enable-analogues`)

Because it uses spectra, it is **off by default** (`k = 0`) and clearly
flagged: a candidate that was never measured often has *relatives* that were
(`enveda-180` is combinatorial families; median closest visible relative of
a difficulty-2 molecule: Tanimoto ≈ 0.58). The spectra of a candidate's
top-K nearest *visible* molecules act as a proxy for its own spectrum. To be
explicit about the legitimacy question: the *answer's* spectra are never
used (they are hidden); the comparison is query-vs-proxy — an empirical
approximation of the structure→spectrum direction, not reference-spectrum
search. Expected to be the strongest single channel where relatives exist,
useless for the ~16 % of molecules with none. Default experiment runs keep
it disabled (`--enable-analogues` turns it on with k = 8; the `--leak-train`
self-test also forces it on so the analogue path stays exercised); we enable
it only to *measure* its contribution (FORUM_FINDINGS.md P1.7).

## 4. Why this should work (and where it will not)

- **Coverage without relatives:** the FP channel and in-silico channel work
  for every candidate, including the hard ~16 % with no visible relative
  and the isomer-level discrimination within families.
- **FP vs. in-silico complementarity:** the Fragmentor is high-precision
  but low-coverage (a peak that matches *some* theoretical fragment is weak
  evidence — big molecules explain almost anything); the FP channel is the
  reverse (smooth, learned, but coarse). Their errors should decorrelate,
  which is what the fusion model exploits.
- **Realistic risk levels:** published FingerID-style retrieval typically
  gets the right structure into the top-25 for a large fraction of
  "known-structure, never-measured" cases; with only ~50–300 candidates per
  query here (not PubChem-scale), even a moderately good FP predictor should
  be competitive. Risks: Morgan r2 fingerprint collisions between
  homologues; the synthetic-vs-natural-product chemistry gap (timsTOF train
  data is almost all `enveda-180`); pool-scale shift on Kaggle (PubChem
  snapshot unknown). The `by_tc_band` breakdown will localize any failure.

## 5. Training data and leakage rules

All learned components (FP model, re-ranker) train on **visible train rows
only**, with rules asserted at runtime (mirroring the repo's conventions):

- **R1** no eval molecule is ever a training example (neither as FP-model
  input nor as a re-ranker pseudo-query) and no split-hidden row is ever
  visible; re-ranker queries are drawn only from molecules visible under
  the split;
- **R2** every re-ranker training example hides *all* of the pseudo-truth's
  rows — the truth is reachable only as a structure;
- **R2b** a shortlist whose truth fell outside its own mass window (possible
  at `--ppm 8.5`) carries no positive label — it is dropped from training
  and counted (`training.n_truth_outside_window`), never padded; on the eval
  side the same molecule lands in the `outside_window` failure bucket;
- **R3** simulator metadata (`difficulty`, `neighbours_removed`,
  `max_visible_tc`) is never a feature (stratification only);
- **R4** one shared featurize/scoring function serves training and
  evaluation;
- **R5** all randomness from one seeded `np.random.default_rng(--seed)`.

## 6. Evaluation protocol

Implemented per FORUM_FINDINGS.md §0 (P0.1–P0.6):

- **Defaults (P1.2):** candidate window `--ppm 8.5` (community sweet spot;
  99 % of timsTOF precursors sit within 5 ppm), peak `--tol 0.01` Da;
  analogue channel off. Sweep 5/8.5/10 and 0.005/0.01/0.02 before
  believing any delta.
- **Headline metrics:** MRR@25, top-1/10/25, candidate recall — overall,
  per `max_visible_tc` band (0–0.3 / 0.3–0.5 / 0.5–0.7 / 0.7–0.85 /
  0.85–1), per `neighbours_removed`, per difficulty, and **conditional on
  recall** (MRR over molecules whose truth is in-window).
- **Same-formula isomer panel (P0.2)** — the metric class 2 is actually
  about: truth + all same-formula candidates (`panel_*`), and truth + the
  top-10 same-formula decoys by the primary model (`paneltop_*`).
  Overall MRR dilutes isomer wins; this does not.
- **Failure ledger (P0.3):** every molecule lands in exactly one bucket —
  `outside_window` (recall problem), `in_window_rank_gt25` (tail),
  `rank_2_25` (ranking problem), `rank_1` (win). Per-method counts in the
  metrics JSON and results parquet.
- **Public-spectra stratification (P0.4):** `has_public_spectra` = the
  truth structure has rows in a public corpus (any `ingest_lib` outside
  `enveda-180` / `enveda-np-examples`). The **no-public-spectra slice is
  the honest headline** for any pretrained channel — a gain that exists
  only there is memorisation.
- **Spectrum-blind control (P0.5):** `--blind-control` re-scores every
  query with the spectra of a mass-adjacent molecule (same windows, same
  candidates, falsified spectral evidence). A channel whose score barely
  drops is reading something other than the spectra; a run with coverage 0
  is flagged vacuous.
- **Uncertainty (P0.6):** `--n-boot 200` paired-by-molecule bootstrap of
  stage2 vs each channel baseline (mean delta, 95 % CI, P(delta > 0));
  `--model-seeds k` fits and averages k HGB seeds; `--train-seed` sweeps
  for dataset-seed error bars.
- **Class-share scenarios (P0.1):** the metrics JSON carries the
  16/45/39 and 28/20/52 class-share estimates next to our working
  20/60/20 — expected overall LB gain = share(class 2) × ΔMRR(class 2);
  only the expected gain scales with the share, never the method.
- **Rank-1 protection (P1.5):** `--protect-top1 <channel>` keeps the named
  channel's top-1 at rank 1 and inserts the fused pick at rank 2 — it
  cannot lose a rank-1 hit of that channel (`stage2_protect_*` columns).
- **Feature ablation (P1.3) + the source-block default:** `--drop-features a,b`
  removes model features (they stay in the saved frames). The visibility/
  source block (`cand_is_addition`, `cand_visible_rows`, `cand_tims_rows`,
  `cand_public_rows`) is **dropped from the model by default**: within the
  simulator every eval truth is a spectrum-less addition, so those flags
  alone rank the truth at MRR ≈ 1.0 — the model would measure the
  simulator's construction rule, not the chemistry, and the number would
  not transfer to the Kaggle pool where most decoys also lack spectra
  (FORUM_FINDINGS.md P1.3 "context, not a truth shortcut"; P2.2: such
  flags looked great on local holdouts and hurt on the LB). Pass
  `--keep-source-features` to measure their contribution explicitly.
- **Splits:** `difficulty2.parquet` (default, 1,000 molecules); draws with
  other `--train-seed`s for error bars; `overall.parquet` to see the
  interaction with class 1/3 in the mixed test-like setting.
- **Baselines printed side by side:** `mass_error`, `entropy_best`,
  `entropy_composite`, `frag_explained`, `mtf_explained`, then the fused
  re-ranker — each channel is evaluated alone *and* in fusion, so its
  marginal value is measured, not assumed.
- **Self-tests:** unit tests (`test_stage2.py`, no data needed) plus a
  `--leak-train` wiring test (pretend the truths were measured: their eval
  spectra enter as their own analogues → MRR must be ≈ 1; never used for
  real numbers).
- **Success criteria:** fused MRR@25 clearly above every single channel
  (bootstrap CI excluding 0), and panel MRR moving even where overall MRR
  does not; the 0.7–1.0 Tc bands ≈ class-1-like quality (MRR ≳ 0.5); the
  0–0.3 band carried by the in-silico channels (anything ≳ 0.1 there is
  pure gain over the class-1-style approach, which scores ~0 there by
  construction).

## 7. Architecture (self-contained file plan)

| file | role | status |
|---|---|---|
| `pool.py` | candidate pool: visible structures + library additions + optional external − level-3 keys; mass-window lookup; external fingerprints | built, tested |
| `insilico.py` | Fragmentor theoretical m/z per (SMILES, adduct), cached; ppm-scaled peak matching | built, tested |
| `metfrag.py` | channel F′ — MetFrag-lite 1/2-bond cuts, H shifts, linear/rarity-weighted explained intensity, gap-to-best (forum P1.1) | built, tested |
| `nnspec.py` | channel F2 — ICEBERG 2.1 neural spectrum→peak-list prediction (pretrained, zero-shot), cached per (SMILES, adduct, CE); matched-cosine scoring | built, smoke-tested |
| `f2_bridge.py` / `f2_batch.py` | F2 inside the stage2 fusion: request collection, ONE ms-pred subprocess per run, persistent prediction cache (`out/cache/f2_predictions.parquet`) | built |
| `f2_discrimination_check.py` | F2 sanity check: rank the true structure among mass-window decoys for real timsTOF queries | built |
| `fpchannel.py` | channel P — spectrum→fingerprint: v1 linear model trained per run on split-visible rows (leak-free by construction), v2 frozen encoder+head with the leak guard | built |
| `train_fp_head.py` | channel-P v2 fingerprint head on the frozen de-novo encoder; same leak-free-by-default exclusion; embeddings cached | built |
| `neighbours.py` | top-K visible structural neighbours (packed-bit Tanimoto) — only used by the optional analogue channel | built, tested |
| `train_denovo.py` | GPU training of the channel-P **v2** model (the repo-wide de novo transformer encoder, `src/enveda_casmi26/fp_model.py`); MLflow-tracked, checkpoints in `out/` | built, smoke-tested |
| `train_all.py` | one-command GPU driver: encoder then head (`train_denovo.py` → `train_fp_head.py`), preflight (data/splits/CUDA), post-run guard verification, `--smoke`; the thing to run on the GPU box | built |
| `spectra_utils.py` | self-contained spectrum cleaning, entropy/cosine similarity maths (no class-1 imports) | built, tested |
| `speclib.py` | self-contained analogue-spectrum library container + builder (own cache under `max_class2_exp/out/cache`) | built, tested |
| `annotations.py` | molecular formulas (RDKit, disk-cached) for the isomer panel + `cand_n_same_formula`; per-row source-library stats (public/timsTOF) for the `has_public_spectra` stratification | built, tested |
| `features.py` | the featurizer (all channels, R4), within-shortlist z-scoring, logistic/HGB fusion with seed ensembles + ablation, ranking/eval helpers (`truth_rank`, failure buckets, panels, protect, bootstrap) | built, tested |
| `stage2.py` | the CLI: pool → channels → train → evaluate → results/metrics/model; the §6 protocol | built (post-forum rewrite) |
| `predict_test.py` | serving: the real test set → pool (train structures only, no additions) → same `featurize` (R4) → fusion model → top-25 `submission.csv`; the Kaggle kernel's entry point | built |
| `test_stage2.py` | correctness tests | 135 checks passing |
| `out/` | results, metrics, models, caches (git-ignored) | — |

## 8. Implementation status and next steps

**The FORUM_FINDINGS.md P0/P1 protocol is implemented** (2026-11-05
rewrite of `features.py` / `stage2.py`, plus new `metfrag.py`,
`annotations.py`, `speclib.py`, `spectra_utils.py`):

| item | status |
|---|---|
| P0.1 class-share scenarios (16/45/39, 28/20/52 next to 20/60/20) | in metrics JSON |
| P0.2 same-formula isomer panel + `cand_n_same_formula` | in `stage2.py` (`panel_*`, `paneltop_*`); `f2_eval.py` still TODO |
| P0.3 failure buckets + conditional MRR | `outside_window` / `in_window_rank_gt25` / `rank_2_25` / `rank_1` |
| P0.4 `has_public_spectra` stratification (public = non-`enveda-*` ingest lib) | `by_public_spectra` + per-row source features |
| P0.5 spectrum-blind control + coverage guard | `--blind-control` |
| P0.6 paired bootstrap + seed-averaged HGB | `--n-boot 200`, `--model-seeds` |
| P1.1 MetFrag-lite channel | `metfrag.py` + `mtf_*` features |
| P1.2 new defaults | `--ppm 8.5`, `--tol 0.01` (sweeps pending) |
| P1.3 source flags + query context + ablation | `cand_is_addition`, `cand_public_rows`, `cand_tims_rows`, `mass_agreement_ppm`, `--drop-features`; the visibility block is **model-excluded by default** (`--keep-source-features` to opt in — alone it ranks the truth at MRR ≈ 1.0 by simulator construction, P1.3's "truth shortcut" caveat) |
| P1.5 rank-1 protection | `--protect-top1` |
| P1.7 analogue channel off by default | `--enable-analogues` (k = 8); `--leak-train` forces it on |
| §10 self-containment | no class-1 imports; own `spectra_utils`/`speclib` + cache under `max_class2_exp/out/cache` |

`test_stage2.py` grew to **135 checks** (all passing); the `--leak-train`
wiring self-test still reaches **MRR 1.000 / recall 1.000** at the new
defaults on the 30-molecule smoke. **Deferred:** the `f2_eval.py` half of
P0.2, the 5-dataset-seed sweep, COCONUT `--external-pool` runs,
linear-vs-sqrt and ppm/tol sweeps, the self-trained ICEBERG upgrade.

**Channels P and F2 are wired into the fusion** (2026-11-05): channel P
runs by default (v1 linear per run, v2 frozen artifacts auto-detected and
leak-guarded), channel F2 runs by default when the ms-pred env + weights
exist (subprocess bridge, persistent cache). No new stage2 flags —
presence of the artifacts decides, and both channels degrade to zero
features when unavailable. All learned-channel trainers
(`train_denovo.py`, `train_fp_head.py`) are leak-free *by default*: every
`dataset-creation/out/*.parquet` molecule is excluded and recorded in the
checkpoint meta, and stage2 refuses artifacts whose guard does not cover
the split being evaluated.

**Next steps, in order:**

1. Full clean run on `difficulty2.parquet` with new defaults
   (`--blind-control`, `--model-seeds 3`), compared against the pre-forum
   baseline (ppm 10, analogue on).
2. Sweeps that the forum says are cheap but real: `--ppm 5/8.5/10`,
   `--tol 0.005/0.01`, `--enable-analogues` contribution run.
3. Channel-P v2 artifacts on the GPU: `train_all.py` (one command =
   `train_denovo.py` then `train_fp_head.py`, both leak-free by default);
   then re-run 1 and compare v1 vs v2 fusion.
4. The results section below, then the 5-seed × multi-model-seed error
   bars (P0.6) before believing any delta.

### 7.1 Training the channel-P v2 model (`train_all.py`)

One command runs both trainers on the GPU box (encoder, then the head on
the frozen encoder), refuses to start without the eval-split parquets or a
CUDA device, and verifies the leak guards of the resulting checkpoints:

```bash
python max_class2_exp/train_all.py --no-mlflow      # full training
python max_class2_exp/train_all.py --smoke           # ~1-minute end-to-end check
python max_class2_exp/train_all.py --include-eval-molecules   # final Kaggle model only
```

What the GPU box needs (rsync the whole repo minus the heavy/irrelevant
parts — `data/train.parquet`, `extern/`, `max_class1_exp/`, `out/`
artifacts+caches, venvs, git, mlflow state; ~590 MB, `train_clean.parquet`
dominates):

```bash
cd <repo parent>   # the directory that contains enveda_casmi/
rsync -avhP \
  --exclude '.venv/' --exclude '.git/' --exclude '__pycache__/' \
  --exclude 'extern/' --exclude 'max_class1_exp/' --exclude 'spectra-viz/' \
  --exclude 'data/.spectra-viz/' \
  --exclude 'data/train.parquet' --exclude 'data/test.parquet' \
  --exclude 'data/test_clean.parquet' --exclude 'data/sample_submission.csv' \
  --exclude 'max_class2_exp/out/' --exclude 'dataset-creation/out/cache/' \
  --exclude 'mlruns/' --exclude 'mlflow.db' \
  enveda_casmi/ <gpu-box>:enveda_casmi/
# on the box: cd enveda_casmi && uv sync  (pyproject has all deps;
# Linux torch wheels are CUDA builds)
```

Without `dataset-creation/out/*.parquet` the driver refuses to train: the
guard would exclude nothing and stage2 would reject the checkpoints. scp
back `out/denovo_v1.pt`, `out/denovo_tokenizer.json`,
`out/denovo_fp_head.pt` (`.last.pt` optional) and the next stage2 run
auto-detects channel-P v2.

The individual trainers, if you want manual control:

```bash
# full run (defaults: 500k spectra, 2 epochs, 512-dim/4-layer encoder)
uv run python max_class2_exp/train_denovo.py

# smoke test (~3 training spectra, tiny model, seconds)
uv run python max_class2_exp/train_denovo.py --spectra 3 --epochs 1 \
    --max-rows 400 --val-structures 10 \
    --embed-dim 64 --n-layers 1 --n-heads 2 --ff-dim 128 \
    --out max_class2_exp/out/denovo_smoke.pt
```

Writes the best checkpoint to `out/denovo_v1.pt` (resume point
`out/denovo_v1.pt.last.pt`) and the tokenizer to
`out/denovo_tokenizer.json`; training data defaults to
`data/train_clean.parquet` when present (else `data/train.parquet`;
`train_cut.parquet` only when passed explicitly), enveda-180 rows are
excluded, a structure-level validation split is carved, and **every
molecule of every `dataset-creation/out/*.parquet` split is excluded and
recorded in the checkpoint `meta`** (leak-free by default;
`--include-eval-molecules` opts out for the final Kaggle model). Then
train the fingerprint head:

```bash
uv run python max_class2_exp/train_fp_head.py
```

→ `out/denovo_fp_head.pt` (same exclusion + meta; embeddings cached in
`out/cache/fp_embed_cache.npz`). The next stage2 run auto-detects the pair.
Inspect runs with
`uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`.

### 7.2 `stage2.py` — the experiment CLI

```bash
# the full protocol on the default split (new post-forum defaults)
uv run python max_class2_exp/stage2.py --blind-control --model-seeds 3

# quick smoke tests
uv run python max_class2_exp/stage2.py --limit 30                 # clean, 30 eval molecules
uv run python max_class2_exp/stage2.py --leak-train --limit 30    # wiring self-test: MRR ~ 1

# the forum sweeps
uv run python max_class2_exp/stage2.py --ppm 5 --tol 0.005        # window/tolerance
uv run python max_class2_exp/stage2.py --enable-analogues          # analogue contribution
uv run python max_class2_exp/stage2.py --keep-source-features      # visibility-flag contribution
uv run python max_class2_exp/stage2.py --drop-features mtf_explained_linear,mtf_rarity_explained,mtf_gap_best
uv run python max_class2_exp/stage2.py --protect-top1 entropy_best # rank-1 protection
uv run python max_class2_exp/stage2.py --model-path out/difficulty2_stage2_model.joblib   # eval-only
```

Key flags (defaults after the forum rewrite): `--ppm 8.5` (was 10),
`--tol 0.01` (was 0.02), `--mtf-tol 0.01`, `--n-boot 200`,
`--model-seeds 1`, `--n-train 2000`, `--model both` (logistic + HGB; the
better one on internal validation is primary), `--k-neighbours 0` (analogue
channel off unless `--enable-analogues`), `--external-pool` (COCONUT),
`--out-dir`. The visibility/source block is excluded from the model unless
`--keep-source-features` (see §6). Outputs land in `out/` as
`{split}_stage2_results.parquet` (per-molecule ranks, buckets, panels,
top-25 keys), `..._stage2_features.parquet` (per-candidate feature rows)
and `..._stage2_metrics.json` (the full §6 protocol).

### 7.3 Serving the fusion on the real test set (Kaggle kernel)

`predict_test.py` is the serving path: every train row visible, the pool is
the train structures themselves — no library additions, because a test truth
absent from train cannot be added by name — featurized with the same
`featurize` (R4) with in-silico + metfrag-lite live, channel P v2 from
`--artifacts` (the guarded load trivially passes with no eval keys), F2 and
the analogue channel off. The fusion model comes from `--model` with its
training-time `dropped` features restored exactly like `--model-path`; the
cache under `--cache-dir` is **read, never rebuilt** (an upload mtime change
would otherwise trigger a silent `ensure_cache` rebuild and break byte-parity
with training).

```bash
uv run python max_class2_exp/predict_test.py --limit 20   # smoke on real test rows
uv run python max_class2_exp/predict_test.py              # full local pass
```

The Kaggle kernel (`scripts/build_kernel2.py` → `../kernels/casmi26-class2-sub`)
embeds the serving closure as base64 payloads (the class-1 build pattern;
no stage2/train code ships) and takes everything else from the attached
private dataset `<user>/enveda-class2-serve`:

| dataset file | what it is |
|---|---|
| `train_clean.parquet` | the cleaned train data the model was trained on |
| `cache/{molecules.parquet, row_key.npy, fps.npy, fps.hash}` | the molecule-key cache, read-only at serve time |
| `denovo_v1.pt` + `denovo_fp_head.pt` | channel-P v2 artifacts (guarded; `denovo_tokenizer.json` optional) |
| `difficulty2_stage2_model.joblib` | the fusion model — serve the one trained F2-off (the box run) |
| `formulas.parquet` | pre-seeds the RDKit formula cache (saves a few kernel minutes) |
| `rdkit-*.whl` | Kaggle's image ships no RDKit (found during the class-1 kernel work); the kernel pip-installs it offline into `/tmp` |

```bash
uv run python scripts/build_kernel2.py
# stage the dataset dir (table above; one flat dir, cache/ as a subdir),
# kaggle datasets create -p <staging dir>, attach it in kernel-metadata.json
# push the kernel dir, wait for "complete", then:
uv run python scripts/submit_kernel.py --kernel <user>/casmi26-class2-sub --version N
```

Serve-time skew notes: F2 and the analogue channel are off exactly as in the
box training run; the visibility/source block is model-dropped, so the
simulator's "truth is an addition" asymmetry cannot fire at serve time;
shortlists shorter than 25 (or empty windows / unusable query masses) are
padded with the mass-nearest pool structures. Serving refuses to run without
the v2 artifacts (`--allow-v1-fp` opts into the v1 fallback — not recommended,
the shipped model was trained on v2 fp features).

## 9. Results

*To be filled after the first clean run.*

## 10. Risks and open questions

- **Pool scale:** ~273 k structures locally vs. PubChem-scale on Kaggle —
  candidate counts and FP precision both shift; run with
  `--external-pool data/coconut_pool.parquet` to measure the effect.
- **Fingerprint collisions** between homologues at Morgan r2/2048.
- **Chemistry gap:** timsTOF train data is synthetic `enveda-180`; the test
  is natural-product-like (knowledge base Q4). Report per source library.
- **Is the FP model the right complexity?** If linear is too weak, the
  transformer head is the fallback — the channel interface stays unchanged.
- **Do analogues stay out?** The optional channel A may be too strong to
  leave off; the honest framing above (its evidence never touches the
  answer's spectra) and its measured contribution decide.
