# Class-1 experiment (`max_class1_exp/`)

How well can we identify **class-1 molecules**, i.e. molecules that already have
reference spectra in train, but measured in a different setting (another
instrument, adduct or collision energy)? This folder holds the experiment.
**Stage 1** (library search) and **Stage 2** (a learned re-ranker over the
stage-1 shortlists, `stage2.py`) are implemented and tested. Stage 3 is
planned and described below.

The eval data comes from the eval-set simulator: by default
`dataset-creation/out/difficulty1.parquet` (see `dataset-creation/README.md`).
Every eval molecule there has `change` = `instrument`, `adduct` or `ce`, and
only the rows that differ in that dimension stay visible in train.

---

## 1. Quick start

From the repo root (`enveda_casmi/`):

```bash
# 1. unit tests of the maths (no data, ~2 s / ~10 s)
uv run python max_class1_exp/test_stage1.py
uv run python max_class1_exp/test_stage2.py

# 2. end-to-end self-test: eval spectra deliberately put into the library -> MRR must be ~1
uv run python max_class1_exp/stage1.py --leak-eval --limit 100
uv run python max_class1_exp/stage2.py --leak-train --limit 100   # same idea for stage 2

# 3. the real evaluation (~1 min reading the library the first time, then cached)
uv run python max_class1_exp/stage1.py

# 4. stage 2: train the re-ranker on the fly and re-rank the stage-1 shortlists
uv run python max_class1_exp/stage2.py --n-train 300 --limit 100   # small run (~minutes)
uv run python max_class1_exp/stage2.py                             # full run (default 3000)
```

The split must exist and match the dataset cache (`load_split` checks this).
Stage 1 never builds or changes the dataset cache.

---

## 2. What stage 1 does

For each eval molecule:

1. **Query mass.** Every eval spectrum gives a neutral mass from its precursor
   m/z and adduct (`enveda_casmi26.pipeline.neutral_mass`); the molecule's query
   mass is the median.
2. **Candidates.** Every molecule with at least one *visible* train row whose
   exact mass lies within `--ppm` (default ±10 ppm) of the query mass. Because
   this works on the neutral mass, it also finds the answer when train only has
   it in another adduct. Median 52 candidates per molecule, at most ~470.
3. **Library.** The visible train spectra of all candidates (all instruments,
   adducts, polarities and CEs), about 467,000 spectra for the default split.
   Read once, in parallel, and cached in `out/cache/`.
4. **Cleaning** (query and library alike, after Li et al., *Nature Methods*
   2021):
   - remove peaks with m/z > precursor − 1.6 Da (the precursor region);
   - merge peaks closer than `--tol` (0.02 Da), greedily from the most intense;
   - remove peaks below `--noise` (1 %) of the base peak;
   - normalise intensities to sum 1.
5. **Similarity.** Each query spectrum is compared with every library spectrum
   of every candidate.
   - **Entropy similarity** (weighted, Li et al.): spectra with low entropy
     (S < 3) get their intensities raised to 0.25 + 0.25·S, then
     `sim = Σ_matched [(a+b)·ln(a+b) − a·ln a − b·ln b] / ln 4`. It is 1 for
     identical spectra and 0 for spectra without shared peaks.
   - **Cosine** on the same cleaned peaks, as the classic baseline.
   - Peaks match if they are within `--tol`; each query peak matches at most
     one peak per library spectrum.
6. **Aggregation and ranking.** Candidate score = mean over the query's
   spectra of the match score. Ranked descending; ties are broken by smaller mass
   error, then by key.

Four scoring variants are evaluated side by side:

| variant | match score of a query spectrum against a candidate |
|---|---|
| `cosine_best` | best cosine over the candidate's library spectra |
| `entropy_best` | best entropy similarity over the candidate's library spectra |
| `entropy_composite` | entropy similarity against **one merged spectrum** per candidate and polarity (all its library spectra averaged and re-centroided). Robust to CE differences, and similar to the test's merged-CE spectra |
| `entropy_combined` | mean of `entropy_best` and `entropy_composite` |

**Leakage guards.** The library contains only rows with `train_mask == True`,
and the code asserts that no eval row is in it (except in `--leak-eval` mode,
which exists to test the pipeline).

---

## 3. Parameters (`stage1.py`)

| flag | default | meaning |
|---|---|---|
| `--split` | `dataset-creation/out/difficulty1.parquet` | the eval split; any split file works (for level 2/3 the answer is never a candidate, by design) |
| `--ppm` | `10` | candidate mass window, ± ppm around the query mass |
| `--tol` | `0.02` | m/z tolerance (Da) for centroiding and peak matching |
| `--noise` | `0.01` | remove peaks below this share of the base peak |
| `--limit` | `0` (all) | only the first N molecules, for quick tests |
| `--workers` | CPUs − 2 | parallel processes |
| `--no-cache` | off | rebuild the library cache |
| `--leak-eval` | off | self-test: also put the eval spectra into the library. Every answer must then rank first (MRR ≈ 1) |
| `--out-dir` | `max_class1_exp/out` | where results go |

---

## 4. Outputs

In `max_class1_exp/out/` (git-ignored):

`<split>_stage1_metrics.json`: MRR@25, top-1, top-10 and top-25 accuracy per
variant, overall and per `change`, plus the candidate recall and runtimes.

`<split>_stage1_results.parquet`: one row per eval molecule.

| column | meaning |
|---|---|
| `molecule_id`, `key`, `change`, `difficulty` | from the split |
| `n_eval_spectra`, `n_candidates`, `n_library_spectra` | sizes |
| `truth_in_candidates` | the answer is in the mass window (if not, it cannot be found) |
| `query_mass` | neutral mass used for the window |
| `rank_<variant>` | rank of the answer (0 = not among the candidates) |
| `rr_<variant>` | reciprocal rank, 0 beyond rank 25 |
| `truth_score_<variant>`, `top_score_<variant>` | score of the answer and of the best candidate (useful for a confidence threshold later) |
| `top25_keys`, `top25_smiles`, `top25_scores` | the top 25 of `entropy_combined` |

The printed table looks like this:

```
MRR@25 (top-1 / top-10)   difficulty1.parquet
                                   overall            instrument        adduct            ce
n molecules                           1000 ...
candidate recall                     ...
cosine_best              0.xxx (0.xx/0.xx) ...
...
```

---

## 5. How correctness was checked

`test_stage1.py` (22 checks, all passing):

- entropy similarity equals the textbook form `1 − (2 S(AB) − S(A) − S(B)) / ln 4`
  on 200 random spectrum pairs, is symmetric, and lies in [0, 1];
  identical spectra give 1 and disjoint spectra give 0;
- the vectorised one-against-many versions equal the pairwise versions, for
  entropy and cosine;
- cosine equals the dot product of aligned unit vectors;
- a query peak is matched at most once per library spectrum; the tolerance
  edge (0.019 Da matches, 0.021 Da does not);
- the entropy weighting rule; the cleaning steps (precursor removal, noise,
  greedy centroiding, normalisation); the merged spectrum;
- the neutral mass of caffeine from `[M+H]+` and `[M+Na]+` within 2 ppm.

End to end: `stage1.py --leak-eval --limit 100` puts the eval spectra into the
library. Result: candidate recall 1.0 and MRR 1.000 for `cosine_best`,
`entropy_best` and `entropy_combined`; `entropy_composite` gets 0.993 (expected,
because the leaked spectrum is averaged with the molecule's other spectra).
This confirms that retrieval, library lookup, scoring and ranking are wired
correctly.

---

## 6. Results

*To be filled in after the full run* (`uv run python max_class1_exp/stage1.py`).

Things to look at:
- **candidate recall** below 1 means the mass window misses answers (raise `--ppm`);
- **per change type**: `ce` should be easiest, `adduct` hardest for pure
  spectrum matching;
- **entropy vs cosine**: the literature expects entropy to win, most of all
  across instruments;
- **composite vs best**: does merging a candidate's spectra help for `ce`?

---

## 7. Stage 2: learned re-ranker (`stage2.py`)

Stage 1 retrieves all candidates within ±10 ppm (recall 1.0) and ranks them by
spectral similarity; all remaining error is ranking error, concentrated in the
`adduct` change type (an adduct change shifts fragment m/z, so literal peak
matching misses matches). Stage 2 keeps the stage-1 shortlists and re-ranks
them with a small tabular model trained on ~16 deterministic features per
(query molecule, candidate) pair — logistic regression and/or scikit-learn's
`HistGradientBoostingClassifier`. No neural network, no deep learning.

**Features** (mean over the query's spectra; "best" = max over the candidate's
library spectra, as in stage 1):

| feature | definition |
|---|---|
| `entropy_best`, `cosine_best`, `entropy_composite` | the stage-1 scores (they are features now) |
| `nl_entropy_best`, `nl_cosine_best` | the same in **neutral-loss space** (`nl = precursor − fragment`; algebraically the "modified cosine": library peaks shifted by the precursor difference). This is what survives an adduct change |
| `matched_peaks`, `explained_intensity` | match statistics against the candidate's best-entropy library spectrum |
| `mass_error_ppm` | abs ppm error of the candidate |
| `n_library_spectra`, `n_candidates` | sizes (`n_candidates` is context only, never a model feature) |
| `has_same_adduct`, `has_same_polarity`, `same_instrument_family` | setting flags (share of the query's spectra) |
| `min_ce_gap` | min |query ce − candidate library ce| (nan → imputed to 100) |
| `n_source_libraries` | how many source libraries contain the molecule (a "well-known compound" prior) |

Continuous features are z-scored **within each shortlist** (the
`contrast_within_set` trick from `CandidateScorer`): the model learns how a
candidate compares to its competitors, not absolute values. Labels:
`y = (candidate == truth)`, exactly one positive per shortlist.

**Training data is generated on the fly, in memory, from the regular train
dataset** — never from a file, never from the eval split. For each sampled
molecule the simulator building blocks (`make_difficulty1.qualifying` /
`choose_rows`) hold out one setting as the pseudo-query, exactly like the eval
split does. Leakage rules:

- **R1** no eval molecule is ever a training query (no positive label for a
  molecule the model is evaluated on) and no split-hidden row is ever visible;
  eval molecules only appear in the roles they also play at evaluation time
  (label-0 decoys, visible library rows). Both are asserted.
- **R2** every training example holds the pseudo-query's rows out of the
  library, so the truth is reachable only through its other settings. Asserted.
- **R3** `change` / `difficulty` are simulator metadata, never features.
- **R4** one shared `featurize` function serves training and evaluation.
- **R5** all randomness from one seeded `np.random.default_rng(--train-seed)`.

10% of the training molecules are held out as internal validation (model
selection between logistic and HGB, reported in the metrics); the evaluation
split is then used once, untouched.

### Usage

```
uv run python max_class1_exp/stage2.py [--split ...] [--ppm 10] [--tol 0.02] [--noise 0.01]
    [--n-train 3000] [--train-mix instrument=0.4,adduct=0.3,ce=0.3] [--train-seed 2027]
    [--model logistic|hgb|both] [--limit N] [--workers N] [--no-cache]
    [--leak-train] [--train-on-eval] [--model-path out/<split>_stage2_model.joblib]
    [--out-dir max_class1_exp/out]
```

| flag | meaning |
|---|---|
| `--n-train`, `--train-mix`, `--train-seed` | size, change-type mix and seed of the on-the-fly training set (mix semantics as in `make_difficulty1.py`; quotas filled instrument → adduct → ce, shortfalls logged, never padded) |
| `--model` | which re-ranker to fit; `both` (default) fits logistic and HGB and reports the better one on internal validation as primary (marked `*` in the table) |
| `--model-path` | **skip training**: load a model saved by a previous run (`out/<split>_stage2_model.joblib`) and only evaluate. Training and evaluation are normally one command; this separates them. The feature list is checked against the model, and any split works (e.g. a split with another seed, or one built later) |
| `--leak-train` | self-test: train on the eval shortlists with their spectra in the library — MRR(stage2) must be ≈ 1. Never use for real numbers |
| `--train-on-eval` | diagnostic, not the headline: also draw training queries from the eval molecules (held-out setting disjoint from the split's hidden rows). Run it with the same `--train-seed` as a clean run; the per-change metric delta (leaked − clean) measures the label-leakage dispute |
| `--limit` | only the first N *eval* molecules (training is unaffected) |
| `--train`, `--cache-dir`, `--ppm-tol`, `--rebuild-cache` | dataset cache flags, as in `make_difficulty1.py` |

### Memory

The training-library build is the pipeline's memory peak (it is where a run
dies if the box runs out of RAM). Worker processes each materialize one full
parquet row group before taking their subset, so the concurrent readers are
capped at 8; the eval library and the 2.5M-row metadata frame are freed before
the build starts. The finished libraries are small (~0.5 GB of arrays even at
`--n-train 6000`) — only the transient reads spike. If a run dies silently,
check the kernel log for an OOM kill (`journalctl -k | grep -i oom`), and run
big commands in an interactive shell or with `tmux set -g remain-on-exit on`
so the exit message is not lost with the pane.

### Outputs

In `max_class1_exp/out/` (git-ignored), with the usual `_<split>` stem:

- `<split>_stage2_results.parquet` — all stage-1 columns plus, per model,
  `rank_stage2_<model>`, `rr_stage2_<model>`, `truth_score_/top_score_stage2_<model>`;
  the primary model also gets the unsuffixed `rank_stage2`, `rr_stage2`,
  `p_truth_stage2`, `truth_score_stage2`, `top_score_stage2`;
- `<split>_stage2_features.parquet` — one row per candidate (molecule_id,
  change, cand_key, label, all feature columns);
- `<split>_stage2_metrics.json` — params, training summary (shortfalls, drops,
  class balance, internal validation, feature importances) and the stage-1
  summary structure extended with `stage2_logistic` / `stage2_hgb`;
- `<split>_stage2_model.joblib` — the chosen model with the feature list and
  the z-scoring conventions.

The printed table is the stage-1 table with the `stage2_logistic` /
`stage2_hgb` rows added (primary marked `*`).

### Stage 2 results

*To be filled in after the full run* (`uv run python max_class1_exp/stage2.py`).

Success criteria (STAGE2_PLAN.md §6): overall MRR@25 of `stage2` ≥
`entropy_best`'s 0.765; `adduct` clearly above stage 1's 0.393 (0.5+ would be
a clear win); `ce` and `instrument` not degraded; neutral-loss and/or entropy
features on top of the feature importances.

Validation done so far (all with `--limit 100`):

- `test_stage2.py`: 24 checks, all passing (neutral-loss equivalence with the
  shifted-peak formulation, match statistics, featurize = pairwise
  similarities + determinism, within-shortlist z-scoring, one positive label
  per shortlist, end-to-end sampler → fit → predict on synthetic data);
- `--leak-train`: MRR(stage2) 0.995 ≈ 1.0 — features → labels → fit →
  predict → rank are wired correctly.

---

## 8. Next stages (not implemented yet)

### Stage 3: learned cross-instrument spectrum embedding

A Siamese network in the style of MS2DeepScore that learns structural
similarity from pairs of spectra. Positive pairs: the same molecule on different
instruments, adducts or CEs (train has ~2,500 molecules with timsTOF and
other-instrument spectra and ~69,000 with several adducts). The embedding
similarity becomes a stage-2 feature. Pretrained DreaMS embeddings are a
shortcut worth comparing.

### Other ideas noted for later

- **Polarity handling:** compare only same-polarity library spectra, or learn
  how `[M+H]+` maps to `[M-H]-`.
- **Query merging:** also merge the query's own spectra into one composite.
- **Parameter sweep:** `--tol` (0.01 / 0.02 / 0.05), `--noise`, top-N peaks,
  `--ppm`.
- **Per source library:** report the score separately for `enveda-180` and
  `enveda-np-examples`, because the test is natural-product-like.
- **Confidence / gating for the full test:** the class is hidden on Kaggle. Use
  `top_score_*` to decide when a library hit is trustworthy, and calibrate it
  on difficulty 1 against difficulty 2 (where the answer is never in the
  library), so that confident library hits go on top and database or de novo
  candidates follow.
- **Several seeds:** run stage 1 on difficulty-1 splits with other seeds to
  get error bars.

---

## 9. Files

| file | content |
|---|---|
| `spectra.py` | cleaning, entropy weighting, entropy and cosine similarity (one query against many), neutral losses, match statistics, merged spectra |
| `library.py` | query mass, candidate index (mass window over visible molecules), parallel library loading and cache (with per-spectrum precursor, ce and source library) |
| `stage1.py` | the stage-1 evaluation CLI |
| `stage2.py` | the stage-2 re-ranker CLI (on-the-fly training, shared featurizer, logistic + HGB, evaluation) |
| `test_stage1.py` | stage-1 correctness tests |
| `test_stage2.py` | stage-2 correctness tests (incl. an end-to-end smoke test on synthetic data) |
| `STAGE2_PLAN.md` | the implementation plan for stage 2 |
| `out/` | results, models and library cache (git-ignored) |
