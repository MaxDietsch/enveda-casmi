# Stage 2 implementation plan: learned re-ranker for the class-1 experiment

A self-contained plan for implementing stage 2 of `max_class1_exp/`. It assumes
access to this repository and no prior conversation. Read sections 1-3 first;
they fix the design. Sections 4-6 are the concrete work items.

---

## 1. Goal and design (fixed decisions — do not change these)

Stage 1 (done, in `stage1.py`) retrieves, for every eval molecule, all train
molecules within ±10 ppm of the query mass and ranks them by spectral
similarity. Current quality on `dataset-creation/out/difficulty1.parquet`
(1000 molecules, change mix instrument 0.4 / adduct 0.3 / ce 0.3):

```
MRR@25                overall   adduct   ce    instrument
entropy_best           0.765   0.393   0.933   0.918
entropy_composite      0.750   0.290   0.969   0.929
```

Candidate recall is 1.0: the answer is always among the candidates. All error
is ranking error, concentrated in the `adduct` change type (an adduct change
shifts fragment m/z by the adduct mass difference, so literal peak matching
misses matches).

Stage 2 is a **learned re-ranker** over stage-1 shortlists:

1. For each (eval molecule, candidate) pair, compute ~16 numeric features.
   Features are produced by deterministic peak matching (existing code plus two
   new similarity functions) — *no neural network, no deep learning*.
2. A small tabular model (logistic regression and/or gradient-boosted trees
   from scikit-learn) maps the features of one candidate to
   P(this candidate is the truth).
3. Candidates are re-ranked by that probability.

**Training data** is generated **on the fly, in memory, from the regular train
dataset**, never from a file. The 1000 eval molecules of the evaluation split
are never *training queries* — the model is never told the answer for a
molecule it will be evaluated on, and their split-hidden rows are invisible to
training. They do appear in the roles they also play at evaluation time: as
mass-window decoys (label 0) and as library content through their visible
rows. The evaluation split is then used once, untouched, as a clean final
test — no cross-validation needed.

### Hard rules (leakage and correctness guards)

- R1. The model must never be *told the answer* for an eval molecule, and
  training must never see a split-hidden row. Concretely:
  (a) no molecule whose `key` appears in the evaluation split's `truth.key`
  may be a training query — i.e. no positive training label may be an eval
  molecule (assert: eval molecule indices ∩ training-query molecule indices
  = ∅); (b) the training visibility mask starts from `split.train_mask`, so
  every row the eval split hides (`hidden_row_ids`, including the eval rows)
  is invisible to training as well (assert: all training library rows satisfy
  `split.train_mask`). Eval molecules MAY appear in training only in the
  roles they play at evaluation time: as decoy candidates (label 0) and via
  their visible rows in the library. Library overlap is the task itself —
  what is forbidden is the *label*, not the molecule.
- R2. Every training example must hold out the query's setting from the
  library: the rows chosen as pseudo-query for molecule m must NOT be in the
  training library for that example (assert), and the truth must be reachable
  only through its *other* settings (this is what makes the training task
  identical to the eval task).
- R3. The split's `change` column (and `difficulty`) is simulator metadata.
  It must never be used as a model feature — the hidden Kaggle test has no
  such label. It is only used for stratification and for the per-change
  metrics table.
- R4. Features at training and evaluation time must come from one shared
  featurization function. No duplicated logic.
- R5. All randomness comes from one seeded `np.random.default_rng(--train-seed)`.
- R6. Follow the repo conventions: `common.log` for output, `common.Progress`
  for progress, CLI flags mirroring `stage1.py`, results written to
  `max_class1_exp/out/` (git-ignored).

---

## 2. Background: the pieces that already exist (reuse, do not rewrite)

Paths below are relative to the repo root `enveda_casmi/`.
`max_class1_exp/*.py` put their own directory and `src/` and `dataset-creation/`
on `sys.path` (see `library.py` lines 25-32 for the pattern).

### `max_class1_exp/spectra.py` — similarity maths (pure numpy, tested)

- `CleanParams(tol=0.02, noise=0.01, precursor_margin=1.6, max_peaks=0)`
- `clean_spectrum(mz, inten, precursor_mz, p) -> (mz, p)` — cleaning per Li et
  al. 2021; output intensities sum to 1.
- `entropy_weight(p)`, `unit_norm(p)`, `merge_spectra(spectra, tol)`
- `entropy_one_vs_many(q_mz, q_p, l_mz, l_p, l_spec, n_spec, tol) -> array`
  — entropy similarity of one query spectrum against `n_spec` library
  spectra. `q_p`/`l_p` must be entropy-weighted and sum to 1 per spectrum;
  `l_spec` maps each library peak to its spectrum index.
- `cosine_one_vs_many(q_mz, q_u, l_mz, l_u, l_spec, n_spec, tol) -> array`
  — same with unit-L2 inputs.
- `entropy_similarity(mz_a, p_a, mz_b, p_b, tol)` / `cosine_similarity(...)` —
  pairwise wrappers for tests.

### `max_class1_exp/library.py` — candidates and the library

- `query_mass(precursor_mz, adducts) -> float` — median neutral mass.
- `CandidateIndex(molecules, row_key, train_mask)` — molecules with >= 1
  visible row, sorted by exact mass; `.window(mass, ppm)` returns
  `(molecule indices, signed ppm error)`.
- `build_library(rows, row_key, train_path, params, split_info, workers,
  use_cache) -> Library` — cleaned spectra of the given train rows, cached in
  `out/cache/` (tag = hash of split content, cache hash, clean params, rows).
- `Library`: per-spectrum table with `row_id, mol, adduct, ionization_mode,
  instrument_type`; arrays `mz, p, pw` (entropy-weighted), `pu` (unit norm),
  `spec` (peak→spectrum index); `peak_start[i]:peak_start[i+1]` = peak range
  of spectrum i; methods `spectra_of(m) -> (s, e)`, `peaks(s, e)`, `get(i)`;
  attributes `mol`, `positive` (bool per spectrum).
- Module constants: `LIB_COLUMNS`, `INFO_COLUMNS = ["adduct", "ionization_mode",
  "instrument_type"]` (columns read from the train parquet for the table).

### `max_class1_exp/stage1.py` — the evaluation to mirror

- `score_molecule(i)` shows the pattern: build per-candidate contiguous
  spectra ranges, run `entropy_one_vs_many` / `cosine_one_vs_many` per query
  spectrum, aggregate with `np.maximum.reduceat(...)` to a per-candidate
  "best", average over query spectra; `_composite(m, positive)` builds and
  caches the merged per-polarity spectrum.
- `rank_order(scores, abs_ppm, mols)` — tie-breaking convention
  (score desc, |mass error| asc, molecule index). Reuse the same convention.
- Main flow: `load_split` → `CandidateIndex` → per-molecule query packing →
  `build_library` (only the candidates' visible rows: 
  `np.flatnonzero(train_mask & np.isin(row_key, need))`) → parallel scoring
  with `Pool` → results parquet + metrics json + printed table.
- Leakage guard pattern (lines 184-186): assert library rows are all visible
  and no eval row is among them.
- `--leak-eval` self-test mode: eval spectra deliberately added to the
  library; MRR must be ~1. Keep the analogous idea for stage 2 (see 6.3).

### `dataset-creation/make_difficulty1.py` — the simulator building blocks

- `qualifying(meta) -> dict` — `change -> {molecule: [qualifying adducts]}`:
  which molecules support which change type. Requires `meta` with columns
  `eligible, is_tims, adduct, ce, mol` (produced by `common.attach_keys`).
- `choose_rows(rng, meta_cols, rows, change, adducts) -> (eval_rows,
  visible_rows)` — for one molecule and change type, splits its rows into
  held-out query rows ("eval") and visible rows. `meta_cols` is
  `{c: meta[c].to_numpy() for c in ("eligible", "is_tims", "adduct", "ce")}`;
  `rows` are the molecule's train row ids (`common.mol_rows(meta)` gives the
  `(order, start)` index so that molecule m owns `order[start[m]:start[m+1]]`).
- Change semantics: `instrument` = query is timsTOF rows, visible = non-timsTOF
  rows in the same adduct; `adduct` = query in adduct A, visible = timsTOF
  rows in other adducts; `ce` = both timsTOF same adduct, disjoint CE sets.

### `dataset-creation/common.py` — shared utilities

- `prepare(args) -> (meta, molecules, row_key)` — loads the train parquet
  metadata and the dataset cache (pass `--train`, `--cache-dir`,
  `--ppm-tol` like `make_difficulty1.py` does).
- `mol_rows(meta)`, `log(msg)`, `Progress(label, n)`,
  `draw_n_spectra(rng, n)`, `parse_mix("instrument=0.4,...", names) -> dict`,
  `split_counts(n, shares) -> dict of quotas`.
- `Eval`-side: `max_class1_exp/dataloader.py` exposes
  `load_split(path) -> split` with `split.info` (parquet metadata dict,
  including `params.cache_dir` and `content_hash`), `split.truth` (DataFrame:
  one row per eval molecule with `molecule_id, key, smiles, difficulty,
  change, eval_row_ids, hidden_row_ids, ...`), `split.train_mask` (bool array
  over all train rows), `split.eval_row_ids`, `split.eval_spectra()` (DataFrame
  with `molecule_id, ms2_mzs, ms2_normalized_intensities, precursor_mz,
  adduct, ionization_mode, ...` — **verify** in `dataloader.py` whether
  `instrument_type` and `collision_energy` are included; if not, extend
  `eval_spectra()` to also read them from the train parquet via row ids).

### Existing trained model precedent

`src/enveda_casmi26/scoring.py` (`CandidateScorer`) is an older logistic
re-ranker over contrasted candidate features, trained by
`scripts/train_scorer.py`. Do not import it into stage 2 (different features),
but **copy its key trick**: features are z-scored *within each candidate set*
(`contrast_within_set`) before fitting and predicting, so the model learns
from how a candidate compares to its competitors, not from absolute values.

---

## 3. The two new similarity functions (in `spectra.py`)

### 3.1 Neutral-loss similarity (= "modified cosine")

**Key identity, use it to avoid writing two functions:** shifting library peaks
by the precursor difference and matching, `|mz_q - (mz_l + prec_q - prec_l)|
<= tol`, is algebraically the same as matching neutral losses
`|nl_q - nl_l| <= tol` with `nl = precursor_mz - peak_mz`. So ONE
implementation covers both "modified cosine" and "neutral-loss similarity":
transform both spectra to neutral-loss space and reuse the existing matching.

Add to `spectra.py`:

```python
def neutral_losses(mz, precursor_mz):
    """nl = precursor_mz - mz, for cleaned spectra (mz < precursor - margin).
    Keep the intensity array unchanged; returns sorted nl m/z."""
```

Then neutral-loss similarity = `cosine_one_vs_many` / `entropy_one_vs_many`
applied to the transformed arrays (unit-norm / entropy-weighted as usual).
Because the cleaning already removed peaks above `precursor - 1.6`, neutral
losses are always > 1.6 Da on both sides — consistent.

### 3.2 Match statistics

The existing `_match` returns match pairs; write one helper that, for a query
spectrum against one library spectrum, returns
`(n_matched_peaks, matched_query_intensity_share)`. The "explained intensity
share" is the sum of the query peaks' p-values that found a partner, divided
by the total (which is 1 after normalisation). Features below use the stats of
the library spectrum that achieved the best entropy similarity (the
"best-match spectrum").

Unit tests for both go in `test_stage1.py` (same style as existing tests):
neutral-loss equivalence with the shifted-match formulation on random pairs;
match stats correctness on hand-built spectra.

---

## 4. New file `max_class1_exp/stage2.py`

One CLI with two phases in a single run (train, then evaluate):

```
uv run python max_class1_exp/stage2.py \
    [--split dataset-creation/out/difficulty1.parquet] \
    [--ppm 10] [--tol 0.02] [--noise 0.01] \
    [--n-train 3000] [--train-mix instrument=0.4,adduct=0.3,ce=0.3] \
    [--train-seed 2027] [--model both] \
    [--limit N] [--workers N] [--no-cache] \
    [--leak-train] [--train-on-eval] \
    [--out-dir max_class1_exp/out]
```

- `--model`: `logistic` | `hgb` | `both` (default `both`; both are fitted, the
  better one on internal validation is reported as primary, both appear in
  the metrics).
- `--leak-train` (self-test): train on the eval molecules themselves and
  evaluate on them. Expected MRR ~1 for the training molecules' own
  shortlists — proves the wiring (features → labels → fit → predict → rank)
  is correct. Never use for real numbers.
- `--train-on-eval` (diagnostic, NOT the headline): relaxes only the label
  half of R1 — training queries are drawn from the eval molecules too
  (each with a held-out setting *different* from the eval's hidden rows; the
  split-hidden rows still never appear). Its purpose is to *measure* the
  leakage dispute: run the clean training and this mode with the same
  `--train-seed`, and report the metric delta (leaked − clean) per change
  type in the log. The delta is the memorization effect; if it is ~0, the
  label rule is demonstrably harmless; if it is positive, it is the exact
  number the clean default protects against. Headline numbers must always
  come from the clean training.
- `--limit`: only the first N *eval* molecules (same semantics as stage 1).

### 4.1 Phase 1: build training shortlists on the fly

```
1. load split (for the exclusion set and later eval):
       eval_keys = set(split.truth["key"])
2. meta, molecules, row_key = common.prepare(args)   # same flags as make_difficulty1
   order, start = common.mol_rows(meta)
   key_index = molecule key -> molecule index (as in stage1.py line 149)
   eval_mols = {int(key_index[k]) for k in eval_keys}
3. qual = make_difficulty1.qualifying(meta)
   pool per change type = qual[change] keys - eval_mols     (R1)
   quotas = common.split_counts(args.n_train, parse_mix(args.train_mix))
   fill quotas in the order instrument, adduct, ce (instrument is the rarest —
   the simulator does the same); log shortfalls, do not pad.
4. rng = np.random.default_rng(args.train_seed)
   For each sampled molecule m with change type c:
       rows = order[start[m]:start[m+1]]
       query_rows, visible_rows = make_difficulty1.choose_rows(
           rng, meta_cols, rows, c, qual[c][m])
       # query spectra: read raw train rows query_rows (ms2_mzs, intensities,
       # precursor_mz, adduct, ionization_mode, instrument_type,
       # collision_energy) and clean each with clean_spectrum;
       # keep the query's adduct/instrument/ce per spectrum for the flags.
5. training visibility mask: start from `split.train_mask` (the eval split
   already hides the eval molecules' hidden rows, so eval molecules keep
   exactly their eval-visible rows and can legitimately appear as decoys and
   library content, R1), then additionally hide the union of held-out query
   rows over all sampled training molecules. This mirrors the eval setup:
   each training query is reachable only through its other settings. Build
   the union before featurizing.
6. cidx = CandidateIndex(molecules, row_key, training_mask)
   candidates = cidx.window(query_mass, ppm)      # same as stage 1
   assert truth m is among candidates; if not (mass edge case), drop the
   molecule from training and count it in the log.
7. library rows = visible rows of all candidates:
       np.flatnonzero(training_mask & np.isin(row_key, candidate_mols))
   build_library(...) with a split_info dict that embeds the training-seed and
   "stage2-train" marker in the cache tag. Because shortlists overlap, build
   ONE library for the union of all training molecules' candidate rows.
   Assert (R2): no query row of any sampled molecule is in the library rows;
   and (R1): all library rows satisfy split.train_mask (no split-hidden row).
```

Determinism note: sampling must consume the rng in a fixed order (per change
type, sorted molecule iteration) so the same `--train-seed` gives the same
training set.

### 4.2 Phase 2: features (shared by training and evaluation)

One function, used verbatim in both phases (R4):

```python
def featurize(query_spectra, query_meta, cands, ppm_err, lib) -> DataFrame
    # one row per candidate, columns:
```

| feature | definition (all aggregated over the query's spectra like stage 1: mean over spectra, "best" = max over the candidate's library spectra) |
|---|---|
| `entropy_best` | stage-1 entropy similarity, best library spectrum |
| `entropy_composite` | stage-1 entropy vs merged per-polarity spectrum |
| `cosine_best` | stage-1 cosine, best library spectrum |
| `nl_cosine_best` | cosine in neutral-loss space (3.1), best library spectrum |
| `nl_entropy_best` | entropy similarity in neutral-loss space, best library spectrum |
| `matched_peaks` | matched-peak count of the best-entropy library spectrum |
| `explained_intensity` | matched query intensity share of the best-entropy match |
| `mass_error_ppm` | abs ppm error of the candidate (from `cidx.window`) |
| `n_library_spectra` | number of the candidate's visible spectra |
| `n_candidates` | shortlist size (constant within a shortlist; context) |
| `has_same_adduct` | share of query spectra whose adduct appears among the candidate's library spectra |
| `has_same_polarity` | share of query spectra whose polarity the candidate has |
| `same_instrument_family` | 1 if the candidate has >= 1 library spectrum on the same side of the timsTOF / non-timsTOF divide as the query |
| `min_ce_gap` | min |query ce − candidate library ce| over the candidate's spectra (nan → impute to a large constant before fitting) |
| `n_source_libraries` | OPTIONAL: number of source libraries containing the molecule, IF a source/library column exists in `molecules.parquet` (check columns first; drop the feature if not available) |

Binary flags stay 0/1 raw; continuous features are z-scored **within each
shortlist** (the `contrast_within_set` trick) before fit and predict.
`n_candidates` and other within-shortlist-constant columns are dropped before
z-scoring (std = 0) — keep them out of the model.

Labels: `y = (candidate molecule index == truth molecule index)`, exactly one
1 per shortlist.

### 4.3 Phase 3: fit

- Reserve 10% of training molecules (random by rng) as internal validation.
- `logistic`: `LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced")`
  on the z-scored features.
- `hgb`: `HistGradientBoostingClassifier` (shallow: `max_depth=3`,
  `max_iter=300`, early stopping on the internal validation molecules,
  `class_weight="balanced"` if supported by the installed sklearn version,
  else pass `sample_weight`). Keep it small — a few thousand training
  molecules is a small dataset (R6: no deep learning).
- Feature importances / coefficients saved to the metrics json.
- Persist the chosen model + feature column list + z-scoring conventions with
  joblib to `out/<split>_stage2_model.joblib`.

### 4.4 Phase 4: evaluate on the untouched split

Reproduce the stage-1 flow on `args.split` exactly (queries, candidate window,
library from `split.train_mask`), then:

1. featurize with the same `featurize` function,
2. predict P(truth) per candidate with the fitted model,
3. rank by probability desc, ties by |mass error| asc, then molecule index
   (`rank_order` convention).

Because `entropy_best` etc. are features, the stage-1 scores and ranks are
already computed — emit them too, so the metrics file contains stage 1 and
stage 2 side by side.

Outputs, mirroring stage 1 naming:

- `out/<split>_stage2_results.parquet`: stage-1 columns (`rank_*`, `rr_*`,
  `truth_score_*`, `top_score_*`, `top25_*`) plus `rank_stage2`, `rr_stage2`,
  `p_truth_stage2`, `truth_score_stage2`, `top_score_stage2`, and the per-
  candidate feature matrix optionally as a separate parquet
  (`<split>_stage2_features.parquet`, one row per candidate with
  `molecule_id`, feature columns, `label` for eval molecules).
- `out/<split>_stage2_metrics.json`: params (train seed, n-train, mix, model),
  training summary (n molecules, n shortlists, shortfalls, class balance,
  internal-validation scores, feature importances), then the same summary
  structure as stage 1 (`overall` + `by_change`, per method: stage-1 variants
  AND `stage2`).
- The printed table: same layout as stage 1, with `stage2_logistic` and
  `stage2_hgb` rows added under the stage-1 rows.

### 4.5 Parallelism

Follow stage 1: build query packs + library in the main process, then a
`Pool` with `imap` over molecules for featurization; the model fit happens in
the main process. Module-level globals for the library/queries pattern
(`_LIB`, `_QUERIES`) can be reused. Keep the composite-spectrum cache per
worker (`_COMPOSITE` pattern) — merged spectra are reused across query
spectra.

---

## 5. Unit tests (`max_class1_exp/test_stage1.py` style)

Add a `test_stage2.py` (or extend the existing file) with tests that need no
data files:

1. Neutral-loss equivalence: for random spectrum pairs with known precursors,
   `nl_cosine(mz_q, prec_q, mz_l, prec_l)` equals cosine computed on
   library peaks shifted by `prec_q - prec_l` (tolerance edge cases included).
2. Match statistics: hand-built spectra with known overlaps →
   `matched_peaks`, `explained_intensity` exact.
3. Featurization determinism: same inputs → identical feature rows.
4. Within-shortlist z-scoring: mean 0, std 1 per shortlist; constant columns
   dropped.
5. Labels: exactly one positive per shortlist.
6. End-to-end smoke test on synthetic data: build ~20 fake molecules with
   spectra in two settings, run the training sampler + fit + predict,
   assert the truth's predicted probability is highest for most shortlists
   (>= 15/20), and assert a model fitted on a shortlist it trained on ranks
   that truth first (the `--leak-train` principle).

---

## 6. Validation protocol for the real run

1. `uv run python max_class1_exp/test_stage2.py` — all green.
2. `uv run python max_class1_exp/stage2.py --leak-train --limit 100` —
   wiring check: MRR(stage2) ≈ 1.0 on the molecules it trained on.
   (Logistic/entropy on leaked data should also stay ~1.)
3. `uv run python max_class1_exp/stage2.py --n-train 300 --limit 100` —
   small real run: library build works, training completes in minutes,
   metrics json sane, stage-2 overall MRR on the 100 molecules is in a
   plausible range (>= stage 1's is the goal but not guaranteed at 300).
4. `uv run python max_class1_exp/stage2.py` — full run
   (`--n-train 3000` default). Runtime budget: library build + featurization
   for 1000 eval + 3000 train molecules; stage 1 scores 1000 molecules in
   ~3 s after a ~30 s library read, so the full run should stay well under
   ~15 min on the current machine.

Success criteria for the full run:

- Overall MRR@25 of `stage2` >= `entropy_best`'s 0.765.
- `adduct` MRR clearly above stage 1's 0.393 (the neutral-loss features are
  aimed exactly there; a rise to 0.5+ would be a clear win).
- `ce` and `instrument` do not degrade (>= their stage-1 values).
- Feature importances put neutral-loss and/or entropy features on top —
  if not, investigate before trusting the model.

Failure analysis aids: `truth_score_stage2` and `p_truth_stage2` per molecule
identify the cases the model is unsure about; the per-change table shows
where losses concentrate.

---

## 7. Pitfalls (things the previous discussion established — do not re-derive)

- **Do not train on the eval split and report metrics on it without
  `--leak-train`.** The molecules may share spectra with the library (that is
  the task — class 1), and eval molecules may appear in training as label-0
  decoys and via their visible rows, but the model must never be *told the
  answer* (positive label) for a molecule it is later evaluated on, and must
  never see a split-hidden row.
- **Do not build training examples without holding out the query setting.**
  Same-setting examples are easy and teach the model the wrong conditional
  ("trust entropy blindly"), which miscalibrates exactly the cross-setting
  cases.
- **`change` is not a feature** (R3).
- **Library cache columns:** if `ce` / `instrument_type` are added to
  `INFO_COLUMNS` for the flags, the cached library tables from stage-1 runs
  lack the new columns. Either extend `_cache_tag` to include the column list,
  or force a rebuild (`--no-cache`) once after changing it. Handle missing
  `collision_energy` gracefully (nan → impute).
- **Feature drift between train and eval:** only one shared `featurize`
  function (R4), and the same cleaning params object in both phases.
- **Composite cache correctness:** the merged-spectrum cache
  (`_COMPOSITE`) must be cleared/kept consistent when the library changes
  between phases (training library vs eval library are different objects —
  use separate caches or key by library identity).
- **Mass-window edge cases:** some sampled training molecules will have no
  candidates or the truth outside the window; skip them, count in the log,
  never pad.
- **sklearn is already a dependency** (check `pyproject.toml`; if
  HistGradientBoostingClassifier is unavailable, logistic alone is an
  acceptable fallback — note it in the metrics).

## 8. Deliverables checklist

- [ ] `spectra.py`: neutral-loss transform + match-statistics helper (+ tests)
- [ ] `stage2.py`: CLI with train-on-the-fly sampler (R1, R2, R5),
      shared `featurize`, logistic + HGB fits, evaluation on the untouched
      split, outputs per 4.4
- [ ] `test_stage2.py` (or extended `test_stage1.py`)
- [ ] `README.md` in `max_class1_exp/`: new section "Stage 2" (usage, outputs,
      results placeholder), mirroring the stage-1 section's style
- [ ] One full run on `difficulty1.parquet` with the printed comparison table
      and the metrics json committed to `out/` (git-ignored, but paste the
      table into the README results section)
