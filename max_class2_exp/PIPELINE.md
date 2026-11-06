# The class-2 pipeline, step by step

This is the code-level companion to [README.md](README.md): what actually
happens, in order, when a spectrum enters the difficulty-2 experiment —
first at **inference** (one query molecule → a ranked candidate list), then
how the **training** machinery that produces the model works.

Everything below refers to code in this folder; function names are given so
you can jump straight to them (`stage2.py` for orchestration, `features.py`
for scoring, `pool.py`, `insilico.py`, `metfrag.py`, `spectra_utils.py`,
`speclib.py`, `annotations.py`).

```
uv run python max_class2_exp/stage2.py --blind-control --model-seeds 3
```

---

## Part A — Inference: what happens to one spectrum

### A0. Loaded once per run (before any spectrum is touched)

| artifact | what it is | source |
|---|---|---|
| `molecules.parquet` (274 k rows) | one row per structure: `key` (tautomer-canonical InChIKey14), `smiles`, `exact_mass`, `n_rows`, `n_eligible` | `dataset-creation` cache |
| `row_key.npy` (2.5 M) | train row → molecule index | same cache |
| `fps.npy` | Morgan r2/2048 fingerprints, packed uint64 | `common.fingerprint_matrix` |
| `split` (`difficulty2.parquet`) | the eval truths, `train_mask`, `library_additions()`, `excluded_library_keys`, `eval_row_ids` | `dataloader.load_split` |
| row sources | per-train-row `ingest_lib` code + is-timsTOF flag (npz-cached) | `annotations.load_row_sources` |
| formulas | key → RDKit `CalcMolFormula` (parquet-cached, incremental) | `annotations.formulas_for` |
| channel-P model | v2 encoder + fp head when `out/denovo_v1.pt` + `out/denovo_fp_head.pt` exist (leak-guarded), else a v1 linear model trained in the run | `fpchannel.build_fp_model` |
| F2 predictions | ICEBERG predicted spectra, `(smiles, adduct, ce)`-keyed, disk-cached | `f2_bridge.ensure_predictions` → `out/cache/f2_predictions.parquet` |

From these, `build_pool` (in `run()`) constructs the **candidate pool**:

* every molecule with ≥ 1 *visible* train row under the split mask,
* **plus the library additions** — the class-2 answers, re-inserted as
  structures with no spectra (`Split.library_additions()`),
* optionally an external pool (COCONUT extract, `--external-pool`),
* minus the level-3 exclusions,
* all sorted by exact mass (`Pool` invariant) with per-member
  `visible_rows`, `is_addition`, `is_external` bookkeeping.

Then `annotate_pool_formulas` / `annotate_pool_sources` attach each
member's molecular formula and its **visible** public/timsTOF row counts
(hidden rows never leak into features).

### A1. Spectrum → query (`build_eval_queries`, `load_query_spectra`)

For one eval molecule, the split hands over its 1–16 timsTOF spectra
(median 4, drawn geometric(0.25), capped 16 — `draw_n_spectra`). Each
spectrum is cleaned **exactly like a test spectrum would be**:

1. `dataloader.clean_peaks` — drop non-finite/non-positive peaks;
2. `spectra_utils.clean_spectrum` with `CleanParams(tol=0.01, noise=0.01)`:
   * drop peaks above `precursor_mz − 1.6` (the precursor region),
   * centroid peaks within `tol` Da (intensity-weighted merge),
   * drop peaks below 1 % of the base peak,
   * normalise intensities to sum 1.

The result per spectrum is a tuple
`(mz, p, positive, precursor_mz, adduct, ce_ev)` — the 6th element (mean
collision energy in eV, NaN when absent) is read only by the F2 channel.

**The query mass** is the *median implied neutral mass* over its spectra:
`speclib.query_mass` converts each `(precursor_mz, adduct)` with
`enveda_casmi26.pipeline.neutral_mass` and takes the median. A molecule
measured with several adducts therefore gets one consistent neutral mass,
and the spread of the per-spectrum implied masses is recorded as
`mass_agreement_ppm` (context only, never a model feature).

Query context recorded per molecule: `n_query_adducts`,
`n_query_polarities`, `has_precursor_peak`, `mass_agreement_ppm`.

### A2. Candidate retrieval (`Pool.window`)

The pool is mass-sorted, so retrieval is two binary searches:

```python
lo, hi = searchsorted(pool.mass, [m*(1 - 8.5e-6), m*(1 + 8.5e-6)])
cands   = arange(lo, hi)                      # member indices
err     = (pool.mass[lo:hi] - m) / m * 1e6    # signed ppm error per candidate
```

`--ppm 8.5` (forum P1.2: 99 % of timsTOF precursors agree within 5 ppm;
wider windows only add decoys). Typical shortlist: **50–300 candidates**,
of which a **median ~26 share the truth's formula** — that is the isomer
problem the whole experiment is about. The truth is guaranteed to be in
the pool (via the additions) but is *not* guaranteed to be inside this
window; if it isn't, the molecule lands in the `outside_window` failure
bucket and every ranking method scores 0 for it.

If the analogue channel is on (`--enable-analogues`, k = 8), the top-8
visible structural neighbours (Tanimoto ≥ 0.30, packed-bit search in
`neighbours.topk_neighbours`) and their cleaned visible spectra
(`speclib.build_library`) are fetched per candidate. By default the
channel is **off** — neighbours `{}` and an empty `Library`, so all
analogue features are 0.

### A3. Per-candidate features (`features.featurize` + `features.channel_block`)

One `(n_candidates × 33)` matrix per query. Feature list = `FEATURES` +
`n_candidates` (context only, never a model input). Grouped by channel:

**Channel M — mass & competition context** (always on):

| feature | definition |
|---|---|
| `mass_error_ppm` | \|signed ppm error\| of the candidate vs the query mass |
| `mass_error_ppm_max` | worst \|ppm\| over **every** spectrum's implied mass — the candidate must fit all adducts |
| `cand_n_same_formula` | how many *other* shortlist candidates share this candidate's RDKit formula (the isomer-competition axis, P0.2) |
| `cand_visible_rows`, `cand_tims_rows`, `cand_public_rows` | log1p of the candidate's visible train-row / timsTOF / public-library row counts |
| `cand_is_addition`, `cand_is_external` | pool membership flags |

> **Why the last four count-features are not model inputs by default:**
> within this simulator every eval truth is a spectrum-less *addition*,
> so `cand_is_addition` / `cand_visible_rows` alone rank the truth at
> MRR ≈ 1.0 (measured). The model would learn the simulator's
> construction rule, not chemistry. They are computed and saved in the
> feature frames, but `model_matrix` drops them unless
> `--keep-source-features` (see README §6).

**Channel A — analogue transfer** (off by default; all zeros then): for
each query spectrum, entropy/cosine similarity against every neighbour
spectrum of the candidate (`entropy_one_vs_many` over one contiguous peak
block), plus the Tanimoto-weighted *merged* neighbour spectrum
(`entropy_composite`), plus neutral-loss entropy. Per candidate the
**best over the query's spectra** is kept (`np.maximum.reduceat`), then
averaged over spectra. `n_analogue_molecules`, `n_analogue_spectra`,
`analogue_best_tc` record the neighbourhood.

**Channel F — in-silico fragmentation (`insilico.py`, the repo
Fragmentor)**: for every (candidate, spectrum) pair, the Fragmentor
enumerates the candidate's theoretical fragment m/z for *that spectrum's
adduct* (`fragment_mzs`, cached per process); observed peaks are matched
within `--frag-ppm 10`. Features averaged over spectra:
`frag_matched_peaks`, `frag_explained_intensity`, `frag_mean_ppm`,
`n_theo_frags` (log1p), `frag_available`.

**Channel F′ — MetFrag-lite (`metfrag.py`, forum P1.1)**: deliberately
dumber than the Fragmentor — 1-bond cuts (weight 1.0) and 2-bond cuts
(weight 0.6) of the candidate, every fragment shifted by h ∈ −2..+3
hydrogens and ± one proton, deduplicated with max-weight merge, cached per
(SMILES, polarity). For each spectrum (`metfrag_match`, tolerance
`--mtf-tol 0.01` Da):

* `mtf_matched_peaks` — matched peak count;
* `mtf_explained_linear` — Σ (matched peak intensity × cut weight) —
  **linear** weighting, the forum's validated choice over sqrt;
* `mtf_rarity_explained` — same, but each peak is down-weighted by
  `1/(1 + #candidates in the shortlist that explain it)` — a peak every
  candidate explains is weak evidence;
* `mtf_gap_best` — the candidate's explained intensity **minus the best
  candidate's** — explained intensity *relative to the competition*, which
  is what ranking actually needs;
* `mtf_mean_ppm`, `mtf_available`.

**Channel P — learned spectrum→fingerprint (`fpchannel.py`, ON by default,
no flag)**: predicts the query's own 2048-bit Morgan r2 fingerprint from
its spectra and scores every candidate's *actual* bits against the
prediction — the CSI:FingerID idea, never comparing the query to reference
spectra. Features: `fp_ll` (per-bit log-likelihood **normalised by bit
priors** — "everything predicted 1" cannot win, because a bit that is
always on in the training molecules carries no information) and
`fp_tanimoto` (thresholded prediction vs candidate bits), plus
`fp_available`. Two implementations, auto-selected:

* **v1 (default, always available)** — a multi-label *linear* model
  (500 intensity-weighted m/z bins + adduct one-hot + scaled precursor →
  2048 sigmoid outputs, BCE) trained **inside every stage2 run** on rows
  visible under the mask in force, minus the sampled pseudo-truths' rows
  — leak-free by *construction* (asserted: R1/R2 extended to channel P).
  Not cached to disk: a stale cached model is exactly the kind of silent
  leak this channel guards against; it costs a couple of CPU minutes.
* **v2 (when the artifacts exist)** — the frozen de-novo transformer
  encoder (`train_denovo.py`) + fingerprint head (`train_fp_head.py`).
  Both trainers exclude **every molecule of every `dataset-creation/out`
  split by default** and record the excluded keys in the checkpoint
  `meta`; `fpchannel.build_fp_model` **refuses** an artifact that cannot
  prove it covered the split being evaluated (retrain it, or move it away
  to fall back to v1). `--include-eval-molecules` (trainer flag, not a
  stage2 flag) opts out for the final Kaggle model only.

**Channel F2 — neural in-silico spectrum prediction (`f2_bridge.py`,
zero-shot ICEBERG 2.1)**: instead of matching observed peaks against an
unweighted theoretical m/z list, a frozen public checkpoint predicts a full
*peak list with intensities* for each candidate, conditioned on adduct,
collision energy (bucketed to 5 eV — the cache key) and instrument; the
matched cosine (ppm-tolerant, `matched_cosine_one_vs_many_ppm`) between
observed and predicted spectrum becomes `f2_cosine_best` /
`f2_cosine_mean`, with `f2_available` marking coverage. ICEBERG lives in
its own venv (`extern/ms-pred/.venv`), so stage2 never imports it: the
bridge collects the unique `(smiles, adduct, ce)` triples a run needs,
predicts the missing ones in **one subprocess** (`f2_batch.py`,
one frozen model per worker) and keeps everything in
`out/cache/f2_predictions.parquet` — the expensive first pass over a fresh
pool is a one-time cost, later runs are instant. No env / no weights →
all `f2_*` features stay 0 and the fusion keeps working. Leakage: the
MassSpecGym checkpoint never saw any Enveda spectrum, so neither eval
truths nor training pseudo-truths can be memorised; a future self-trained
ICEBERG must be molecule-disjoint from every split before its predictions
may enter this cache.

Every feature is **averaged over the query's usable spectra**; a candidate
that is scoreable in principle but has no reference spectra anywhere
(the truth) gets 0 on all spectrum-comparison channels and lives entirely
on channels F/F′/P/M. The channel-P/F2 columns are appended by
`append_channel_blocks` — the *same* `channel_block` function for eval,
training and blind control (R4).

### A4. Model scoring and ranking

`predict(model, X, [n_candidates])` → `model_matrix` → `predict_proba[:,1]`.

`model_matrix` (the exact same transform at train and predict time, R4):

1. take the `FEATURES` columns minus `n_candidates`,
   minus the dropped set (source block by default + any `--drop-features`);
2. z-score every **continuous** feature **within the shortlist** — the
   model sees how a candidate compares to its competitors, not absolute
   values; constant columns become 0;
3. leave the raw flags (`frag_available`, `mtf_available`,
   `fp_available`, `f2_available`, `cand_is_external`, …) unscaled.

The classifier is the winner of {logistic, HGB} on internal validation
(see B6), possibly a `ModelEnsemble` of `--model-seeds` HGB fits whose
probabilities are averaged (forum P0.6).

**Ranking order** (`rank_order`): score descending, ties broken by
\|ppm error\| ascending, then candidate index. Optional
`--protect-top1 <channel>` (`protect_top1_order`, forum P1.5) pins that
channel's top-1 at rank 1 and inserts the fused pick at rank 2 — it can
never lose the protected channel's rank-1 hits.

### A5. What comes out per molecule

In `..._stage2_results.parquet`: the truth's rank / reciprocal rank /
bucket / panel ranks for every method (each channel baseline *and* the
fused model), `top25_keys`, `top25_smiles`, `top25_scores`, plus the query
context. In `..._stage2_features.parquet`: the full per-candidate feature
rows with labels. In `..._stage2_metrics.json`: the whole §6 protocol —
overall and per-Tc-band summaries, failure buckets, same-formula panel
metrics, `by_public_spectra`, paired bootstrap vs each baseline, class-share
scenarios.

The **blind control** (`--blind-control`, P0.5) re-runs A1–A4 with every
query's spectra swapped with a mass-adjacent molecule's (same windows,
same candidates, falsified spectral evidence). A channel that barely drops
was never reading the spectra; a run with candidate coverage 0 is flagged
vacuous.

---

## Part B — How the training works (`stage2.py`, clean mode)

The ranker is trained **on the fly** on pseudo-class-2 problems built the
same way the eval set was built. No eval molecule, spectrum or hidden row
is ever touched (asserted, not promised).

### B1. Draw the pseudo-truths (`sample_training_examples`)

From the molecules that are *visible* under the split (≥ 1 visible row)
and eligible as a truth (`n_eligible ≥ 1`, mass window), **minus every
eval molecule** (R1) — `--n-train 2000` of them by seeded permutation
(R5: one `default_rng(args.train_seed)`). Each drawn molecule:

* keeps `k` of its eligible rows as pseudo-query rows (`draw_n_spectra`,
  the same geometric draw as the real eval),
* has **all** its rows marked hidden (R2) — it will be reachable in the
  training pool only as a structure,
* and is appended to the training pool as an *addition*.

### B2. Build the training world

* `mask_train` = the split's visible rows minus every pseudo-truth's rows;
* `build_pool` over that mask, with the 2 000 pseudo-truths as additions;
* `formulas_for` on the pool (incremental disk cache — first run computes
  ~270 k formulas, later runs are instant) → `cand_n_same_formula` has
  real formulas to work with;
* neighbour lists + analogue library for the training pool **if** the
  channel is on — with two hard asserts:
  `R1: split-hidden row in the training library` and
  `R2: pseudo-query row in the training library`. The same two rules are
  asserted for channel P's training rows in `fpchannel.build_fp_model`
  (R1/R2 extend to every learned channel).

### B3. Build the training shortlists (`build_training_queries`)

For each pseudo-truth: query mass from its drawn rows, ± 8.5 ppm window in
the training pool, cleaned spectra. Examples whose truth fell outside
their own window or whose spectra clean to nothing are **dropped and
counted** (`dropped` dict), never padded — they carry no positive label
(R2b). A second filter drops any remaining shortlist whose truth is not a
candidate (`n_truth_outside_window`), so the `y` vector always has exactly
one positive per shortlist (asserted).

Typical clean-run numbers: 2 000 shortlists, ~190 k candidate rows,
~1 % positive.

### B4. Featurize — *identically to inference* (R4)

`featurize_queries` on the training shortlists — the same `featurize`,
the same tolerances, the same channels. This is the expensive step
(MetFrag-lite and the Fragmentor enumerate every unique candidate
structure, ~0.1–0.3 s each, cached per worker process); it is why a run
takes ~25 min for the smoke size and ~1 h for the full split.

### B5. Channels P and F2, trained/predicted under the same rules

Right after featurization (both sides), `run()` builds the run's channel
models and appends their columns — through the *same* `channel_block`
function, so R4 holds by construction:

* **channel P v1** trains *inside the run* on `mask_train` rows only —
  visible under the split (so no eval molecule's row, R1) and with the
  sampled pseudo-truths' rows hidden before training (R2). Both facts are
  **asserted**, not promised. The model is deliberately *not* cached to
  disk; it costs a couple of CPU minutes (≤ 200 k spectra, seeded
  subsample). In `--model-path` / `--leak-train` mode there are no
  pseudo-truths, so v1 trains on the plain split-visible rows — a
  documented train/serve skew of the on-the-fly channel; the frozen v2
  artifacts have no such skew.
* **channel P v2** (frozen `train_denovo.py` encoder + `train_fp_head.py`
  head) is used only when both artifacts exist *and* their checkpoint
  `meta.excluded_keys` proves that no molecule of the split being
  evaluated was trained on — otherwise the run **refuses** them (leak-free
  is the default; there is no override flag in stage2). The trainers
  exclude every `dataset-creation/out/*.parquet` key by default.
* **channel F2** needs no training (zero-shot public weights, no Enveda
  spectrum in their corpus). Predictions for the eval + training + blind
  shortlists are requested together (`f2_bridge.collect_requests`) and
  computed in one ms-pred subprocess, appended to the persistent
  `out/cache/f2_predictions.parquet`.

### B6. Fit, validate, pick the primary model

* `is_val`: ~10 % of the training molecules are held out (seeded);
* `model_matrix` over all shortlists at once (within-shortlist
  z-scoring is what makes train and predict consistent);
* `fit_models` on the non-validation rows:
  * **logistic** — `LogisticRegression(max_iter=5000, class_weight="balanced")`,
    wrapped in a 1-member `ModelEnsemble` for uniformity; its coefficients
    land in the metrics JSON as feature importances;
  * **HGB** — `HistGradientBoostingClassifier(max_depth=3, max_iter=300,
    early_stopping)` (plain boosting below 200 rows so tiny runs work),
    class-weight balanced with a `sample_weight` fallback, **fitted
    `--model-seeds` times with different `random_state`s and averaged**
    (P0.6: pin seeds, average);
* internal validation: MRR of each model on the held-out 10 %,
  `entropy_best` as the no-model baseline → the better model becomes
  `primary`;* HGB permutation importance on the validation slice (logistic already
  gave coefficients);
* the primary model is saved to `{stem}_stage2_model.joblib` with its
  feature list, its dropped-feature set and the training seed — reloading
  it later (`--model-path`) re-applies *its* drop set exactly
  (`replace=True`) and asserts the feature list matches.

### B7. The three run modes

| mode | training set | purpose |
|---|---|---|
| clean (default) | B1–B5 on the fly | the real experiment |
| `--leak-train` | the eval shortlists themselves, with the truth's eval spectra leaked into its own analogue library, analogue channel forced on | wiring self-test: MRR must come out ≈ 1, otherwise something in A1–A4 is broken; never used for real numbers |
| `--model-path M` | none | re-evaluate an existing model (e.g. a new split) without retraining |

### B8. Where state lives between runs (all incremental, disk-cached)

| cache | what it avoids |
|---|---|
| `dataset-creation/out/cache/` | molecule keys, row mapping, fingerprints |
| `max_class2_exp/out/cache/formulas.parquet` | RDKit formula enumeration (274 k) |
| `max_class2_exp/out/cache/row_sources_*.npz` | per-row ingest-lib/tims flags |
| `max_class2_exp/out/cache/library_*.parquet/.npz` | cleaned analogue-library spectra (keyed by rows + params + split info) |
| `max_class2_exp/out/cache/f2_predictions.parquet` | ICEBERG candidate predictions (keyed by smiles + adduct + 5 eV CE bucket; `ok=False` rows are cached as permanently unavailable) |
| `max_class2_exp/out/cache/fp_embed_cache.npz` | encoder embeddings for fp-head training (keyed by data + excluded-key hash + encoder) |
| channel-P v1 | **deliberately not cached** — retrained per run on the split-visible rows; a stale cached model would be a silent leak |
| in-process LRU | `fragment_mzs`, `metfrag_mzs`, composite-spectrum merges |

---

## The one-paragraph version

A difficulty-2 *query* is 1–16 cleaned timsTOF spectra of a molecule whose
every reference row was erased. From the median implied neutral mass, a
±8.5 ppm window over the mass-sorted candidate pool produces 50–300
structures — the truth among them as a spectrum-less structure. Every
candidate is scored by pure structure-side chemistry (Fragmentor +
MetFrag-lite cuts), a learned spectrum→fingerprint model (channel P,
leak-free by construction or by guarded artifacts), ICEBERG's predicted
spectrum (channel F2, zero-shot), mass context, and — only if enabled —
spectra of its nearest measured relatives; a within-shortlist z-scored
logistic/HGB fusion turns those into a ranking, and the evaluation asks
the only question that matters for class 2: can the model lift the
never-measured truth above its same-formula isomers, and does it still do
that when the spectra are falsified, when only structures without public
spectra are counted, and when the truth-shortcut flags are switched off.

## The channel-P v2 artifacts (optional; GPU for the encoder)

One command for both steps (`train_all.py`: preflight, encoder, head,
guard verification — the thing to run on the GPU box, README 7.1 has the
scp list):

```bash
python max_class2_exp/train_all.py --no-mlflow        # or --smoke first
```

The individual steps it drives:

```bash
# 1) de novo encoder (GPU, hours; leak-free by default: every
#    dataset-creation/out split molecule excluded + recorded in meta)
uv run python max_class2_exp/train_denovo.py --no-mlflow
# 2) fingerprint head on the frozen encoder (CPU-ok; same exclusion)
uv run python max_class2_exp/train_fp_head.py
```

Artifacts land in `max_class2_exp/out/denovo_v1.pt` +
`denovo_fp_head.pt`; the next stage2 run picks them up automatically (no
flag) — *if* their `meta.excluded_keys` cover the split being evaluated;
otherwise the run refuses them with instructions and falls back is *not*
silent: you retrain or move the files away to use v1. For the final
Kaggle model only, `--include-eval-molecules` on both trainers trains on
every molecule (and stage2 will then refuse the artifacts locally — by
design, since there is nothing left to evaluate honestly with them).
