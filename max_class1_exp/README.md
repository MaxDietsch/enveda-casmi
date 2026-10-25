# Class-1 experiment (`max_class1_exp/`)

How well can we identify **class-1 molecules**, i.e. molecules that already have
reference spectra in train, but measured in a different setting (another
instrument, adduct or collision energy)? This folder holds the experiment.
**Stage 1** (library search) is implemented and tested. Stages 2 and 3 are
planned and described below.

The eval data comes from the eval-set simulator: by default
`dataset-creation/out/difficulty1.parquet` (see `dataset-creation/README.md`).
Every eval molecule there has `change` = `instrument`, `adduct` or `ce`, and
only the rows that differ in that dimension stay visible in train.

---

## 1. Quick start

From the repo root (`enveda_casmi/`):

```bash
# 1. unit tests of the maths (no data, ~2 s)
uv run python max_class1_exp/test_stage1.py

# 2. end-to-end self-test: eval spectra deliberately put into the library -> MRR must be ~1
uv run python max_class1_exp/stage1.py --leak-eval --limit 100

# 3. the real evaluation (~1 min reading the library the first time, then cached)
uv run python max_class1_exp/stage1.py
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

## 7. Next stages (not implemented yet)

### Stage 2: learned re-ranker over the top ~50 candidates

Features per candidate, then a ranking model. Start with the repo's logistic
`CandidateScorer`; `HistGradientBoostingClassifier` from scikit-learn is better
and needs no new dependency.

- best and mean entropy similarity, cosine, and **modified cosine** (peaks
  shifted by the precursor mass difference, which helps across adducts);
- **neutral-loss similarity** (precursor − fragment), which often survives an
  adduct change;
- matched peaks and explained intensity share;
- mass error (ppm);
- flags: same adduct / polarity / instrument type in the library, CE
  difference;
- how many source libraries contain the molecule (a "well-known compound" prior).

Training data: difficulty-1 splits with other seeds, built only from the
visible rows, so the re-ranker learns exactly the cross-setting cases.

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

## 8. Files

| file | content |
|---|---|
| `spectra.py` | cleaning, entropy weighting, entropy and cosine similarity (one query against many), merged spectra |
| `library.py` | query mass, candidate index (mass window over visible molecules), parallel library loading and cache |
| `stage1.py` | the evaluation CLI |
| `test_stage1.py` | correctness tests |
| `out/` | results and library cache (git-ignored) |
