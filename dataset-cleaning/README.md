# Dataset cleaning (`dataset-cleaning/`)

One-time spectrum curation for the Enveda CASMI 2026 repo. Produces cleaned
copies of the competition data that everything else — the `dataset-creation/`
eval sets, the training/scoring scripts, the kernel — reads instead of the
raw parquet files.

```bash
uv run python dataset-cleaning/clean_data.py             # once, after download_data.py
uv run python dataset-cleaning/clean_data.py --sample 10000 --no-save   # quick trial
```

Outputs (in `data/`, git-ignored):

| file | content |
|---|---|
| `train_clean.parquet` | all 2,539,608 train rows, cleaned peak arrays + `num_peaks_clean`, `n_removed`, `drop_reason` |
| `test_clean.parquet` | the test spectra, cleaned the same way |

## The 7-step cleaner

The logic lives in `src/enveda_casmi26/cleaning.py` (`SpectrumCleaner`,
`CleanConfig`) — the same module the Kaggle kernel imports, so local data and
submission-time cleaning can never drift apart. In order:

1. **Precursor cutoff** — drop peaks above `precursor_mz + 1.5 Da`
   (impossible fragments in MS2).
2. **Relative intensity floor** — drop peaks below 0.1 % of the base peak.
3. **Deisotoping** — remove C-13 companions ~1.003 Da above a kept peak
   (0.01 Da tolerance, ≤ 3 isotope orders, satellite never brighter than its
   anchor).
4. **Mass-defect junk filter** — below 350 Da no organic z=1 ion has an m/z
   fractional part in 0.44–0.56; peaks there are ghosts.
5. **Top-K per window** — keep the 6 most intense peaks per 50 Da window.
6. **Global cap** — at most 128 peaks per spectrum.
7. **Transform + renormalize** — sqrt intensities, rescale to base peak 1.0.

Spectra left with fewer than `--min-peaks` (5) peaks are **flagged**
(`drop_reason="too_few_peaks"`), never deleted. All thresholds are CLI flags;
see `--help`. The per-step removal counts are printed in the summary.

## Row alignment — the invariant everything relies on

Cleaning **never drops or reorders rows**. Row `i` of `train_clean.parquet`
is row `i` of `train.parquet`, so every row-id-based file (the
`dataset-creation/` split files, `spectrum_id = "s<row_id>"`) stays valid for
both. Because of this:

- `--sample` is for trial runs only and is never saved over `train_clean.parquet`.
- `scripts/make_split.py` asserts the alignment explicitly.

## How `dataset-creation/` picks it up

`dataset-creation/dataloader.py` resolves the train file on its own:

1. an explicitly passed `train_path` wins;
2. otherwise `data/train_clean.parquet` **if it exists** — every split you
   create then automatically trains and evaluates on cleaned spectra;
3. otherwise the raw `data/train.parquet` (and the dataloader logs that
   fallback).

So the flow is: run `clean_data.py` once → every eval set built with
`make_difficulty1.py` / `make_difficulty2.py` / `make_overall.py` is served
cleaned spectra by `load_split()` / `split.eval_spectra()` /
`split.iter_train()` with no extra code. The minimal test cleaning inside
`eval_spectra()` (drop peaks above precursor + 2 Da, rescale) is a no-op on
already-cleaned data, so the two paths compose safely.

Re-run `clean_data.py` only when the raw data changed or you changed cleaning
parameters.
