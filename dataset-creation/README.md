# Eval-set simulator (`dataset-creation/`)

Local evaluation sets for the Enveda CASMI 2026 challenge that look like the
hidden Kaggle test. You train on what is left of `data/train.parquet`,
predict the eval molecules, and score with MRR@25, exactly as on Kaggle.
The difference: here you know the answers, the difficulty of every molecule,
and why it is difficult.


**Contents**

1. [Quick start](#1-quick-start)
2. [Core ideas](#2-core-ideas)
3. [How a set is built](#3-how-a-set-is-built)
4. [The scripts and all their parameters](#4-the-scripts-and-all-their-parameters)
5. [Recipes](#5-recipes)
6. [Output files](#6-output-files)
7. [Using a split for training and evaluation](#7-using-a-split-for-training-and-evaluation)
8. [Validation](#8-validation)
9. [Cache and reproducibility](#9-cache-and-reproducibility)
10. [What the current sets look like](#10-what-the-current-sets-look-like)
11. [Known limits](#11-known-limits)
12. [File overview](#12-file-overview)

---

## 1. Quick start

Run everything from the repo root (`enveda_casmi/`). It needs
`data/train.parquet` and the project environment (`uv sync`); no extra
dependencies.

```bash
# once (~16 min): molecule keys, fingerprints, neighbour counts
uv run python dataset-creation/common.py

# the eval sets (10–15 s each)
uv run python dataset-creation/make_difficulty1.py                 # -> out/difficulty1.parquet + .png
uv run python dataset-creation/make_difficulty2.py                 # -> out/difficulty2.parquet + .png
uv run python dataset-creation/make_difficulty2.py --mock-library  # -> out/difficulty3.parquet + .png
uv run python dataset-creation/make_overall.py                     # -> out/overall.parquet + .png

# check that a set is leak-free and reproducible
uv run python dataset-creation/check_split.py dataset-creation/out/overall.parquet
```

Then in Python:

```python
import sys

sys.path.insert(0, "dataset-creation")
from dataloader import load_split

split = load_split("dataset-creation/out/overall.parquet")
test_df = split.eval_spectra()  # same columns as Kaggle's test.parquet
truth = split.truth  # answers + difficulty per molecule_id
```

If you skip the first command, the first make script builds the cache itself.

---

## 2. Core ideas

### 2.1 Never copy spectra, only reference them

An output file is small (~250 KB): one row per eval molecule with the **row
ids** of `train.parquet` that serve as its eval spectra, and the row ids
that must be hidden from training. The row id is the 0-based row position
in `train.parquet`. `dataloader.py` is the only place that combines a split
file with the real data. Nothing is duplicated, and every eval spectrum can
be traced back to its train row (`spectrum_id = "s<row_id>"`).

### 2.2 What is a "molecule"

The scorer compares the first block of the InChIKey after tautomer
canonicalisation. We use exactly that as the molecule **key**:
`TautomerEnumerator().Canonicalize(mol)` → `MolToInchiKey()[:14]`.
Hiding always works on this key, so a tautomer that is stored under another
`inchikey14` in train cannot leak the answer. Train has 275,810 distinct
`inchikey14` values but only 274,195 keys; 1,615 tautomer duplicates are
merged. Use `common.molecule_key(smiles)` to compute the key of your own
predictions.

### 2.3 Which spectra can be eval spectra ("eligible" rows)

The hidden test is 100 % timsTOF, uses ten adducts and was cleaned in a
known way. A train row may become an eval spectrum only if:

| rule | why |
|---|---|
| `instrument_type == "timsTOF"` | the test instrument |
| adduct is one of `[M+H]+ [M+NH4]+ [M-H2O+H]+ [M-2H2O+H]+ [M+Na]+ [M+K]+ [M-H]- [M-H2O-H]- [M+CH2O2-H]- [M+Cl]-` | the test adducts |
| `collision_energy_ev` is present | the test always has it |
| `base_peak_intensity >= 1000` | the test dropped weaker spectra |
| `abs(precursor_error_ppm) <= --ppm-tol` (10) | the test dropped spectra whose precursor did not match the structure |
| `num_peaks >= 1` | |

A molecule can be an eval molecule if it has at least one eligible row and an
exact mass between 157 and 1159 Da (the test range). About 180,000 molecules
qualify. Almost all of them come from `enveda-180`, because that is where the
timsTOF data is (see [§11](#11-known-limits)).

When the eval spectra are built, the test cleaning is applied: peaks above
precursor m/z + 2 Da are removed and intensities are rescaled to a maximum of 1.0.

### 2.4 Three difficulty levels

The hidden test has three novelty classes. Each level imitates one of them:

| level | imitates | eval molecule in train? | similar molecules in train? | in the candidate library? |
|---|---|---|---|---|
| **1** | class 1: public spectra exist | **yes**, but in a different setting (see below) | yes | yes |
| **2** | class 2: structure known (PubChem/COCONUT), no spectra | **no**, all its rows are hidden | yes, except for `--neighbour-removed-frac` of the molecules | **yes**, must be added (see [§7.3](#73-rules-for-a-fair-evaluation)) |
| **3** | class 3: novel structure | no | as level 2 | **no**, must be removed |
| **overall** | the hidden test | 20 % level 1, 60 % level 2, 20 % level 3 (`--share`) | | |

**Level 1, the three "change" types.** The eval molecule keeps some train
spectra, but they differ from its eval spectra in exactly one way:

| `change` | what stays visible in train | what it simulates |
|---|---|---|
| `instrument` | only the molecule's **non-timsTOF** spectra (Orbitrap, Q-TOF, …) in the eval adduct; all its timsTOF spectra are hidden | the typical class-1 case: the molecule is in public libraries, but measured on other instruments. Default 60 % of level 1 |
| `adduct` | only its timsTOF spectra in **other adducts**, e.g. `[M+H]+` visible, eval on `[M+Na]+` | library spectra exist, but for another ion form. Default 20 % |
| `ce` | only its timsTOF spectra in the eval adduct at **other collision energies** | the same ion is known, but fragmented at another energy. Default 20 % |

Collision energies are not matched for `instrument`, because their units
differ between instruments (NCE, eV, V).

**Level 2 and 3, neighbourhood removal.** For a share of the molecules
(`--neighbour-removed-frac`, default 10 %), every molecule with Morgan
Tanimoto ≥ `--neighbour-tc` (default 0.5) to the eval molecule is hidden as
well. This creates the case "nothing similar in train". Without it, most
eval molecules still have a close analogue in train, because `enveda-180`
consists of combinatorial families.

### 2.5 Diversity

No two eval molecules in one file may have Tanimoto ≥ `--diversity-tc`
(default 0.6) to each other, across all levels. And a molecule's chance to be
picked is divided by its number of neighbours at that threshold, so a family
of 50 near-identical compounds is about as likely to contribute a molecule as
one isolated compound.

All similarities in this folder are Morgan fingerprints (radius 2,
2048 bits) and Tanimoto. There is no clustering step.

---

## 3. How a set is built

```
train.parquet ──► cache (once)                       ──► make_*.py (seconds)                          ──► split file
                  • key per row (tautomer InChIKey14)     1. candidate molecules                          out/<name>.parquet
                  • molecule table (mass, ion mode)       2. weighted, diverse sampling                   out/<name>.png
                  • Morgan fingerprints                   3. eval spectra per molecule
                  • neighbour counts                      4. hidden rows (+ neighbourhood removal)
                                                          5. similarity to nearest visible molecule
                                                          6. sanity report, plots
```

1. **Candidates.** Molecules with ≥ 1 eligible row and mass 157–1159 Da.
   Level 1 also needs the right train rows for its change type (2,474
   molecules qualify for `instrument`, ~69,000 for `adduct`, ~178,000 for
   `ce`).
2. **Sampling.** Weighted sampling without replacement (Efraimidis–Spirakis
   keys). The weights combine:
   - 1 / number of neighbours (diversity balance, [§2.5](#25-diversity));
   - mass-bin weights toward the test-like target shares: 157–250 Da 15 %,
     250–350 Da 35 %, 350–500 Da 30 %, 500–800 Da 15 %, 800–1160 Da 5 %;
   - an ion-mode weight toward `--ion-pos-share` (70 % positive).

   Mass and ion weights are fitted by iterative proportional fitting. The
   sampler walks the weighted order and skips any molecule that is too similar
   to one already picked or that was already used by another level.
3. **Eval spectra.** For each molecule, `k` is drawn from a geometric
   distribution (p = 0.25, so a median of 3, at most 16). It is then capped at
   the number of candidate rows, and `k` rows are drawn at random from them.
4. **Hiding.** Level 1: every row of the molecule except the visible rows of
   its change type. Level 2/3: every row of the molecule, plus all rows of its
   neighbours if neighbourhood removal applies to it.
5. **Annotation.** For each eval molecule, the most similar molecule that is
   still visible after hiding is found (`max_visible_tc`,
   `nearest_visible_key`).
6. **Report.** A sanity report compares the realised distributions with the
   test description. It only warns and never changes the sample. A
   diagnostic figure is written next to the file.

**Overall set order.** Level 2 and 3 are drawn first, because they decide
which neighbours are hidden. Level 1 is drawn last and never from a hidden
neighbour, otherwise its visible rows would disappear. The three molecule
sets are disjoint.

Shortfalls (not enough qualifying molecules) are printed as `SHORTFALL` and
stored in the file. They are never padded.

---

## 4. The scripts and all their parameters

Every script prints its options with `--help`.

### 4.1 Shared by `make_difficulty1.py`, `make_difficulty2.py`, `make_overall.py`

**Size and sampling**

| flag | default | meaning |
|---|---|---|
| `--n-molecules` | `1000` | Number of eval molecules. The test has ~400; 1000 lowers the noise of the score (standard error about ±0.013 instead of ±0.02). |
| `--diversity-tc` | `0.6` | No two eval molecules may have Tanimoto ≥ this. Lower means a more diverse set but fewer usable candidates. A new value triggers a one-off neighbour count (~10 min, cached). |
| `--no-diversity-balance` | off | Weight every molecule the same instead of 1 / number of neighbours. Large `enveda-180` families then dominate. |
| `--ion-pos-share` | `0.7` | Target share of positive-mode molecules. A sampling weight, not a hard quota. |
| `--seed` | `2026` | The only source of randomness. Same seed and same cache give an identical file; another seed gives an independent draw. |

**Which rows can be eval spectra**

| flag | default | meaning |
|---|---|---|
| `--ppm-tol` | `10` | Maximum \|precursor error\| (ppm) of an eval spectrum. The other eligibility rules ([§2.3](#23-which-spectra-can-be-eval-spectra-eligible-rows)) are fixed. |

**Paths and cache**

| flag | default | meaning |
|---|---|---|
| `--train` | `data/train.parquet` | Train file. |
| `--cache-dir` | `dataset-creation/out/cache` | Cache location. |
| `--rebuild-cache` | off | Recompute keys, fingerprints and neighbour counts. Only needed if the cache is broken; a changed train file is detected automatically. |
| `--out` | see the script | Output parquet. The figure is written as the same path with `.png`. |

### 4.2 `make_difficulty1.py`: same molecule, different setting

| flag | default | meaning |
|---|---|---|
| `--change-mix` | `instrument=0.6,adduct=0.2,ce=0.2` | Share of molecules per change type ([§2.4](#24-three-difficulty-levels)). Normalised to 1. A single type is allowed, e.g. `instrument=1` (at most 2,474 molecules qualify). |
| `--out` | `out/difficulty1.parquet` | |

### 4.3 `make_difficulty2.py`: unseen molecules (difficulty 2, or 3)

| flag | default | meaning |
|---|---|---|
| `--neighbour-removed-frac` | `0.10` | Share of eval molecules whose neighbours are hidden as well. Must lie in [0.05, 0.20]. |
| `--neighbour-tc` | `0.5` | Tanimoto at or above which a molecule counts as a neighbour. Typical neighbourhood size of a candidate: 0.4 → median 44 (max ~600), 0.5 → median 8 (max ~120), 0.6 → median 2, 0.7 → median 0. Lower is harder. |
| `--max-removed-neighbours` | `200` | Molecules with more neighbours than this are not used for removal, so one pick cannot hide a large part of train. |
| `--mock-library` | off | Difficulty 3: same selection, but every molecule gets `exclude_from_library = True`. |
| `--out` | `out/difficulty2.parquet`, or `out/difficulty3.parquet` with `--mock-library` | |

With the same seed, difficulty 2 and difficulty 3 contain the same molecules
and differ only in the label and the library flag. Use another `--seed` if you
want different molecules.

### 4.4 `make_overall.py`: the mixed set

| flag | default | meaning |
|---|---|---|
| `--share` | `0.2,0.6,0.2` | Shares of difficulty 1, 2 and 3. |
| `--change-mix` | `instrument=0.6,adduct=0.2,ce=0.2` | For the difficulty-1 part ([§4.2](#42-make_difficulty1py-same-molecule-different-setting)). |
| `--neighbour-removed-frac`, `--neighbour-tc`, `--max-removed-neighbours` | `0.10`, `0.5`, `200` | For the difficulty-2 and -3 parts ([§4.3](#43-make_difficulty2py-unseen-molecules-difficulty-2-or-3)). |
| `--out` | `out/overall.parquet` | |

### 4.5 `common.py`: build the cache

Not needed in daily use, because the make scripts build missing cache parts
themselves. Useful to build the cache ahead of time (e.g. on a fresh clone).

| flag | default | meaning |
|---|---|---|
| `--train`, `--cache-dir`, `--rebuild-cache` | | as in [§4.1](#41-shared-by-make_difficulty1py-make_difficulty2py-make_overallpy) |
| `--diversity-tc` | `0.6` | Which neighbour counts to precompute. |
| `--workers` | CPUs − 2 | Parallel processes. |

### 4.6 `check_split.py <file>`: validate a set

| argument | meaning |
|---|---|
| `<file>` | The split to check. Train path, cache and all parameters are read from the file's metadata. |
| `--no-rerun` | Skip the reproducibility check (the rebuild with the same seed). |

Exit code 0 means all checks passed and 1 means at least one failed. See [§8](#8-validation).

### 4.7 `plots.py <file>`: redraw the figure

| argument | meaning |
|---|---|
| `<file>` | The split. |
| `--out` | PNG path, default `<file>.png`. |

### 4.8 `dataloader.py`: Python API

See [§7](#7-using-a-split-for-training-and-evaluation).

---

## 5. Recipes

**The test-like default set**
```bash
uv run python dataset-creation/make_overall.py
```

**Several independent draws** (compare methods on all of them, paired)
```bash
for s in 2026 2027 2028 2029 2030; do
  uv run python dataset-creation/make_overall.py --seed $s --out dataset-creation/out/overall_s$s.parquet
done
```

**A Kaggle-sized set** (to look at the scale of the test, not for decisions)
```bash
uv run python dataset-creation/make_overall.py --n-molecules 400 --out dataset-creation/out/overall_400.parquet
```

**Only the realistic class-1 case**
```bash
uv run python dataset-creation/make_difficulty1.py --change-mix instrument=1 --n-molecules 1000
```

**Harder unseen molecules** (more and larger neighbourhoods hidden)
```bash
uv run python dataset-creation/make_difficulty2.py --neighbour-removed-frac 0.2 --neighbour-tc 0.4 \
    --out dataset-creation/out/difficulty2_hard.parquet
```

**A different mix**, e.g. half known and half unknown
```bash
uv run python dataset-creation/make_overall.py --share 0.5,0.4,0.1 --out dataset-creation/out/overall_easy.parquet
```

**After `train.parquet` changed**: nothing to do. The cache detects the new
file (size and modification time) and rebuilds; old split files then refuse to
load (cache hash mismatch) and must be regenerated.

Give every variant its own `--out`, otherwise it overwrites the default file.

---

## 6. Output files

`dataset-creation/out/` is git-ignored.

| file | content |
|---|---|
| `<name>.parquet` | the split: one row per eval molecule |
| `<name>.png` | the diagnostic figure |
| `cache/` | shared cache ([§9](#9-cache-and-reproducibility)) |

### 6.1 Columns of a split file

| column | type | meaning |
|---|---|---|
| `molecule_id` | str | `d{difficulty}_{6 hex}`, unique. Used as `molecule_id` in the eval spectra. |
| `key` | str | tautomer-canonical InChIKey14: **the answer** the scorer compares |
| `smiles` | str | representative SMILES of the answer (the most frequent one in train) |
| `difficulty` | int8 | 1, 2 or 3 |
| `change` | str / null | `instrument`, `adduct` or `ce` (level 1 only) |
| `neighbours_removed` | bool | similar molecules were hidden as well |
| `removed_keys` | list<str> | keys of those neighbours; all of their rows are hidden |
| `exclude_from_library` | bool | level 3: the structure must not be in any candidate pool |
| `eval_row_ids` | list<int64> | train rows used as eval spectra |
| `hidden_row_ids` | list<int64> | all hidden rows of this molecule (includes the eval rows; neighbour rows are not listed here) |
| `n_eval_spectra` | int16 | number of eval spectra |
| `exact_mass` | float64 | monoisotopic mass of the neutral molecule |
| `ion_mode` | str | `positive` or `negative` (majority over eligible rows) |
| `max_visible_tc` | float32 | highest Tanimoto to any molecule still visible in train after hiding. 1.0 for level 1 (the molecule itself stays). Use it to report scores per similarity band. |
| `nearest_visible_key` | str | key of that most similar visible molecule |

### 6.2 Metadata

The parquet key-value metadata `casmi_split` holds JSON with: the script
name, all parameters, the seed, the cache hash, the train file's size and
modification time, the full sanity report, the shortfalls and a hash of the table
content. Read it with:

```python
import common  # with dataset-creation on sys.path

info = common.read_split_info("dataset-creation/out/overall.parquet")
info["params"], info["sanity_report"], info["shortfalls"]
```

### 6.3 The sanity report and the figure

Each make script prints realised values next to the test description, e.g.

```
molecules 1000 (test ~400)    spectra 2676 | ~3750 at the test rate of 1500/400
spectra/molecule {'min': 1.0, 'median': 3.0, 'max': 9.0} | min 1, median 3, max 16
mass {'min': 169.07, 'median': 332.14, 'max': 954.1} | min 157, median 348, max 1159
mass bins 157-250: 0.16/0.15, 250-350: 0.46/0.35, 350-500: 0.35/0.30, 500-800: 0.02/0.15, 800-1160: 0.00/0.05
...
WARNING: mass bin 500-800: 0.02 vs target 0.15
```

Warnings are information, not errors ([§11](#11-known-limits) explains the
usual ones). The figure has 11 panels:

| panel | shows |
|---|---|
| Mass distribution | eval molecules against the eligible pool, eval and test medians |
| Mass-bin shares vs target | realised share per mass bin next to the target |
| Spectra per molecule | realised counts next to the draw distribution |
| Adducts (spectra) | all ten test adducts, zeros included; positive-mode share |
| Collision energy (single) | eV values of single-energy spectra |
| Single vs merged CE | number of energies per spectrum |
| Peaks per spectrum | before removing peaks above the precursor |
| Base-peak intensity | with the test's 1,000-count cut-off |
| Composition | molecules per difficulty, change type and neighbourhood removal |
| Similarity to nearest visible molecule | `max_visible_tc` per difficulty, with the removal threshold |
| Neighbourhood removal | neighbours hidden per molecule |

---

## 7. Using a split for training and evaluation

### 7.1 Loading

```python
import sys

sys.path.insert(0, "dataset-creation")
from dataloader import load_split

split = load_split("dataset-creation/out/overall.parquet")
# optional: load_split(path, train_path="data/train.parquet", cache_dir="dataset-creation/out/cache")
```

`load_split` takes about 0.1 s. It refuses a split that was built against
another cache (see [§9](#9-cache-and-reproducibility)).

| attribute / method | returns |
|---|---|
| `split.eval_spectra()` | DataFrame with exactly the `test.parquet` columns (`molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities, base_peak_intensity, adduct, ionization_mode, instrument_type, precursor_mz, collision_energy_ev, collision_energy_orig, collision_energy_orig_units`), test cleaning applied. Reads only the row groups that hold eval rows (~5 s for 2,700 spectra). Spectra left without peaks are dropped and logged. |
| `split.iter_train(columns)` | generator of `pyarrow.Table`s with the **visible** train rows, one row group (~131k rows) at a time, plus a `row_id` column. Use it for anything memory-heavy (peak arrays). |
| `split.train_table(columns)` | the same as one table |
| `split.train_mask` | bool array, one entry per train row; `True` = may be used |
| `split.eval_row_ids` | all eval row ids |
| `split.truth` | the split file as a DataFrame ([§6.1](#61-columns-of-a-split-file)) |
| `split.library_additions()` | `key, smiles` of the level-2 molecules, to add to a candidate library built from train |
| `split.excluded_library_keys` | set of level-3 keys, to remove from every candidate pool |
| `split.info` | the metadata ([§6.2](#62-metadata)) |

### 7.2 A full evaluation loop

```python
import sys

sys.path.insert(0, "dataset-creation")
import pandas as pd
import common
from dataloader import load_split

split = load_split("dataset-creation/out/overall.parquet")
test_df = split.eval_spectra()

# 1. train only on visible rows
for batch in split.iter_train(
    ["normalized_smiles", "adduct", "ms2_mzs", "ms2_normalized_intensities"]
):
    ...  # fit your model

# 2. candidate library: visible train structures + level-2 structures
#    + your external pools (PubChem, COCONUT), minus level-3 keys
library = (
    split.train_table(["normalized_smiles"])
    .to_pandas()["normalized_smiles"]
    .drop_duplicates()
)
library = pd.concat([library, split.library_additions()["smiles"]], ignore_index=True)
# external = ...; external = external[~external_keys.isin(split.excluded_library_keys)]

# 3. predict: molecule_id -> up to 25 ranked SMILES
predictions = my_model.predict(test_df, library)  # dict[str, list[str]]


# 4. score with the scorer's key
def reciprocal_ranks(predictions, truth):
    rr = []
    for mid, key in zip(truth["molecule_id"], truth["key"]):
        ranked = [common.molecule_key(s) for s in predictions.get(mid, [])[:25]]
        rr.append(1.0 / (ranked.index(key) + 1) if key in ranked else 0.0)
    return pd.Series(rr, index=truth.index)


rr = reciprocal_ranks(predictions, split.truth)
print("MRR@25", rr.mean())

# 5. break it down
t = split.truth.assign(rr=rr)
print(t.groupby("difficulty").rr.mean())
print(t.groupby("change").rr.mean())
print(
    t.groupby(
        pd.cut(t.max_visible_tc, [0, 0.4, 0.5, 0.6, 0.8, 1.0]), observed=False
    ).rr.mean()
)
```

### 7.3 Rules for a fair evaluation

1. **Train only on rows where `train_mask` is True** (`iter_train` and
   `train_table` do this for you). This includes every model, statistic and
   lookup table built from train: fragment priors, CE priors, decoys, and so on.
2. **Level 2 needs its structure in the candidate pool.** These molecules
   imitate "known structure, no spectra". A library built from visible train
   rows alone does not contain them, so add `split.library_additions()` (or
   make sure your external pool, e.g. PubChem, contains them). Otherwise
   level 2 silently becomes level 3.
3. **Level 3 must be removed from every pool**, external pools included:
   drop `split.excluded_library_keys` after computing the keys with
   `common.molecule_key`.
4. **Compare by key, not by SMILES**: `common.molecule_key(smiles)` gives the
   scorer's key.
5. **Do not tune on the truth.** Use one seed for development and other
   seeds for the final comparison.

---

## 8. Validation

`check_split.py` loads a split exactly as the dataloader does and checks:

| # | check |
|---|---|
| 1 | no eval row is visible |
| – | every eval row belongs to its molecule's key |
| 2 | no level-2/3 key has a visible row |
| 3 | every level-1 molecule keeps ≥ 1 visible row, and every visible row differs in the declared dimension (`instrument`: no visible timsTOF row; `adduct`: no visible row in an eval adduct; `ce`: no visible row in an eval adduct with an eval CE) |
| 4a | neighbour-removed molecules have no visible molecule with Tanimoto ≥ `--neighbour-tc` (recomputed from the fingerprints) |
| 4b | hidden neighbours have no visible row |
| 4c | the stored `max_visible_tc` matches a recomputation |
| 5 | all eval rows are eligible timsTOF rows in the ten adducts |
| 6 | `molecule_id`, keys and eval rows are unique (the levels are disjoint) |
| 7 | no two eval molecules have Tanimoto ≥ `--diversity-tc` |
| 8 | rebuilding with the same parameters gives the same table content |
| 9 | the sanity report is printed |

It prints `[PASS]` or `[FAIL]` for every check and exits with 1 if any check fails.

---

## 9. Cache and reproducibility

`dataset-creation/out/cache/` (git-ignored, ~95 MB):

| file | content | build time |
|---|---|---|
| `molecules.parquet` | one row per key: `key, smiles, exact_mass, n_rows, n_eligible, has_tims, has_other, ion_mode` | ~6 min (tautomer canonicalisation of 277k SMILES) |
| `row_key.npy` | for every train row, the index of its molecule in `molecules.parquet` | (same step) |
| `fps.npy` | Morgan fingerprints (r2, 2048 bits) of every molecule, packed | ~5 s |
| `neighbour_counts_tc<x>.npy` | neighbours per molecule at diversity threshold `x` | ~10 min per threshold, on first use |

The `.hash` files tie the derived files to `molecules.parquet`.

- The cache is rebuilt automatically when the size or modification time of
  `train.parquet` changes.
- Every split stores the sha256 of `molecules.parquet`. `load_split` refuses a
  split built against another cache, because row ids and keys might not
  match any more.
- Same seed + same parameters + same cache → identical table
  (`check_split.py` check 8). Only the parquet bytes may differ.

To keep two caches side by side (e.g. two train versions), pass different
`--cache-dir` values to the make scripts and `cache_dir=` to `load_split`.

---

## 10. What the current sets look like

Default parameters, seed 2026:

| | difficulty 1 | difficulty 2 / 3 | overall | test |
|---|---|---|---|---|
| molecules | 1000 | 1000 | 1000 (200/600/200) | ~400 |
| spectra | 2,155 | 2,751 | 2,676 | ~1,500 (~3.75 per molecule) |
| spectra per molecule, median | 2 | 3 | 3 | 3 |
| mass, median (Da) | 338 | 335 | 332 | 348 |
| share 500–1160 Da | 0.03 | 0.02 | 0.02 | ~0.20 (target) |
| `max_visible_tc`, median | 1.0 | 0.58 | 0.61 | – |
| share with nothing ≥ 0.5 visible | 0 % | 16 % | 13 % | – |
| max Tanimoto between eval molecules | 0.59 | 0.58 | 0.59 | – |
| build time | ~12 s | ~9 s | ~15 s | |

All four pass `check_split.py`.

---

## 11. Known limits

Details and ideas in [POTENTIAL_IMPROVEMENTS.md](POTENTIAL_IMPROVEMENTS.md).

- **Chemistry.** Eval spectra must be timsTOF, and train's timsTOF data is
  almost entirely `enveda-180` (synthetic, drug-like screening compounds) plus
  the 250 natural products of `enveda-np-examples`. The test is
  natural-product-like. Report your scores per source library (the
  `ingest_lib` of the eval rows) to see the gap.
- **Heavy molecules.** Only ~46 eligible molecules weigh more than 500 Da, so
  the 500–1160 Da target shares cannot be reached (warning in every report).
- **Spectra per molecule** are capped by what a molecule has in train, so
  the average stays below the test's ~3.75, especially in level 1.
- **Analogues.** Even unseen molecules usually keep a relative at Tanimoto
  ~0.6 in train. Use `max_visible_tc` to see how your method depends on it.
- **Fingerprint collisions.** A few molecules have Tanimoto 1.0 to a
  different molecule (homologues that Morgan radius 2 cannot tell apart).
- **Level 3 is a flag.** The dataset cannot remove structures from PubChem or
  COCONUT; your evaluation code must do it (rule 3 in [§7.3](#73-rules-for-a-fair-evaluation)).

---

## 12. File overview

| file | role |
|---|---|
| `common.py` | data loading, eligibility, molecule keys, cache, fingerprints and similarity, sampling, output schema, train mask, sanity report; CLI to build the cache |
| `make_difficulty1.py` | level 1 (`select_difficulty1`, reused by `make_overall.py`) |
| `make_difficulty2.py` | level 2 and 3 (`select_difficulty2`, reused by `make_overall.py`) |
| `make_overall.py` | the 20/60/20 mix |
| `dataloader.py` | `load_split` and `Split`: the only place where splits meet `train.parquet` |
| `check_split.py` | validation |
| `plots.py` | the diagnostic figure |
| `PLAN.md` | the original specification (since then: Butina clustering replaced by similarity, defaults changed) |
| `POTENTIAL_IMPROVEMENTS.md` | known gaps and ideas |
| `out/` | outputs and cache (git-ignored) |
