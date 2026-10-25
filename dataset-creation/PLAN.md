# Task: build the eval-dataset creator in `enveda_casmi/dataset-creation/`

You are working in the repo `enveda_casmi/` (Kaggle "Enveda CASMI 2026": predict
up to 25 ranked SMILES per molecule from MS/MS spectra, scored by MRR@25 on the
InChIKey first block). Run everything with `uv run python ...` from the repo
root. RDKit 2026.3.6, pandas, pyarrow, numpy and scikit-learn are already
installed. Read `competition-pages/data-description.md` (one level up) and
`src/enveda_casmi26/pipeline.py` before you start.

The goal is a **test-set simulator**. It creates local eval sets that look
like the hidden Kaggle test, at three difficulty levels plus an overall mix.
The scripts do **not** copy spectra. They write small files that reference
rows of `data/train.parquet`, and a dataloader combines those files with the
real data at load time.


---

## 1. Facts about the data (already checked)

- `data/train.parquet`: 2,539,608 rows, 21 row groups, ~3 GB, 18 columns:
  `ingest_lib, normalized_smiles, inchikey, inchikey14, molecular_formula,
  ionization_mode, instrument_type, adduct, adduct_orig, precursor_mz,
  precursor_error_ppm, ms2_mzs, ms2_normalized_intensities, num_peaks,
  base_peak_intensity, collision_energy_ev (list<double>),
  collision_energy_orig, collision_energy_orig_units`.
- There is no spectrum ID column. **The 0-based row index in file order is the
  spectrum ID** (`row_id`).
- `instrument_type` is free text. Exactly `"timsTOF"` marks 1,154,969 rows,
  almost all from `enveda-180` (synthetic drug-like compounds), plus
  `enveda-np-examples`.
- Units of `collision_energy_orig_units` outside timsTOF: mostly `NCE`, then
  `unknown`, `eV` and `V`. CE is therefore **not comparable across
  instruments**.
- With the eligibility rules in §3.2 (keyed on the `inchikey14` column), I
  measured:
  - 823,142 eligible rows covering 179,943 molecules, median 4 eligible
    spectra per molecule;
  - **only 2,508 "bridge" molecules**, meaning molecules with eligible timsTOF
    spectra plus non-timsTOF spectra;
  - 68,996 molecules with eligible timsTOF spectra in ≥ 2 adducts;
  - 177,866 molecules with ≥ 2 distinct CE values within one adduct.
  - Total distinct `inchikey14` values: 275,810.
- `.gitignore` ignores `*.parquet` and `*.npy`, so outputs and caches are not
  committed. That is intended.
- Helpers you may reuse: `enveda_casmi26.pipeline.neutral_mass`, and
  `precursor_error_ppm`, which is already present per row.

## 2. Target: what the hidden test looks like

- ~1,500 spectra of ~400 molecules, 1–16 spectra per molecule (median 3).
- 100% `timsTOF`.
- Neutral monoisotopic mass 157–1,159 Da (median 348).
- Adducts, all ten: `[M+H]+, [M+NH4]+, [M-H2O+H]+, [M-2H2O+H]+, [M+Na]+,
  [M+K]+, [M-H]-, [M-H2O-H]-, [M+CH2O2-H]-, [M+Cl]-`.
- Cleaning applied to test spectra:
  - precursor consistent with the structure;
  - base peak ≥ 1000 raw counts;
  - peaks above precursor + 2 Da removed;
  - intensities normalised so the max is 1.0.
- Novelty classes are hidden. Our levels approximate them:
  - **Difficulty 1:** the molecule has spectra in train, but they differ from
    the eval spectra in one dimension (instrument, adduct or CE).
  - **Difficulty 2:** the molecule has no spectra in train. For some
    molecules, the whole structural cluster is gone too.
  - **Difficulty 3:** like 2, plus the structure is excluded from the
    candidate library (mock of "not in PubChem").
  - **Overall:** 20% difficulty 1, 60% difficulty 2, 20% difficulty 3.

Main principles:
- Diversify eval molecules through **structural clustering**.
- Spectrum count, mass and CE distributions should come out right by
  themselves. They are only **sanity-checked** in a report, never forced.
- **Never** put an eval spectrum in the visible training data.

## 3. Files to create

```
dataset-creation/
  README.md              usage, parameters, output schema, how to load
  common.py              shared logic (§3.1–§3.5)
  make_difficulty1.py    CLI, §4
  make_difficulty2.py    CLI, §5 (difficulty 3 = flag)
  make_overall.py        CLI, §6
  dataloader.py          §7
  check_split.py         CLI, validates an output file (§8)
  out/                   outputs (git-ignored)
  out/cache/             cached molecule table, row keys, clusters
```
Scripts must run as `uv run python dataset-creation/make_difficulty2.py ...`.
The directory name has a hyphen, so it is not importable as a package. Each
script adds its own directory to `sys.path` and imports `common`, following
the `sys.path.insert` pattern in `scripts/make_split.py`. It also adds
`src/` so `enveda_casmi26` imports.

### 3.1 Loading the metadata
`load_meta(train_path)` reads only these columns: `normalized_smiles,
inchikey14, instrument_type, adduct, ionization_mode, precursor_mz,
precursor_error_ppm, base_peak_intensity, collision_energy_ev,
collision_energy_orig_units, num_peaks`. Do **not** read the peak arrays here.
Add `row_id = np.arange(n)`. Convert `collision_energy_ev` to a hashable
tuple column `ce` (None stays None).

### 3.2 Eval-eligible rows
A row is eval-eligible when all of these hold:
1. `instrument_type == "timsTOF"`;
2. `adduct` is one of the ten test adducts;
3. `collision_energy_ev` is not null;
4. `base_peak_intensity >= 1000`;
5. `abs(precursor_error_ppm) <= 10` (parameter `--ppm-tol`);
6. `num_peaks >= 1`. Also check, in the dataloader when the eval spectra are
   built, that at least one peak ≤ precursor + 2 Da remains. Drop the
   spectrum there if none does, and log it. Do not read the peaks in the
   builder.

### 3.3 Molecule key (cached)
- The scorer compares tautomer-canonical InChIKey14. Per distinct
  `normalized_smiles` (about 276k), compute
  `rdMolStandardize.TautomerEnumerator().Canonicalize(mol)` and then
  `MolToInchiKey(...)[:14]`. Use multiprocessing, because it is slow.
- If a SMILES fails to parse, fall back to the `inchikey14` column and count
  how often that happens.
- **One molecule = one key.** Hiding always works on the key, so a tautomer
  duplicate under a different `inchikey14` cannot leak.
- Cache the results:
  - `out/cache/molecules.parquet`, one row per key: `key, smiles` (most
    frequent SMILES), `exact_mass` (RDKit `ExactMolWt`), `n_rows`,
    `n_eligible`, `has_tims`, `has_other` (any non-timsTOF row), `ion_mode`
    (majority over eligible rows, ties positive), `cluster_id`.
  - `out/cache/row_key.npy`: int32 array of length n_rows, the index into
    `molecules.parquet` for each train row.
- Cache files are reused when they exist. Recompute them when `--rebuild-cache`
  is given or the train file's size or mtime changed; store both in the
  parquet metadata.

### 3.4 Clustering (cached, in `molecules.parquet`)
- Default: **Butina** on Morgan fingerprints (radius 2, 2048 bits, via
  `rdFingerprintGenerator.GetMorganGenerator`), at Tanimoto threshold
  `--cluster-tc 0.6`, over **all** molecules.
- Butina needs the neighbour lists at ≥ 0.6. Build them with
  `DataStructs.BulkTanimotoSimilarity` in parallel chunks, then run the
  standard Butina assignment yourself (RDKit's `Butina.ClusterData` wants a
  full distance matrix, which is too big).
- **Time it first** on a random 20k subset and extrapolate (cost grows
  roughly as n²). If the full run is projected above ~2 h on this machine,
  stop and report the timing instead of running it.
- Provide `--cluster-method {butina,scaffold}`. `scaffold` means the generic
  Bemis–Murcko scaffold (`MurckoScaffold.MakeScaffoldGeneric`) as the
  cluster label, and serves as a fast fallback.
- Molecules whose SMILES fails to parse each get their own singleton cluster.

### 3.5 Shared sampling helpers
- `draw_n_spectra(rng, available)`: draw `k ~ Geometric(p=0.25)` on {1, 2, …},
  redraw while `k > 16`, and return `min(k, available)`. The median is about
  3.
- Molecule eligibility for any eval role: `n_eligible >= 1` and `exact_mass`
  between 157 and 1159.
- Weighted molecule sampling toward the **mass-bin targets**:
  - bins `157, 250, 350, 500, 800, 1160`;
  - target shares `0.15, 0.35, 0.30, 0.15, 0.05`;
  - draw weight = target share / candidate share of that bin, drawn without
    replacement.
  - `--ion-pos-share 0.7` weights ion mode the same way.
- Respect `--max-per-cluster` (default 2): count eval molecules per cluster
  across all levels within one output file.
- `sanity_report(df_out, meta)` prints and returns the realised values next
  to the test description:
  - number of molecules and spectra;
  - spectra per molecule (min, median, max);
  - mass (min, median, max) and mass-bin shares;
  - adduct counts and ion-mode shares;
  - CE distribution (single versus merged, and eV quantiles);
  - share of molecules per difficulty and per `change`;
  - share of eval molecules whose cluster is gone from the visible train.

  The report only warns and never changes the sample. It is also stored as
  JSON in the output parquet's metadata.

All randomness comes from one `np.random.default_rng(--seed)` (default 2026).
The same seed and inputs must give identical output.

## 4. `make_difficulty1.py`: same molecule, one dimension changed

CLI: `--n-molecules 400 --change-mix instrument=0.34,adduct=0.33,ce=0.33
--max-per-cluster 2 --seed 2026 --out dataset-creation/out/difficulty1.parquet`.
`--change-mix` may also be a single value such as `instrument=1`.

For each molecule `m` with change type `c`, choose eval rows `E` (eligible
timsTOF rows) and define which of `m`'s rows stay **visible**. Every other
row of `m` is hidden. `E` is never visible.

| change | eligible molecule | choose | visible rows of m |
|---|---|---|---|
| `instrument` | has eligible timsTOF rows in adduct `a` **and** non-timsTOF rows in the same adduct `a` | `a` uniformly among qualifying adducts; `E` drawn from eligible timsTOF rows in `a` | non-timsTOF rows in adduct `a` only. All timsTOF rows are hidden. CE is not matched, because units differ (see §1). |
| `adduct` | eligible timsTOF rows in ≥ 2 distinct adducts | eval adduct `a`; `E` drawn from eligible timsTOF rows in `a` | timsTOF rows in adducts ≠ `a`. All non-timsTOF rows and all rows in `a` are hidden. |
| `ce` | some adduct `a` has eligible timsTOF rows with ≥ 2 distinct CE tuples | `a`, then split its distinct CE tuples randomly into an eval part (≥ 1) and a visible part (≥ 1); `E` drawn from eligible rows in `a` with eval CE | timsTOF rows in `a` whose CE tuple is in the visible part. Everything else is hidden. |

- `|E| = draw_n_spectra(rng, number of candidate rows)`.
- Fill each change type's quota separately, in the order instrument (rarest,
  ~2.5k bridge molecules), then adduct, then ce. A molecule gets one change
  type at most.
- Log any shortfall and do not pad it.
- Cluster removal never applies here: `cluster_removed = False`.

## 5. `make_difficulty2.py`: unseen molecules (difficulty 3 via flag)

CLI: `--n-molecules 400 --max-per-cluster 2 --cluster-removed-frac 0.10
--mock-library --seed 2026 --out dataset-creation/out/difficulty2.parquet`.

- Candidates are all eligible molecules (§3.5).
- **Spread over clusters:**
  1. Shuffle the cluster IDs.
  2. Walk through them, sampling ≤ `--max-per-cluster` molecules per cluster
     with the mass and ion weights.
  3. Stop when `n` is reached.

  Simpler alternative: weighted sampling with a per-cluster cap. Pick one and
  document it.
- Every eval molecule has **all** its rows hidden. `E` comes from its eligible
  rows via `draw_n_spectra`.
- **Cluster removal.** For a fraction `--cluster-removed-frac` of the eval
  molecules (validated to lie in [0.05, 0.20], default 0.10), set
  `cluster_removed = True`. The dataloader then hides every row of every
  molecule in that cluster. This creates the "no similar molecule in train"
  cases.
  - Choose these molecules so their clusters are not huge: skip clusters with
    more than `--max-removed-cluster-size 200` molecules (configurable), so
    one choice cannot wipe out a large part of the train set.
  - Log the total number of rows hidden by cluster removal.
- `--mock-library` produces difficulty 3: set `exclude_from_library = True`
  and `difficulty = 3`; without it, `difficulty = 2`. There is no external
  candidate pool file yet, so for now this is only a flag the dataloader
  exposes (§7).

## 6. `make_overall.py`

CLI: `--n-molecules 400 --share 0.2,0.6,0.2 --cluster-removed-frac 0.10
--max-per-cluster 2 --seed 2026 --out dataset-creation/out/overall.parquet`.

- Import the selection functions from the two scripts. Each script must
  expose a function like `select_difficulty1(meta, molecules, n, rng,
  exclude_keys, cluster_counts, ...)` that returns the rows of the output
  table. Do not shell out.
- Order:
  1. Difficulty 2, then difficulty 3: they decide which clusters get removed.
  2. Difficulty 1.
- Constraints:
  - The three molecule sets are disjoint.
  - A difficulty-1 molecule must **not** lie in a cluster with
    `cluster_removed = True`, or its visible rows would be wiped. Exclude
    those clusters from difficulty-1 candidates.
  - `--max-per-cluster` counts across all three levels.
- Realised shares are reported. Shortfalls are logged, not padded.

## 7. `dataloader.py`

```python
@dataclass
class Split:
    truth: pd.DataFrame          # the output file, one row per eval molecule
    train_mask: np.ndarray       # bool, len = n train rows; True = usable for training/library
    eval_row_ids: np.ndarray     # train row ids of the eval spectra
    excluded_library_keys: set[str]   # keys that must not appear in the candidate pool
    def eval_spectra(self, train_path=...) -> pd.DataFrame: ...
    def train_table(self, columns, train_path=...) -> pa.Table: ...  # visible rows only

def load_split(split_path, train_path="data/train.parquet",
               cache_dir="dataset-creation/out/cache") -> Split
```

- `train_mask` starts as all True. Then:
  - set False for every `hidden_row_ids` entry, which includes the `E` rows;
  - set False for every row whose key's `cluster_id` is in
    `{cluster_id where cluster_removed}`, using `row_key.npy` and
    `molecules.parquet`.

  Assert that the cache hash stored in the split's metadata matches the cache
  on disk.
- `eval_spectra()` returns exactly the `test.parquet` columns, in the order
  `molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities,
  base_peak_intensity, adduct, ionization_mode, instrument_type, precursor_mz,
  collision_energy_ev, collision_energy_orig, collision_energy_orig_units`.
  - Build it by reading only the needed row groups and rows. Map row ids to
    (row group, offset) through the parquet metadata. Do not load the full 3
    GB file.
  - Apply test cleaning: drop peaks with m/z > `precursor_mz + 2.0`, then
    rescale so the max is 1.0. Drop spectra left with no peaks, and log them.
  - `spectrum_id = f"s{row_id}"`, which keeps the mapping back to the row
    traceable.
- `train_table()` reads the requested columns, filtered by `train_mask`.
  Process one row group at a time to bound memory.
- The dataloader is the only place that combines output files with
  `train.parquet`.

## 8. Output file schema (one row per eval molecule)

| column | type | meaning |
|---|---|---|
| `molecule_id` | str | `d{difficulty}_{6 hex}` from the seeded RNG, unique |
| `key` | str | tautomer-canonical InChIKey14, the answer |
| `smiles` | str | representative SMILES, the answer |
| `difficulty` | int8 | 1, 2 or 3 |
| `change` | str/null | `instrument`, `adduct` or `ce` (difficulty 1 only) |
| `cluster_id` | int64 | |
| `cluster_removed` | bool | the whole cluster is hidden from train |
| `exclude_from_library` | bool | difficulty 3 |
| `eval_row_ids` | list<int64> | train rows used as eval spectra |
| `hidden_row_ids` | list<int64> | all rows of this molecule hidden from train (⊇ eval rows; excludes cluster-removal rows) |
| `n_eval_spectra` | int16 | |
| `exact_mass` | float64 | |
| `ion_mode` | str | |

Parquet key-value metadata (JSON): script name, all CLI parameters, seed, the
cache hash (sha256 of `molecules.parquet`), train size and mtime, the
`sanity_report`, and shortfalls. The output should be well under 1 MB.

## 9. `check_split.py <file>`: must pass for every output

1. No eval row is visible: `train_mask[eval_row_ids]` is all False.
2. For difficulty 2 and 3, no visible row has an eval molecule's key.
3. For difficulty 1, every eval molecule still has ≥ 1 visible row. Each
   visible row differs from every eval spectrum of that molecule in the
   declared dimension:
   - `instrument`: no visible timsTOF row;
   - `adduct`: no visible row in the eval adduct;
   - `ce`: no visible row in the eval adduct with an eval CE tuple.
4. For `cluster_removed`, no visible row belongs to that cluster.
5. Eval rows are all eligible (§3.2), all timsTOF, and all in the ten adducts.
6. Molecule sets are disjoint across levels, and `molecule_id` is unique.
7. No cluster has more than `max_per_cluster` eval molecules.
8. Rerunning with the same seed gives an identical file (compare a hash of the
   table content, not the file bytes).
9. Print the sanity report.

## 10. Order of work

1. `common.py` with key and cache: run it and report time and counts. Compare
   the number of distinct keys with the 275,810 `inchikey14` values.
2. Clustering: time it on 20k molecules, then the full run, or report back if
   the projection is too slow. Report the number of clusters, the size
   distribution and the share of singletons.
3. `make_difficulty2.py`, `dataloader.py` and `check_split.py`. Run them for
   difficulty 2 and 3.
4. `make_difficulty1.py`, then `make_overall.py`. Run `check_split.py` on all
   four outputs.
5. `README.md`: purpose, the three levels, every CLI parameter with its
   default, the output schema, and a 10-line example of `load_split` used for
   training and eval.
6. Final report: per output file, the sanity report next to the test targets,
   any shortfalls, and runtimes.

Keep the style of the existing `scripts/`: a module docstring with usage,
`argparse`, plain functions, no new dependencies. Do not modify anything
under `src/` or `scripts/`.
