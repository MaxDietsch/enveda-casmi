"""Combine an eval-set file from dataset-creation/out/ with train.parquet.

This is the only place where split files and the real data meet.

    import sys; sys.path.insert(0, "dataset-creation")
    from dataloader import load_split
    split = load_split("dataset-creation/out/overall.parquet")
    test_df = split.eval_spectra()                     # looks like test.parquet
    train = split.train_table(["normalized_smiles", "ms2_mzs"])   # visible rows only
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

TEST_COLUMNS = ["molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
                "base_peak_intensity", "adduct", "ionization_mode", "instrument_type",
                "precursor_mz", "collision_energy_ev", "collision_energy_orig",
                "collision_energy_orig_units"]
PRECURSOR_MARGIN = 2.0

# One-time cleaned copy written by dataset-cleaning/clean_data.py. Row-aligned
# with train.parquet (cleaning never drops rows), so split row ids are valid
# for both and eval spectra / training rows are served pre-cleaned.
CLEANED_TRAIN = common.PROJECT_ROOT / "data" / "train_clean.parquet"


def resolve_train_path(train_path=None) -> Path:
    """Explicit path wins; else the cleaned copy if it exists; else raw train."""
    if train_path is not None:
        return Path(train_path)
    if CLEANED_TRAIN.exists():
        return CLEANED_TRAIN
    return common.TRAIN


def _row_group_offsets(train_path: Path) -> np.ndarray:
    md = pq.ParquetFile(train_path).metadata
    sizes = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
    return np.concatenate([[0], np.cumsum(sizes)])


def read_rows(row_ids: np.ndarray, columns: list[str], train_path: Path = common.TRAIN) -> pa.Table:
    """Read specific train rows (in the given order) touching only their row groups."""
    row_ids = np.asarray(row_ids, dtype=np.int64)
    bounds = _row_group_offsets(train_path)
    pf = pq.ParquetFile(train_path)
    rg = np.searchsorted(bounds, row_ids, side="right") - 1
    parts, got = [], []
    for g in np.unique(rg):
        sel = row_ids[rg == g]
        t = pf.read_row_group(int(g), columns=columns)
        parts.append(t.take(pa.array(sel - bounds[g])))
        got.append(sel)
    if not parts:
        return pf.schema_arrow.empty_table().select(columns)
    table = pa.concat_tables(parts)
    pos = {r: i for i, r in enumerate(np.concatenate(got))}
    return table.take(pa.array([pos[r] for r in row_ids]))


def clean_peaks(mzs, ints, precursor_mz: float):
    """Test cleaning: drop peaks above precursor + 2 Da, rescale max to 1.0."""
    mzs = np.asarray(mzs if mzs is not None else [], dtype=float)
    ints = np.asarray(ints if ints is not None else [], dtype=float)
    keep = mzs <= precursor_mz + PRECURSOR_MARGIN
    mzs, ints = mzs[keep], ints[keep]
    if len(ints) == 0 or ints.max() <= 0:
        return None, None
    return mzs.tolist(), (ints / ints.max()).tolist()


@dataclass
class Split:
    truth: pd.DataFrame
    train_mask: np.ndarray
    eval_row_ids: np.ndarray
    excluded_library_keys: set
    info: dict = field(default_factory=dict)
    train_path: Path | None = None  # None -> resolve_train_path() at use time

    def eval_spectra(self, train_path: Path | None = None) -> pd.DataFrame:
        """Eval spectra with exactly the test.parquet columns, test cleaning applied."""
        train_path = Path(train_path or self.train_path or resolve_train_path())
        row_mol = {int(r): mid for mid, rows in zip(self.truth["molecule_id"], self.truth["eval_row_ids"])
                   for r in rows}
        rows = np.array(sorted(row_mol, key=lambda r: (row_mol[r], r)), dtype=np.int64)
        cols = [c for c in TEST_COLUMNS if c not in ("molecule_id", "spectrum_id")]
        df = read_rows(rows, cols, train_path).to_pandas()
        mz_out, int_out, keep = [], [], []
        for mz, it, prec in zip(df["ms2_mzs"], df["ms2_normalized_intensities"], df["precursor_mz"]):
            m, i = clean_peaks(mz, it, prec)
            keep.append(m is not None)
            mz_out.append(m)
            int_out.append(i)
        df["ms2_mzs"] = mz_out
        df["ms2_normalized_intensities"] = int_out
        df.insert(0, "molecule_id", [row_mol[int(r)] for r in rows])
        df.insert(1, "spectrum_id", [f"s{int(r)}" for r in rows])
        keep = np.array(keep, dtype=bool)
        if (~keep).any():
            common.log(f"[dataloader] dropped {int((~keep).sum())} eval spectra with no peak "
                       f"<= precursor + {PRECURSOR_MARGIN} Da: {df.loc[~keep, 'spectrum_id'].tolist()}")
        return df.loc[keep, TEST_COLUMNS].reset_index(drop=True)

    def iter_train(self, columns: list[str], train_path: Path | None = None):
        """Yield visible train rows one row group at a time (pa.Table with row_id)."""
        train_path = Path(train_path or self.train_path or resolve_train_path())
        bounds = _row_group_offsets(train_path)
        pf = pq.ParquetFile(train_path)
        for g in range(pf.metadata.num_row_groups):
            m = self.train_mask[bounds[g]:bounds[g + 1]]
            if not m.any():
                continue
            t = pf.read_row_group(g, columns=columns)
            t = t.append_column("row_id", pa.array(np.arange(bounds[g], bounds[g + 1], dtype=np.int64)))
            yield t.filter(pa.array(m))

    def train_table(self, columns: list[str], train_path: Path | None = None) -> pa.Table:
        """Visible train rows only (plus row_id), read one row group at a time."""
        return pa.concat_tables(list(self.iter_train(columns, train_path)))

    def library_additions(self) -> pd.DataFrame:
        """Structures to add to a train-derived candidate library (key, smiles).

        Difficulty-2 molecules mimic class 2 ("in PubChem/COCONUT, no public
        spectra"): their rows are hidden, so a library built from visible train
        rows alone would not contain them and they would behave like
        difficulty 3. Add these rows to the candidate pool (and drop
        excluded_library_keys from every pool, including external ones).
        """
        t = self.truth
        return t.loc[~t["exclude_from_library"].astype(bool) & (t["difficulty"] >= 2),
                     ["key", "smiles"]].reset_index(drop=True)

    def visible_library_keys(self, molecules: pd.DataFrame, row_key: np.ndarray) -> set:
        """Keys with >= 1 visible row (convenience for building a library)."""
        return set(molecules["key"].to_numpy()[np.unique(row_key[self.train_mask])])


def load_split(split_path, train_path=None, cache_dir=common.CACHE) -> Split:
    """Load a split. train_path=None auto-prefers data/train_clean.parquet
    (one-time output of dataset-cleaning/clean_data.py) over raw train.parquet."""
    split_path = Path(split_path)
    train_path = resolve_train_path(train_path)
    if train_path == CLEANED_TRAIN:
        common.log(f"[dataloader] using cleaned train data: {train_path}")
    cache_dir = Path(cache_dir)
    info = common.read_split_info(split_path)
    on_disk = common.cache_hash(cache_dir)
    if info.get("cache_hash") != on_disk:
        raise RuntimeError(f"{split_path} was built against cache {info.get('cache_hash', '?')[:12]}, "
                           f"but {cache_dir} holds {on_disk[:12]}; rebuild the split or the cache")
    truth = pd.read_parquet(split_path)
    molecules = pd.read_parquet(cache_dir / "molecules.parquet")
    row_key = np.load(cache_dir / "row_key.npy")
    n_train = pq.ParquetFile(train_path).metadata.num_rows
    if len(row_key) != n_train:
        raise RuntimeError("row_key.npy does not match train.parquet")
    mask = common.compute_train_mask(truth, row_key, molecules)
    eval_rows = np.concatenate([np.asarray(r, dtype=np.int64) for r in truth["eval_row_ids"]]) \
        if len(truth) else np.empty(0, np.int64)
    excluded = set(truth.loc[truth["exclude_from_library"].astype(bool), "key"])
    return Split(truth=truth, train_mask=mask, eval_row_ids=eval_rows,
                 excluded_library_keys=excluded, info=info, train_path=train_path)
