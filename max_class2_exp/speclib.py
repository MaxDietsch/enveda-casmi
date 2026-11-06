"""The analogue channel's reference-spectrum container (self-contained).

Copied from `max_class1_exp/library.py` so that `max_class2_exp/` no longer
imports class-1 code (README §2). It serves only the optional analogue
channel (spectra of candidate structures' nearest visible molecules); the
core class-2 channels (in-silico, MetFrag-lite, mass) never look at it.

  query_mass   median implied neutral mass of a query's spectra
  Library      cleaned spectra sorted by molecule, with peak arrays
  build_library  read + clean + cache the given train rows

Differences from the class-1 original:
  * the cache lives in max_class2_exp/out/cache (tagged with "class2"), so
    the two experiments never share or overwrite cache files;
  * an empty row list returns an empty Library instead of crashing (the
    analogue channel is off by default now, which means empty libraries);
  * the tag_extra field allows marking the leak self-test library.
"""

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "dataset-creation"))

import common  # noqa: E402  (dataset-creation, repo-wide)
from enveda_casmi26.pipeline import neutral_mass  # noqa: E402
from spectra_utils import CleanParams, clean_spectrum, entropy_weight, unit_norm  # noqa: E402

CACHE = HERE / "out" / "cache"
LIB_COLUMNS = ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz"]
# Part of the library cache tag, so extending it invalidates old caches once.
INFO_COLUMNS = ["adduct", "ionization_mode", "instrument_type", "precursor_mz",
                "collision_energy_ev", "ingest_lib"]


def ce_mean(v) -> float:
    """Scalar collision energy of a row: mean of the merged CE list (nan if none)."""
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return float("nan")
    return float(a.mean()) if a.size else float("nan")


# ------------------------------------------------------------------ candidates

def query_mass(precursor_mz, adducts) -> float:
    """Median neutral mass over a molecule's spectra (nan if none is usable)."""
    m = [neutral_mass(float(p), a) for p, a in zip(precursor_mz, adducts)]
    m = [x for x in m if np.isfinite(x)]
    return float(np.median(m)) if m else float("nan")


def implied_masses(spectra) -> np.ndarray:
    """Per-spectrum implied neutral masses, aligned with the spectra tuples
    (nan where the precursor/adduct is unusable)."""
    return np.array([neutral_mass(float(s[3]), s[4]) for s in spectra], dtype=float)


class Library:
    """Cleaned reference spectra, sorted by molecule.

    Per spectrum: row_id, mol, positive (polarity), adduct, instrument and the
    peak range [peak_start[i], peak_start[i+1]). Per peak: mz, p (sum 1 per
    spectrum), pw (entropy-weighted, sum 1), pu (unit L2 norm), spec.
    """

    def __init__(self, table: pd.DataFrame, mz: np.ndarray, p: np.ndarray, lens: np.ndarray):
        if len(table) == 0:
            # the analogue channel is off: an empty library must be usable
            self.table = table.reset_index(drop=True)
            for name, arr in (("mz", mz), ("p", p)):
                setattr(self, name, np.asarray(arr, dtype=float))
            self.peak_start = np.zeros(1, dtype=np.int64)
            self.spec = np.empty(0, dtype=np.int64)
            self.pw = np.empty(0)
            self.pu = np.empty(0)
            self.mol = np.empty(0, dtype=np.int64)
            self.positive = np.empty(0, dtype=bool)
            self.adduct = np.empty(0, dtype=object)
            self.tims = np.empty(0, dtype=bool)
            self.prec = np.empty(0)
            self.ce = np.empty(0)
            self.ingest = np.empty(0, dtype=object)
            self.nlmz = np.empty(0)
            self._mols, self._first = np.empty(0, dtype=np.int64), np.zeros(1, dtype=np.int64)
            return
        order = np.lexsort((table["row_id"].to_numpy(), table["mol"].to_numpy()))
        starts = np.r_[0, np.cumsum(lens)]
        self.table = table.iloc[order].reset_index(drop=True)
        new_lens = lens[order]
        self.peak_start = np.r_[0, np.cumsum(new_lens)].astype(np.int64)
        idx = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in order]) if len(order) else \
            np.empty(0, dtype=np.int64)
        self.mz = mz[idx]
        self.p = p[idx]
        self.spec = np.repeat(np.arange(len(order)), new_lens)
        self.pw = np.empty_like(self.p)
        self.pu = np.empty_like(self.p)
        for i in range(len(order)):
            s, e = self.peak_start[i], self.peak_start[i + 1]
            self.pw[s:e] = entropy_weight(self.p[s:e])
            self.pu[s:e] = unit_norm(self.p[s:e])
        mol = self.table["mol"].to_numpy()
        self.mol = mol
        self.positive = (self.table["ionization_mode"] == "positive").to_numpy()
        self.adduct = self.table["adduct"].to_numpy()
        self.tims = (self.table["instrument_type"] == common.TIMS).to_numpy()
        self.prec = self.table["precursor_mz"].to_numpy(dtype=float) \
            if "precursor_mz" in self.table else np.full(len(self.table), np.nan)
        self.ce = np.array([ce_mean(v) for v in self.table["collision_energy_ev"]]) \
            if "collision_energy_ev" in self.table else np.full(len(self.table), np.nan)
        self.ingest = self.table["ingest_lib"].to_numpy() \
            if "ingest_lib" in self.table else np.full(len(self.table), np.nan)
        # neutral-loss m/z per peak (precursor - fragment); peak order follows
        # mz, and the matching functions only require the *query* to be sorted.
        # A missing precursor propagates nan (matches nothing).
        self.nlmz = self.prec[self.spec] - self.mz
        self._mols, first = np.unique(mol, return_index=True)
        self._first = np.r_[first, len(mol)]

    def __len__(self) -> int:
        return len(self.table)

    def spectra_of(self, m: int) -> tuple[int, int]:
        """Spectrum index range [s, e) of molecule m (empty if absent)."""
        i = np.searchsorted(self._mols, m)
        if i >= len(self._mols) or self._mols[i] != m:
            return 0, 0
        return int(self._first[i]), int(self._first[i + 1])

    def peaks(self, s: int, e: int) -> tuple[int, int]:
        return int(self.peak_start[s]), int(self.peak_start[e])

    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        s, e = self.peak_start[i], self.peak_start[i + 1]
        return self.mz[s:e], self.p[s:e]


# ------------------------------------------------------------------- library

def _clean_row_group(args):
    train_path, rg, offsets, params = args
    pf = pq.ParquetFile(train_path)
    t = pf.read_row_group(rg, columns=LIB_COLUMNS).take(offsets)
    mzs = t.column("ms2_mzs").to_pylist()
    ints = t.column("ms2_normalized_intensities").to_pylist()
    prec = t.column("precursor_mz").to_numpy(zero_copy_only=False)
    out_mz, out_p, lens = [], [], np.zeros(len(offsets), dtype=np.int64)
    for i, (m, it, pr) in enumerate(zip(mzs, ints, prec)):
        cm, cp = clean_spectrum(m, it, pr, params)
        out_mz.append(cm)
        out_p.append(cp)
        lens[i] = len(cm)
    cat = (lambda xs: np.concatenate(xs) if xs else np.empty(0))
    return cat(out_mz), cat(out_p), lens


def _cache_tag(split_info: dict, rows: np.ndarray, params: CleanParams) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"folder": "max_class2_exp", "split": split_info.get("content_hash"),
                         "cache": split_info.get("cache_hash"), "params": asdict(params),
                         "columns": list(INFO_COLUMNS),
                         "tag_extra": split_info.get("tag_extra")}, sort_keys=True).encode())
    h.update(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def build_library(rows: np.ndarray, row_key: np.ndarray, train_path: Path, params: CleanParams,
                  split_info: dict, workers: int | None = None, use_cache: bool = True) -> Library:
    """Cleaned spectra of the given train rows (cached by rows + params + split).

    An empty row list returns an empty Library (the analogue channel off).
    """
    rows = np.unique(np.asarray(rows, dtype=np.int64))
    if len(rows) == 0:
        return Library(pd.DataFrame(columns=["row_id", "mol", *INFO_COLUMNS]),
                        np.empty(0), np.empty(0), np.empty(0, dtype=np.int64))
    tag = _cache_tag(split_info, rows, params)
    CACHE.mkdir(parents=True, exist_ok=True)
    tab_path, arr_path = CACHE / f"library_{tag}.parquet", CACHE / f"library_{tag}.npz"
    if use_cache and tab_path.exists() and arr_path.exists():
        common.log(f"[speclib] cached ({tab_path.name}, {len(rows):,} spectra)")
        z = np.load(arr_path)
        return Library(pd.read_parquet(tab_path), z["mz"], z["p"], z["lens"])

    t0 = time.time()
    common.log(f"[speclib] reading and cleaning {len(rows):,} train spectra ...")
    md = pq.ParquetFile(train_path).metadata
    bounds = np.r_[0, np.cumsum([md.row_group(i).num_rows for i in range(md.num_row_groups)])]
    rg = np.searchsorted(bounds, rows, side="right") - 1
    tasks = [(str(train_path), int(g), rows[rg == g] - bounds[g], params) for g in np.unique(rg)]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    # each task materializes one *full* parquet row group before taking its subset,
    # so cap the concurrent readers to bound the transient memory spike
    mzs, ps, lens = [], [], []
    with Pool(min(workers, len(tasks), 8)) as pool:
        prog = common.Progress("[speclib] row groups", len(tasks))
        for m, p, n in pool.imap(_clean_row_group, tasks):
            mzs.append(m)
            ps.append(p)
            lens.append(n)
            prog.update()
        prog.close()
    info = pq.read_table(train_path, columns=INFO_COLUMNS).take(rows).to_pandas()
    table = pd.DataFrame({"row_id": rows, "mol": row_key[rows].astype(np.int64),
                          "adduct": info["adduct"].to_numpy(),
                          "ionization_mode": info["ionization_mode"].to_numpy(),
                          "instrument_type": info["instrument_type"].to_numpy(),
                          "precursor_mz": info["precursor_mz"].to_numpy(dtype=float),
                          "collision_energy_ev": [list(v) if v is not None else None
                                                  for v in info["collision_energy_ev"]],
                          "ingest_lib": info["ingest_lib"].to_numpy()})
    mz, p, lens = np.concatenate(mzs), np.concatenate(ps), np.concatenate(lens)
    tmp_tab, tmp_arr = CACHE / f"{tag}.tmp.parquet", CACHE / f"{tag}.tmp.npz"
    table.to_parquet(tmp_tab)
    np.savez(tmp_arr, mz=mz, p=p, lens=lens)
    os.replace(tmp_tab, tab_path)    # atomic publish: a crash mid-write can never leave
    os.replace(tmp_arr, arr_path)    # a half-written cache that the next run would load
    common.log(f"[speclib] {len(rows):,} spectra, {len(mz):,} peaks after cleaning "
              f"({(lens == 0).sum():,} empty) in {time.time() - t0:.0f}s")
    return Library(table, mz, p, lens)
