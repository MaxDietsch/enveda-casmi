"""Candidate retrieval and the reference-spectrum library for stage 1.

Candidates: every molecule with at least one *visible* train row whose exact
mass lies within --ppm of the query's neutral mass (median over the query's
spectra of the mass implied by precursor m/z and adduct).

Library: the cleaned spectra of all visible train rows of all candidates.
Only the rows that can ever be compared are read, one row group at a time and
in parallel, and the result is cached in max_class1_exp/out/cache/.
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

import common  # noqa: E402  (dataset-creation)
from enveda_casmi26.pipeline import neutral_mass  # noqa: E402
from spectra import CleanParams, clean_spectrum, entropy_weight, unit_norm  # noqa: E402

CACHE = HERE / "out" / "cache"
LIB_COLUMNS = ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz"]
INFO_COLUMNS = ["adduct", "ionization_mode", "instrument_type"]


# --------------------------------------------------------------- candidates

def query_mass(precursor_mz, adducts) -> float:
    """Median neutral mass over a molecule's spectra (nan if none is usable)."""
    m = [neutral_mass(float(p), a) for p, a in zip(precursor_mz, adducts)]
    m = [x for x in m if np.isfinite(x)]
    return float(np.median(m)) if m else float("nan")


class CandidateIndex:
    """Molecules with >= 1 visible train row, sorted by exact mass."""

    def __init__(self, molecules: pd.DataFrame, row_key: np.ndarray, train_mask: np.ndarray):
        visible = np.zeros(len(molecules), dtype=bool)
        visible[row_key[train_mask]] = True
        mass = molecules["exact_mass"].to_numpy(dtype=float)
        ok = visible & np.isfinite(mass)
        idx = np.flatnonzero(ok)
        order = np.argsort(mass[idx], kind="stable")
        self.mol = idx[order]
        self.mass = mass[idx][order]

    def window(self, mass: float, ppm: float) -> tuple[np.ndarray, np.ndarray]:
        """(molecule indices, signed mass error in ppm) within ±ppm of mass."""
        if not np.isfinite(mass):
            return np.empty(0, dtype=np.int64), np.empty(0)
        lo, hi = np.searchsorted(self.mass, [mass * (1 - ppm * 1e-6), mass * (1 + ppm * 1e-6)])
        return self.mol[lo:hi], (self.mass[lo:hi] - mass) / mass * 1e6


# ------------------------------------------------------------------ library

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


class Library:
    """Cleaned reference spectra, sorted by molecule.

    Per spectrum: row_id, mol, positive (polarity), adduct, instrument and the
    peak range [peak_start[i], peak_start[i+1]). Per peak: mz, p (sum 1 per
    spectrum), pw (entropy-weighted, sum 1), pu (unit L2 norm), spec.
    """

    def __init__(self, table: pd.DataFrame, mz: np.ndarray, p: np.ndarray, lens: np.ndarray):
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


def _cache_tag(split_info: dict, rows: np.ndarray, params: CleanParams) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"split": split_info.get("content_hash"), "cache": split_info.get("cache_hash"),
                         "params": asdict(params)}, sort_keys=True).encode())
    h.update(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def build_library(rows: np.ndarray, row_key: np.ndarray, train_path: Path, params: CleanParams,
                  split_info: dict, workers: int | None = None, use_cache: bool = True) -> Library:
    """Cleaned spectra of the given train rows (cached by rows + params + split)."""
    rows = np.unique(np.asarray(rows, dtype=np.int64))
    tag = _cache_tag(split_info, rows, params)
    CACHE.mkdir(parents=True, exist_ok=True)
    tab_path, arr_path = CACHE / f"library_{tag}.parquet", CACHE / f"library_{tag}.npz"
    if use_cache and tab_path.exists() and arr_path.exists():
        common.log(f"[library] cached ({tab_path.name}, {len(rows):,} spectra)")
        z = np.load(arr_path)
        return Library(pd.read_parquet(tab_path), z["mz"], z["p"], z["lens"])

    t0 = time.time()
    common.log(f"[library] reading and cleaning {len(rows):,} train spectra ...")
    md = pq.ParquetFile(train_path).metadata
    bounds = np.r_[0, np.cumsum([md.row_group(i).num_rows for i in range(md.num_row_groups)])]
    rg = np.searchsorted(bounds, rows, side="right") - 1
    tasks = [(str(train_path), int(g), rows[rg == g] - bounds[g], params) for g in np.unique(rg)]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    mzs, ps, lens = [], [], []
    with Pool(min(workers, len(tasks))) as pool:
        prog = common.Progress("[library] row groups", len(tasks))
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
                          "instrument_type": info["instrument_type"].to_numpy()})
    mz, p, lens = np.concatenate(mzs), np.concatenate(ps), np.concatenate(lens)
    table.to_parquet(tab_path)
    np.savez(arr_path, mz=mz, p=p, lens=lens)
    common.log(f"[library] {len(rows):,} spectra, {len(mz):,} peaks after cleaning "
               f"({(lens == 0).sum():,} empty) in {time.time() - t0:.0f}s")
    return Library(table, mz, p, lens)
