"""Shared logic for the eval-set simulator (see README.md).

Loads the train metadata, decides which rows are eval-eligible, builds the
cached molecule table (tautomer-canonical key, mass), the Morgan fingerprints
and the neighbour counts, and holds the similarity queries, the sampling
helpers and the sanity report used by every make_*.py script.

Build or refresh the cache on its own (the make scripts also do it on demand):
    uv run python dataset-creation/common.py [--rebuild-cache] [--diversity-tc 0.6]
"""

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TRAIN = PROJECT_ROOT / "data" / "train.parquet"
OUT = HERE / "out"
CACHE = OUT / "cache"

TIMS = "timsTOF"
TEST_ADDUCTS = ("[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+",
                "[M+K]+", "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-")
MASS_MIN, MASS_MAX = 157.0, 1159.0
MASS_BINS = np.array([157, 250, 350, 500, 800, 1160], dtype=float)
MASS_TARGET = np.array([0.15, 0.35, 0.30, 0.15, 0.05])
MAX_SPECTRA = 16

META_COLUMNS = ["normalized_smiles", "inchikey14", "instrument_type", "adduct",
                "ionization_mode", "precursor_mz", "precursor_error_ppm",
                "base_peak_intensity", "collision_energy_ev",
                "collision_energy_orig_units", "num_peaks"]

TEST_TARGETS = {
    "n_molecules": 400, "n_spectra": 1500,
    "spectra_per_molecule": {"min": 1, "median": 3, "max": 16},
    "mass": {"min": 157, "median": 348, "max": 1159},
    "mass_bin_shares": dict(zip([f"{int(a)}-{int(b)}" for a, b in zip(MASS_BINS[:-1], MASS_BINS[1:])],
                                MASS_TARGET.tolist())),
    "instrument": "100% timsTOF",
    "adducts": list(TEST_ADDUCTS),
    "difficulty_shares_overall": {"1": 0.2, "2": 0.6, "3": 0.2},
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _fmt_secs(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}m{sec % 60:02d}s" if sec >= 60 else f"{sec}s"


class Progress:
    """Live progress line: done/total, percent, elapsed, estimated time left.

    On a terminal the line is updated in place; when the output is piped or
    logged, a new line is printed at most every 10 s.
    """

    def __init__(self, label: str, total: int, every: float = 1.0):
        self.label, self.total = label, max(int(total), 1)
        self.tty = sys.stdout.isatty()
        self.every = every if self.tty else max(every, 10.0)
        self.n, self.t0, self.last, self.closed = 0, time.time(), 0.0, False

    def update(self, k: int = 1) -> None:
        self.n += k
        now = time.time()
        if now - self.last >= self.every:
            self.last = now
            self._show(now)

    def _show(self, now: float, final: bool = False) -> None:
        el = now - self.t0
        rate = self.n / el if el > 0 else 0.0
        left = (self.total - self.n) / rate if rate > 0 else 0.0
        msg = (f"{self.label} {self.n:,}/{self.total:,} ({100 * self.n / self.total:.0f}%) "
               f"| {_fmt_secs(el)} elapsed" + ("" if final else f", ~{_fmt_secs(left)} left")
               + f" | {rate:,.0f}/s")
        if self.tty:
            print("\r" + msg.ljust(100), end="\n" if final else "", flush=True)
        else:
            print(msg, flush=True)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._show(time.time(), final=True)


def pool_map(pool, func, items: list, label: str, chunksize: int = 1) -> list:
    """pool.map with a live progress line (results in input order)."""
    prog = Progress(label, len(items))
    out = []
    for r in pool.imap(func, items, chunksize=chunksize):
        out.append(r)
        prog.update()
    prog.close()
    return out


# --------------------------------------------------------------------- meta

def load_meta(train_path: Path = TRAIN) -> pd.DataFrame:
    """Per-row metadata (no peak arrays), with row_id and a hashable ce tuple."""
    t0 = time.time()
    log(f"[load] reading {len(META_COLUMNS)} metadata columns of {Path(train_path).name} ...")
    table = pq.read_table(train_path, columns=META_COLUMNS)
    log(f"[load] {table.num_rows:,} rows read in {_fmt_secs(time.time() - t0)}; converting ...")
    ce = table.column("collision_energy_ev").to_pylist()
    meta = table.drop(["collision_energy_ev"]).to_pandas()
    meta["ce"] = pd.Series([None if v is None else tuple(v) for v in ce],
                           dtype=object, index=meta.index)
    meta["row_id"] = np.arange(len(meta), dtype=np.int64)
    return meta


def eligible_mask(meta: pd.DataFrame, ppm_tol: float = 10.0) -> np.ndarray:
    """Rows that may serve as eval spectra (PLAN §3.2)."""
    return ((meta["instrument_type"] == TIMS).to_numpy()
            & meta["adduct"].isin(TEST_ADDUCTS).to_numpy()
            & meta["ce"].notna().to_numpy()
            & (meta["base_peak_intensity"] >= 1000).fillna(False).to_numpy()
            & (meta["precursor_error_ppm"].abs() <= ppm_tol).fillna(False).to_numpy()
            & (meta["num_peaks"] >= 1).fillna(False).to_numpy())


# --------------------------------------------------------------------- keys

def _smiles_info(smiles: str):
    """(tautomer-canonical InChIKey14 or None, exact mass or nan)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None, float("nan")
    mass = Descriptors.ExactMolWt(mol)
    try:
        taut = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        key = Chem.MolToInchiKey(taut)[:14] or None
    except Exception:
        key = None
    return key, mass


def molecule_key(smiles: str) -> str | None:
    """Tautomer-canonical InChIKey first block of a SMILES, the key used in
    every output file (and by the competition scorer). None if unparsable."""
    return _smiles_info(smiles)[0]


def _train_stamp(train_path: Path) -> dict:
    st = os.stat(train_path)
    return {"train_size": st.st_size, "train_mtime": st.st_mtime}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_cache_meta(path: Path) -> dict:
    md = pq.read_schema(path).metadata or {}
    return json.loads(md.get(b"casmi_cache", b"{}"))


def _write_molecules(molecules: pd.DataFrame, path: Path, info: dict) -> None:
    table = pa.Table.from_pandas(molecules, preserve_index=False)
    md = dict(table.schema.metadata or {})
    md[b"casmi_cache"] = json.dumps(info).encode()
    pq.write_table(table.replace_schema_metadata(md), path)


def build_keys(meta: pd.DataFrame, ppm_tol: float, workers: int) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Molecule table and row -> molecule index."""
    t0 = time.time()
    smi_codes, smi_uniques = pd.factorize(meta["normalized_smiles"], use_na_sentinel=True)
    smi_list = [s if isinstance(s, str) else "" for s in smi_uniques]
    log(f"[keys] {len(smi_list):,} distinct SMILES; canonicalising tautomers on {workers} workers")
    with Pool(workers) as pool:
        info = pool_map(pool, _smiles_info, smi_list, "[keys] tautomer keys", chunksize=256)
    log(f"[keys] RDKit pass {_fmt_secs(time.time() - t0)}; assigning keys to rows ...")

    smi_key = np.array([k for k, _ in info], dtype=object)
    smi_mass = np.array([m for _, m in info], dtype=float)

    # Fallback: the inchikey14 column (one most frequent value per SMILES).
    fail = np.array([k is None for k in smi_key])
    n_fail_smiles = int(fail.sum())
    if n_fail_smiles:
        sub = pd.DataFrame({"code": smi_codes, "ik": meta["inchikey14"].to_numpy()})
        sub = sub[(sub["code"] >= 0) & fail[np.maximum(sub["code"], 0)]]
        fb = sub.groupby("code")["ik"].agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
        smi_key[fb.index.to_numpy()] = fb.to_numpy()
    row_key_str = np.where(smi_codes >= 0, smi_key[np.maximum(smi_codes, 0)], None)
    # Rows with no SMILES at all fall back to inchikey14 too.
    missing = pd.isna(row_key_str)
    row_key_str[missing] = meta["inchikey14"].to_numpy()[missing]
    n_fail_rows = int((fail[np.maximum(smi_codes, 0)] & (smi_codes >= 0)).sum() + (smi_codes < 0).sum())
    row_key_str = np.where(pd.isna(row_key_str), "UNKNOWN", row_key_str)

    keys = np.array(sorted(set(row_key_str)), dtype=object)
    key_index = {k: i for i, k in enumerate(keys)}
    row_key = np.fromiter((key_index[k] for k in row_key_str), dtype=np.int32, count=len(row_key_str))

    # Representative SMILES = most frequent SMILES per key (ties: lowest code).
    pair = pd.DataFrame({"key": row_key, "code": smi_codes})
    pair = pair[pair["code"] >= 0]
    cnt = pair.groupby(["key", "code"]).size().reset_index(name="n")
    cnt = cnt.sort_values(["key", "n", "code"], ascending=[True, False, True]).drop_duplicates("key")
    rep_code = np.full(len(keys), -1, dtype=np.int64)
    rep_code[cnt["key"].to_numpy()] = cnt["code"].to_numpy()
    has_rep = rep_code >= 0

    elig = eligible_mask(meta, ppm_tol)
    is_tims = (meta["instrument_type"] == TIMS).to_numpy()
    pos = (meta["ionization_mode"] == "positive").to_numpy()
    nk = len(keys)
    n_rows = np.bincount(row_key, minlength=nk)
    n_elig = np.bincount(row_key, weights=elig, minlength=nk).astype(np.int64)
    n_tims = np.bincount(row_key, weights=is_tims, minlength=nk)
    n_other = n_rows - n_tims
    pos_e = np.bincount(row_key, weights=elig & pos, minlength=nk)
    pos_all = np.bincount(row_key, weights=pos, minlength=nk)
    pos_share = np.where(n_elig > 0, pos_e / np.maximum(n_elig, 1), pos_all / np.maximum(n_rows, 1))

    molecules = pd.DataFrame({
        "key": keys.astype(str),
        "smiles": np.where(has_rep, np.array(smi_list, dtype=object)[np.maximum(rep_code, 0)], None),
        "exact_mass": np.where(has_rep, smi_mass[np.maximum(rep_code, 0)], np.nan),
        "n_rows": n_rows.astype(np.int64),
        "n_eligible": n_elig,
        "has_tims": n_tims > 0,
        "has_other": n_other > 0,
        "ion_mode": np.where(pos_share >= 0.5, "positive", "negative"),
    })
    stats = {"n_smiles": len(smi_list), "n_keys": nk,
             "n_inchikey14": int(meta["inchikey14"].nunique()),
             "smiles_parse_or_key_failures": n_fail_smiles,
             "rows_using_inchikey14_fallback": n_fail_rows,
             "keys_seconds": round(time.time() - t0, 1)}
    return molecules, row_key, stats


# -------------------------------------------------------------------- cache

def ensure_cache(train_path: Path = TRAIN, cache_dir: Path = CACHE, meta: pd.DataFrame | None = None,
                 rebuild: bool = False, workers: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (molecules, row_key), building the key cache when missing or stale."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mol_path, rk_path = cache_dir / "molecules.parquet", cache_dir / "row_key.npy"
    stamp = _train_stamp(train_path)
    if mol_path.exists() and rk_path.exists() and not rebuild:
        cached = _read_cache_meta(mol_path)
        if {k: cached.get(k) for k in stamp} == stamp:
            log(f"[cache] molecule keys: cached ({mol_path})")
            return pd.read_parquet(mol_path), np.load(rk_path)
        log("[cache] train file changed; rebuilding")
    else:
        log(f"[cache] molecule keys: building (~6 min) -> {mol_path}")
    if meta is None:
        t0 = time.time()
        meta = load_meta(train_path)
        log(f"[cache] loaded meta {len(meta):,} rows in {_fmt_secs(time.time() - t0)}")
    molecules, row_key, info = build_keys(meta, 10.0, workers or max(1, (os.cpu_count() or 2) - 2))
    log(f"[keys] {json.dumps(info)}")
    np.save(rk_path, row_key)
    _write_molecules(molecules, mol_path, {**info, **stamp})
    return molecules, row_key


def cache_hash(cache_dir: Path = CACHE) -> str:
    return file_sha256(Path(cache_dir) / "molecules.parquet")


# ------------------------------------------------------------- similarity

def _fp_rows(smiles: list[str]) -> np.ndarray:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = np.zeros((len(smiles), 2048 // 64), dtype=np.uint64)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s) if s else None
        if mol is not None:
            out[i] = np.packbits(gen.GetFingerprintAsNumPy(mol).astype(np.uint8)).view(np.uint64)
    return out


def fingerprint_matrix(molecules: pd.DataFrame, cache_dir: Path = CACHE, workers: int | None = None) -> np.ndarray:
    """Morgan r2/2048 fingerprints of every molecule as packed uint64 rows (cached).

    Unparsable molecules get an all-zero row (similarity 0 to everything).
    """
    path, tag = Path(cache_dir) / "fps.npy", Path(cache_dir) / "fps.hash"
    h = cache_hash(cache_dir)
    if path.exists() and tag.exists() and tag.read_text().strip() == h:
        log(f"[fps] fingerprints: cached ({path})")
        return np.load(path)
    log(f"[fps] fingerprints: building -> {path}")
    t0 = time.time()
    smiles = molecules["smiles"].fillna("").tolist()
    chunks = [smiles[i:i + 2000] for i in range(0, len(smiles), 2000)]
    with Pool(workers or max(1, (os.cpu_count() or 2) - 2)) as pool:
        fps = np.concatenate(pool_map(pool, _fp_rows, chunks, "[fps] chunks of 2,000 molecules"))
    np.save(path, fps)
    tag.write_text(h)
    log(f"[fps] {len(fps):,} fingerprints in {_fmt_secs(time.time() - t0)}")
    return fps


_SIM_FPS = None
_SIM_COLS = None
_SIM_CNT = None


def _tc_row(q: int) -> np.ndarray:
    """Tanimoto of molecule q to all molecules. Works word by word on the
    column-major copy and skips the query's empty 64-bit words (fingerprints
    are sparse), which is ~10x faster than a row-wise popcount."""
    inter = np.zeros(_SIM_COLS.shape[1], dtype=np.int32)
    for w in np.nonzero(_SIM_FPS[q])[0]:
        inter += np.bitwise_count(_SIM_COLS[w] & _SIM_FPS[q, w])
    union = _SIM_CNT + _SIM_CNT[q] - inter
    return np.divide(inter, union, out=np.zeros(len(union), dtype=np.float32), where=union > 0)


def _neighbours_of(args):
    q, tc = args
    return np.nonzero(_tc_row(q) >= tc)[0].astype(np.int64)


def _count_of(args):
    q, tc = args
    return int((_tc_row(q) >= tc).sum())


def _max_tc(args):
    q, allowed = args
    sims = _tc_row(q)
    sims[~allowed] = -1.0
    i = int(np.argmax(sims))
    return float(sims[i]), i


class Similarity:
    """Tanimoto queries of single molecules against all molecules (forked pool)."""

    def __init__(self, fps: np.ndarray, workers: int | None = None):
        global _SIM_FPS, _SIM_COLS, _SIM_CNT
        _SIM_FPS = fps
        _SIM_COLS = np.ascontiguousarray(fps.T)
        _SIM_CNT = np.bitwise_count(fps).sum(axis=1, dtype=np.int32)
        self.workers = workers or max(1, (os.cpu_count() or 2) - 2)

    def neighbours(self, queries, tc: float) -> list[np.ndarray]:
        """Molecule indices with Tanimoto >= tc to each query (query included)."""
        queries = list(queries)
        if not queries:
            return []
        with Pool(min(self.workers, len(queries))) as pool:
            return pool_map(pool, _neighbours_of, [(int(q), tc) for q in queries],
                            f"[similarity] neighbourhoods (Tc >= {tc:g})", chunksize=4)

    def neighbour_counts(self, queries, tc: float) -> np.ndarray:
        """Number of molecules with Tanimoto >= tc to each query (query included)."""
        queries = list(queries)
        if not queries:
            return np.empty(0, dtype=np.int64)
        with Pool(min(self.workers, len(queries))) as pool:
            return np.array(pool_map(pool, _count_of, [(int(q), tc) for q in queries],
                                     f"[neighbours] molecules (Tc >= {tc:g})", chunksize=256), dtype=np.int64)

    def tc_to(self, q: int, others: np.ndarray) -> np.ndarray:
        """Tanimoto of molecule q to the molecules in others (in-process)."""
        f = _SIM_FPS[others]
        inter = np.bitwise_count(f & _SIM_FPS[q]).sum(axis=1, dtype=np.int32)
        union = _SIM_CNT[others] + _SIM_CNT[q] - inter
        return np.divide(inter, union, out=np.zeros(len(union), dtype=np.float32), where=union > 0)

    def max_visible(self, queries, visible: np.ndarray) -> list[tuple[float, int]]:
        """(highest Tanimoto, molecule index) to any molecule with visible[m] True."""
        queries = list(queries)
        if not queries:
            return []
        with Pool(min(self.workers, len(queries))) as pool:
            return pool_map(pool, _max_tc, [(int(q), visible) for q in queries],
                            "[similarity] nearest visible molecule", chunksize=8)


def neighbour_counts(molecules: pd.DataFrame, sim: Similarity, tc: float,
                     cache_dir: Path = CACHE) -> np.ndarray:
    """Neighbours (Tanimoto >= tc, self included) per molecule, cached per tc.

    Only molecules with timsTOF rows (the possible eval molecules) are counted;
    the rest get 0.
    """
    path = Path(cache_dir) / f"neighbour_counts_tc{tc:g}.npy"
    tag = path.with_suffix(".hash")
    h = cache_hash(cache_dir)
    if path.exists() and tag.exists() and tag.read_text().strip() == h:
        log(f"[neighbours] counts at Tc >= {tc:g}: cached ({path})")
        return np.load(path)
    t0 = time.time()
    counts = np.zeros(len(molecules), dtype=np.int64)
    idx = np.nonzero(molecules["has_tims"].to_numpy())[0]
    log(f"[neighbours] counting neighbours at Tanimoto >= {tc:g} for {len(idx):,} molecules (~10 min)")
    counts[idx] = sim.neighbour_counts(idx, tc)
    np.save(path, counts)
    tag.write_text(h)
    log(f"[neighbours] done in {_fmt_secs(time.time() - t0)}; median {np.median(counts[idx]):.0f}, "
        f"max {counts.max()}")
    return counts


class Diversity:
    """Keeps eval molecules apart: no two may have Tanimoto >= tc.

    One instance is shared by all levels of one output file.
    """

    def __init__(self, sim: Similarity, tc: float):
        self.sim, self.tc = sim, tc
        self.picked: list[int] = []

    def accept(self, m: int) -> bool:
        if not self.picked:
            return True
        return bool(self.sim.tc_to(m, np.asarray(self.picked)).max() < self.tc)

    def add(self, m: int) -> None:
        self.picked.append(int(m))


# ------------------------------------------------------------- per-row frame

def attach_keys(meta: pd.DataFrame, molecules: pd.DataFrame, row_key: np.ndarray,
                ppm_tol: float = 10.0) -> pd.DataFrame:
    """Add key index and eligibility to meta (in place) and refresh n_eligible."""
    assert len(row_key) == len(meta), "row_key.npy does not match train.parquet; use --rebuild-cache"
    meta["mol"] = row_key
    meta["eligible"] = eligible_mask(meta, ppm_tol)
    meta["is_tims"] = (meta["instrument_type"] == TIMS).to_numpy()
    molecules["n_eligible"] = np.bincount(row_key, weights=meta["eligible"].to_numpy(),
                                          minlength=len(molecules)).astype(np.int64)
    return meta


def mol_rows(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(row ids sorted by molecule, start offsets) for O(1) lookup of a molecule's rows."""
    order = np.argsort(meta["mol"].to_numpy(), kind="stable")
    start = np.searchsorted(meta["mol"].to_numpy()[order], np.arange(meta["mol"].max() + 2))
    return order.astype(np.int64), start


def eval_candidates(molecules: pd.DataFrame) -> np.ndarray:
    """Molecule indices eligible for any eval role (PLAN §3.5)."""
    m = molecules
    return np.nonzero((m["n_eligible"] >= 1).to_numpy()
                      & (m["exact_mass"] >= MASS_MIN).to_numpy()
                      & (m["exact_mass"] <= MASS_MAX).to_numpy())[0]


# ---------------------------------------------------------------- sampling

def draw_n_spectra(rng: np.random.Generator, available: int) -> int:
    k = int(rng.geometric(0.25))
    while k > MAX_SPECTRA:
        k = int(rng.geometric(0.25))
    return min(k, available)


def target_weights(molecules: pd.DataFrame, cand: np.ndarray, ion_pos_share: float = 0.7,
                   base_weights: np.ndarray | None = None, iters: int = 20) -> np.ndarray:
    """Draw weights so the candidate distribution matches the mass-bin and ion targets.

    base_weights (per molecule) is the starting weight, normally
    1 / neighbour count, so a family of 50 similar molecules carries about
    as much weight as one isolated molecule. Mass and ion factors are then
    fitted by iterative proportional fitting on that base distribution.
    """
    mass = molecules["exact_mass"].to_numpy()[cand]
    mbin = np.clip(np.searchsorted(MASS_BINS, mass, side="right") - 1, 0, len(MASS_TARGET) - 1)
    pos = (molecules["ion_mode"].to_numpy()[cand] == "positive")
    w = np.ones(len(cand)) if base_weights is None else np.asarray(base_weights, float)[cand].copy()
    ion_target = np.array([1 - ion_pos_share, ion_pos_share])
    for _ in range(iters):
        share = np.bincount(mbin, weights=w, minlength=len(MASS_TARGET)) / w.sum()
        f = np.divide(MASS_TARGET, share, out=np.zeros_like(share), where=share > 0)
        w = w * f[mbin]
        share = np.bincount(pos.astype(int), weights=w, minlength=2) / w.sum()
        f = np.divide(ion_target, share, out=np.zeros_like(share), where=share > 0)
        w = w * f[pos.astype(int)]
    return w / w.sum()


def weighted_order(rng: np.random.Generator, weights: np.ndarray) -> np.ndarray:
    """A weighted random permutation (Efraimidis–Spirakis): walking it in order
    is weighted sampling without replacement."""
    u = rng.random(len(weights))
    with np.errstate(divide="ignore"):
        keys = np.where(weights > 0, np.log(u) / weights, -np.inf)
    return np.argsort(-keys, kind="stable")


def pick_diverse(rng: np.random.Generator, molecules: pd.DataFrame, cand: np.ndarray, n: int,
                 diversity: Diversity, ion_pos_share: float = 0.7, base_weights: np.ndarray | None = None,
                 accept=None) -> list[int]:
    """Weighted sample of up to n molecules from cand, each at Tanimoto < diversity.tc
    to every eval molecule picked so far (diversity is updated in place).
    accept(mol) -> bool can veto a pick."""
    if n <= 0 or len(cand) == 0:
        return []
    w = target_weights(molecules, cand, ion_pos_share, base_weights)
    picked = []
    for i in weighted_order(rng, w):
        if w[i] <= 0:
            break
        m = int(cand[i])
        if accept is not None and not accept(m):
            continue
        if not diversity.accept(m):
            continue
        picked.append(m)
        diversity.add(m)
        if len(picked) == n:
            break
    return picked


def selection_context(args: argparse.Namespace, molecules: pd.DataFrame) -> tuple[Similarity, Diversity, np.ndarray | None]:
    """(similarity queries, diversity tracker, base weights) for one output file."""
    sim = similarity(molecules, args.cache_dir)
    base = None
    if not args.no_diversity_balance:
        base = 1.0 / np.maximum(neighbour_counts(molecules, sim, args.diversity_tc, args.cache_dir), 1)
    return sim, Diversity(sim, args.diversity_tc), base


def new_molecule_id(rng: np.random.Generator, difficulty: int, used: set) -> str:
    while True:
        mid = f"d{difficulty}_{int(rng.integers(0, 16 ** 6)):06x}"
        if mid not in used:
            used.add(mid)
            return mid


def parse_mix(text: str, names: tuple[str, ...]) -> dict:
    out = {}
    for part in text.split(","):
        k, v = part.split("=")
        k = k.strip()
        if k not in names:
            raise ValueError(f"unknown change type {k!r}; expected one of {names}")
        out[k] = float(v)
    total = sum(out.values())
    return {k: v / total for k, v in out.items()}


def split_counts(n: int, shares: list[float]) -> list[int]:
    """Largest-remainder rounding of n * shares."""
    raw = np.array(shares, dtype=float) / sum(shares) * n
    base = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - base), kind="stable")[: n - base.sum()]:
        base[i] += 1
    return base.tolist()


# ------------------------------------------------------------------ output

OUTPUT_SCHEMA = pa.schema([
    ("molecule_id", pa.string()), ("key", pa.string()), ("smiles", pa.string()),
    ("difficulty", pa.int8()), ("change", pa.string()),
    ("neighbours_removed", pa.bool_()), ("removed_keys", pa.list_(pa.string())),
    ("exclude_from_library", pa.bool_()),
    ("eval_row_ids", pa.list_(pa.int64())), ("hidden_row_ids", pa.list_(pa.int64())),
    ("n_eval_spectra", pa.int16()), ("exact_mass", pa.float64()), ("ion_mode", pa.string()),
    ("max_visible_tc", pa.float32()), ("nearest_visible_key", pa.string()),
])


def make_record(molecules: pd.DataFrame, m: int, molecule_id: str, difficulty: int, change,
                eval_rows, hidden_rows, removed_keys=(), exclude_from_library=False) -> dict:
    row = molecules.iloc[m]
    eval_rows = sorted(int(r) for r in eval_rows)
    hidden_rows = sorted(int(r) for r in hidden_rows)
    return {"molecule_id": molecule_id, "key": row["key"], "smiles": row["smiles"],
            "difficulty": difficulty, "change": change,
            "neighbours_removed": len(removed_keys) > 0, "removed_keys": sorted(removed_keys),
            "exclude_from_library": bool(exclude_from_library),
            "eval_row_ids": eval_rows, "hidden_row_ids": hidden_rows,
            "n_eval_spectra": len(eval_rows), "exact_mass": float(row["exact_mass"]),
            "ion_mode": row["ion_mode"], "max_visible_tc": None, "nearest_visible_key": None}


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    if not records:
        return OUTPUT_SCHEMA.empty_table().to_pandas()
    return pa.Table.from_pylist(records, schema=OUTPUT_SCHEMA).to_pandas()


def _plain(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_plain(x) for x in v]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, np.floating):
        return None if np.isnan(v) else round(float(v), 6)
    return v.item() if hasattr(v, "item") else v


def content_hash(df: pd.DataFrame) -> str:
    """Hash of the table content (independent of parquet byte layout)."""
    h = hashlib.sha256()
    for rec in df.sort_values("molecule_id").to_dict("records"):
        h.update(json.dumps({k: _plain(v) for k, v in rec.items()}, sort_keys=True).encode())
    return h.hexdigest()


def write_output(df: pd.DataFrame, path: Path, info: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=OUTPUT_SCHEMA, preserve_index=False)
    info = {**info, "content_hash": content_hash(df)}
    md = dict(table.schema.metadata or {})
    md[b"casmi_split"] = json.dumps(info, default=str).encode()
    pq.write_table(table.replace_schema_metadata(md), path)
    log(f"[out] wrote {path} ({path.stat().st_size / 1024:.0f} KiB, {len(df)} molecules)")


def read_split_info(path: Path) -> dict:
    md = pq.read_schema(path).metadata or {}
    return json.loads(md.get(b"casmi_split", b"{}"))


def base_info(script: str, args: argparse.Namespace, train_path: Path, cache_dir: Path) -> dict:
    return {"script": script, "params": vars(args), "seed": args.seed,
            "cache_hash": cache_hash(cache_dir), **_train_stamp(train_path)}


# ------------------------------------------------------------- train mask

def compute_train_mask(df: pd.DataFrame, row_key: np.ndarray, molecules: pd.DataFrame) -> np.ndarray:
    """True = row may be used for training / library (PLAN §7)."""
    mask = np.ones(len(row_key), dtype=bool)
    for ids in df["hidden_row_ids"]:
        mask[np.asarray(ids, dtype=np.int64)] = False
    for ids in df["eval_row_ids"]:
        mask[np.asarray(ids, dtype=np.int64)] = False
    removed = {k for keys in df["removed_keys"] for k in keys}
    if removed:
        idx = np.nonzero(molecules["key"].isin(removed).to_numpy())[0]
        mask[np.isin(row_key, idx)] = False
    return mask


def similarity(molecules: pd.DataFrame, cache_dir: Path = CACHE) -> "Similarity":
    return Similarity(fingerprint_matrix(molecules, cache_dir))


def annotate_similarity(df: pd.DataFrame, molecules: pd.DataFrame, row_key: np.ndarray,
                        sim: "Similarity") -> pd.DataFrame:
    """Fill max_visible_tc / nearest_visible_key: the most similar molecule that
    still has a visible train row once this split is applied."""
    if not len(df):
        return df
    mask = compute_train_mask(df, row_key, molecules)
    visible = np.zeros(len(molecules), dtype=bool)
    visible[row_key[mask]] = True
    key_index = pd.Series(np.arange(len(molecules)), index=molecules["key"])
    res = sim.max_visible(key_index[df["key"]].to_numpy(), visible)
    df = df.copy()
    df["max_visible_tc"] = np.array([round(t, 4) for t, _ in res], dtype=np.float32)
    df["nearest_visible_key"] = [molecules["key"].iat[i] if t >= 0 else None for t, i in res]
    return df


# ------------------------------------------------------------ sanity report

def _q(x, qs=(0.0, 0.5, 1.0)):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return {}
    return {name: round(float(np.quantile(x, q)), 2) for name, q in zip(("min", "median", "max"), qs)}


def sanity_report(df_out: pd.DataFrame, meta: pd.DataFrame, train_mask: np.ndarray | None = None,
                  row_key: np.ndarray | None = None, molecules: pd.DataFrame | None = None,
                  verbose: bool = True, cache_dir: Path = CACHE) -> dict:
    """Realised distributions next to the test description. Warns only."""
    rows = np.concatenate([np.asarray(r, dtype=np.int64) for r in df_out["eval_row_ids"]]) \
        if len(df_out) else np.empty(0, np.int64)
    ev = meta.iloc[rows]
    n_mol = len(df_out)
    mass = df_out["exact_mass"].to_numpy(dtype=float)
    mbin = np.clip(np.searchsorted(MASS_BINS, mass, side="right") - 1, 0, len(MASS_TARGET) - 1)
    bin_names = list(TEST_TARGETS["mass_bin_shares"])
    ce_len = ev["ce"].map(lambda t: len(t) if t is not None else 0)
    ce_vals = np.concatenate([np.asarray(t, float) for t in ev["ce"] if t is not None]) \
        if len(ev) else np.empty(0)
    rep = {
        "n_molecules": n_mol, "n_spectra": int(len(rows)),
        "spectra_per_molecule": _q(df_out["n_eval_spectra"]),
        "mass": _q(mass),
        "mass_bin_shares": {b: round(float((mbin == i).mean()), 3) if n_mol else 0.0
                            for i, b in enumerate(bin_names)},
        "adduct_counts": ev["adduct"].value_counts().to_dict(),
        "ion_mode_share_spectra": ev["ionization_mode"].value_counts(normalize=True).round(3).to_dict(),
        "ion_mode_share_molecules": df_out["ion_mode"].value_counts(normalize=True).round(3).to_dict(),
        "instrument_counts": ev["instrument_type"].value_counts().to_dict(),
        "ce": {"single_share": round(float((ce_len == 1).mean()), 3) if len(ev) else 0.0,
               "merged_share": round(float((ce_len > 1).mean()), 3) if len(ev) else 0.0,
               "ev_quantiles": {str(q): round(float(np.quantile(ce_vals, q)), 1)
                                for q in (0.05, 0.25, 0.5, 0.75, 0.95)} if len(ce_vals) else {}},
        "difficulty_shares": df_out["difficulty"].astype(int).astype(str).value_counts(normalize=True)
        .round(3).to_dict(),
        "change_shares": df_out["change"].fillna("none").value_counts(normalize=True).round(3).to_dict(),
        "neighbours_removed_share": round(float(df_out["neighbours_removed"].mean()), 3) if n_mol else 0.0,
    }
    if train_mask is not None and row_key is not None and molecules is not None and n_mol:
        rep["n_visible_train_rows"] = int(train_mask.sum())
        fps = fingerprint_matrix(molecules, cache_dir)
        key_index = pd.Series(np.arange(len(molecules)), index=molecules["key"])
        f = fps[key_index[df_out["key"]].to_numpy()]
        c = np.bitwise_count(f).sum(axis=1)
        nearest = np.zeros(len(f))
        for i in range(len(f)):
            inter = np.bitwise_count(f & f[i]).sum(axis=1)
            t = np.divide(inter, c + c[i] - inter, out=np.zeros(len(f)), where=(c + c[i] - inter) > 0)
            t[i] = 0
            nearest[i] = t.max() if len(f) > 1 else 0
        rep["max_tc_between_eval"] = {"max": round(float(nearest.max()), 3),
                                      "median": round(float(np.median(nearest)), 3)}
    tc = df_out["max_visible_tc"].to_numpy(dtype=float) if n_mol else np.empty(0)
    tc = tc[np.isfinite(tc)]
    if len(tc):
        rep["max_visible_tc"] = {
            "quantiles": {str(q): round(float(np.quantile(tc, q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "share_below_0.4": round(float((tc < 0.4).mean()), 3),
            "share_below_0.5": round(float((tc < 0.5).mean()), 3),
            "median_by_difficulty": df_out.groupby("difficulty")["max_visible_tc"].median().round(3)
            .rename(lambda d: str(int(d))).to_dict()}

    warnings = []
    if n_mol:
        med = rep["spectra_per_molecule"]["median"]
        if not 2 <= med <= 4:
            warnings.append(f"median spectra/molecule {med} (test 3)")
        if rep["spectra_per_molecule"]["max"] > MAX_SPECTRA:
            warnings.append("more than 16 spectra for a molecule")
        if not 300 <= rep["mass"]["median"] <= 400:
            warnings.append(f"median mass {rep['mass']['median']} (test 348)")
        for b, t in TEST_TARGETS["mass_bin_shares"].items():
            if abs(rep["mass_bin_shares"][b] - t) > 0.07:
                warnings.append(f"mass bin {b}: {rep['mass_bin_shares'][b]} vs target {t}")
        if set(ev["instrument_type"].unique()) - {TIMS}:
            warnings.append("non-timsTOF eval spectra")
        missing = set(TEST_ADDUCTS) - set(rep["adduct_counts"])
        if missing:
            warnings.append(f"adducts absent from eval: {sorted(missing)}")
    rep["warnings"] = warnings

    if verbose:
        log("---- sanity report (realised | test) ----")
        log(f"molecules {rep['n_molecules']} (test ~400)    spectra {rep['n_spectra']} | "
            f"~{round(3.75 * rep['n_molecules'])} at the test rate of 1500/400")
        log(f"spectra/molecule {rep['spectra_per_molecule']} | min 1, median 3, max 16")
        log(f"mass {rep['mass']} | min 157, median 348, max 1159")
        log("mass bins " + ", ".join(f"{b}: {rep['mass_bin_shares'][b]:.2f}/{t:.2f}"
                                     for b, t in TEST_TARGETS["mass_bin_shares"].items()))
        log(f"adducts {rep['adduct_counts']}")
        log(f"ion mode (spectra) {rep['ion_mode_share_spectra']}  (molecules) {rep['ion_mode_share_molecules']}")
        log(f"instrument {rep['instrument_counts']} | 100% timsTOF")
        log(f"CE {rep['ce']}")
        log(f"difficulty {rep['difficulty_shares']}  change {rep['change_shares']}")
        log(f"neighbours_removed share {rep['neighbours_removed_share']}")
        if "max_tc_between_eval" in rep:
            log(f"max Tanimoto between eval molecules {rep['max_tc_between_eval']}")
        if "max_visible_tc" in rep:
            log(f"max Tanimoto to a visible train molecule {rep['max_visible_tc']}")
        for w in warnings:
            log(f"WARNING: {w}")
    return rep


# ------------------------------------------------------------ script glue

def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-molecules", type=int, default=1000,
                        help="number of eval molecules (shortfalls are logged, not padded)")
    parser.add_argument("--diversity-tc", type=float, default=0.6,
                        help="no two eval molecules may have Morgan Tanimoto >= this")
    parser.add_argument("--no-diversity-balance", action="store_true",
                        help="weight molecules equally instead of by 1 / number of neighbours")
    parser.add_argument("--ion-pos-share", type=float, default=0.7,
                        help="target share of positive-mode molecules (sampling weight)")
    parser.add_argument("--ppm-tol", type=float, default=10.0,
                        help="max |precursor error| in ppm for a row to be an eval spectrum")
    parser.add_argument("--seed", type=int, default=2026,
                        help="random seed; same seed and cache give an identical file")
    parser.add_argument("--train", type=Path, default=TRAIN, help="train parquet")
    parser.add_argument("--cache-dir", type=Path, default=CACHE, help="cache directory")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="recompute keys, fingerprints and neighbour counts")


def prepare(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Load meta and the cache; returns (meta with keys, molecules, row_key)."""
    t0 = time.time()
    meta = load_meta(args.train)
    log(f"[load] meta {len(meta):,} rows in {time.time() - t0:.0f}s")
    molecules, row_key = ensure_cache(args.train, args.cache_dir, meta=meta, rebuild=args.rebuild_cache)
    attach_keys(meta, molecules, row_key, args.ppm_tol)
    return meta, molecules, row_key


def finish(df: pd.DataFrame, args: argparse.Namespace, script: str, meta: pd.DataFrame,
           molecules: pd.DataFrame, row_key: np.ndarray, shortfalls: dict, extra: dict | None = None) -> None:
    mask = compute_train_mask(df, row_key, molecules)
    rep = sanity_report(df, meta, mask, row_key, molecules, cache_dir=args.cache_dir)
    info = base_info(script, args, args.train, args.cache_dir)
    info.update(sanity_report=rep, shortfalls=shortfalls, **(extra or {}))
    for k, v in shortfalls.items():
        if v:
            log(f"SHORTFALL {k}: {v}")
    write_output(df, args.out, info)
    from plots import plot_split
    plot_split(df, meta, molecules, Path(args.out).with_suffix(".png"),
               f"{Path(args.out).name} · {script} · seed {args.seed}", getattr(args, "neighbour_tc", None))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=TRAIN, help="train parquet")
    parser.add_argument("--cache-dir", type=Path, default=CACHE, help="cache directory")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="recompute keys, fingerprints and neighbour counts")
    parser.add_argument("--diversity-tc", type=float, default=0.6,
                        help="threshold for the neighbour counts to precompute")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes (default: CPUs - 2)")
    args = parser.parse_args()
    t0 = time.time()
    log("== step 1/3: molecule keys")
    molecules, row_key = ensure_cache(args.train, args.cache_dir, rebuild=args.rebuild_cache,
                                      workers=args.workers)
    if args.rebuild_cache:
        for p in Path(args.cache_dir).glob("*.hash"):
            p.unlink()
    log("== step 2/3: Morgan fingerprints")
    sim = similarity(molecules, args.cache_dir)
    log(f"== step 3/3: neighbour counts at Tanimoto >= {args.diversity_tc:g}")
    neighbour_counts(molecules, sim, args.diversity_tc, args.cache_dir)
    log(json.dumps(_read_cache_meta(Path(args.cache_dir) / "molecules.parquet"), indent=1))
    log(f"[cache] ready: {len(molecules):,} molecules, {len(row_key):,} rows, "
        f"{_fmt_secs(time.time() - t0)} total")


if __name__ == "__main__":
    main()
