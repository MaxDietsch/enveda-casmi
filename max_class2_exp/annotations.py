"""Molecule-level annotations for class-2 evaluation and features.

Two independent pieces (forum findings P0.2 and P0.4):

* **Molecular formulas** (RDKit `CalcMolFormula`), disk-cached by structure
  key. They power the *same-formula isomer panel* (the metric that measures
  what class 2 is actually about: ~98 % of wrong winners share the truth's
  formula) and the `cand_n_same_formula` feature.

* **Source-library statistics**: which train rows are timsTOF and which live
  in a *public* spectra corpus (any `ingest_lib` outside `enveda-180` /
  `enveda-np-examples`, following the forum's S1/S2 definitions). Per-row
  codes are cached once (mask-independent); per-molecule *visible* counts are
  recomputed per run so hidden rows never leak into features.

The `has_public_spectra` flag stratifies all results (P0.4): a pretrained
model's gain on structures with public spectra elsewhere is memorisation
(`748524`: ICEBERG +0.02 overall, ~0 on structures not in MassSpecGym).
"""

import hashlib
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "dataset-creation"))

import common  # noqa: E402  (dataset-creation, repo-wide)

# Enveda-curated libraries: the only sources whose spectra are NOT in a public
# corpus. Everything else (pluskal_ms2, riken, gnps, massbank, mona,
# spectraverse, msdial, drug_plus, masaryk, ...) counts as public overlap.
ENV_LIBS = {"enveda-180", "enveda-np-examples"}
CACHE_DIR = HERE / "out" / "cache"


# ------------------------------------------------------------------ formulas

def _formula_of(smiles: str) -> str:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    try:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        return rdMolDescriptors.CalcMolFormula(mol) if mol is not None else ""
    except Exception:
        return ""


def formulas_for(keys, smiles, cache_path: Path | None = None) -> dict[str, str]:
    """{key: formula} for the given structures, computed in parallel and
    disk-cached (key, formula) pairs; unparseable structures map to "".
    ~273k structures is a few minutes single-core, seconds with the pool.
    """
    cache_path = Path(cache_path or CACHE_DIR / "formulas.parquet")
    cache: dict[str, str] = {}
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cache = dict(zip(cached["key"].astype(str), cached["formula"].astype(str)))
    todo = []
    seen = set()
    for k, s in zip(keys, smiles):
        k = str(k)
        if k in cache or k in seen:
            continue
        seen.add(k)
        todo.append((k, str(s) if s is not None else ""))
    if todo:
        workers = max(1, (os.cpu_count() or 2) - 2)
        with get_context("fork").Pool(min(workers, 16)) as pool:
            got = pool.map(_formula_of, [s for _, s in todo], chunksize=256)
        cache.update({k: f for (k, _), f in zip(todo, got)})
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp.parquet")
        pd.DataFrame({"key": list(cache.keys()), "formula": list(cache.values())}).to_parquet(tmp)
        os.replace(tmp, cache_path)
        common.log(f"[annotations] computed {len(todo):,} new formulas "
                   f"({len(cache):,} cached at {cache_path.name})")
    return cache


def annotate_pool_formulas(pool, cache: dict[str, str]) -> None:
    """pool.formulas: object array aligned to pool members ('' where unknown)."""
    pool.formulas = np.array([cache.get(str(k), "") for k in pool.keys], dtype=object)


# ------------------------------------------------------- source-library stats

def load_row_sources(train_path: Path, n_rows: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per-train-row (ingest_lib code, is_timsTOF), cached to disk.

    Row-level facts are visibility-mask-independent, so they are computed
    once per train file and reused across runs and masks. lib_names is
    stored as a JSON string (a bare object array does not round-trip
    through npz/np.load).
    """
    train_path = Path(train_path)
    h = hashlib.sha256(str(train_path).encode())
    h.update(str(train_path.stat().st_size).encode() if train_path.exists() else b"?")
    h.update(str(n_rows).encode())
    tag = h.hexdigest()[:16]
    path = CACHE_DIR / f"row_sources_{tag}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        return z["lib_codes"], z["tims"], json.loads(str(z["lib_names"].item()))
    t = pq.read_table(train_path, columns=["ingest_lib", "instrument_type"]).to_pandas()
    codes, uniq = pd.factorize(t["ingest_lib"].astype(str), sort=True)
    lib_names = [str(u) for u in uniq]
    tims = (t["instrument_type"].astype(str) == common.TIMS).to_numpy()
    lib_codes = codes.astype(np.int16)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, lib_codes=lib_codes, tims=tims, lib_names=json.dumps(lib_names))
    os.replace(str(tmp), str(path))
    common.log(f"[annotations] row sources: {len(t):,} rows, libs {lib_names}")
    return lib_codes, tims, lib_names


def visible_source_stats(row_key: np.ndarray, mask: np.ndarray, lib_codes: np.ndarray,
                         tims: np.ndarray, lib_names: list[str], n_molecules: int) -> dict[str, np.ndarray]:
    """Per-molecule counts over VISIBLE rows only (hidden rows never leak):

      public_rows  visible rows from a public spectra corpus (any instrument)
      tims_rows    visible timsTOF rows (any library)
    """
    row_key = np.asarray(row_key, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    rows = np.flatnonzero(mask)
    mols = row_key[rows]
    public_lib = np.array([lib not in ENV_LIBS for lib in lib_names])
    public = public_lib[lib_codes[rows]]
    out = {"public_rows": np.zeros(n_molecules, dtype=np.int32),
           "tims_rows": np.zeros(n_molecules, dtype=np.int32)}
    np.add.at(out["public_rows"], mols[public], 1)
    np.add.at(out["tims_rows"], mols[tims[rows]], 1)
    return out


def annotate_pool_sources(pool, stats: dict[str, np.ndarray]) -> None:
    """Attach per-member visible public/tims row counts to the pool
    (external members have no train rows: zeros)."""
    n_mol = len(stats["public_rows"])
    ext = pool.struct >= n_mol
    pool.public_rows = stats["public_rows"][np.where(ext, 0, pool.struct)].astype(np.float32)
    pool.tims_rows = stats["tims_rows"][np.where(ext, 0, pool.struct)].astype(np.float32)
    pool.public_rows[ext] = 0.0
    pool.tims_rows[ext] = 0.0


def truth_public_flags(truth_keys, stats: dict[str, np.ndarray], key_to_mol) -> np.ndarray:
    """has_public_spectra per eval molecule (P0.4 stratification): the truth
    structure has rows in a public library *in the underlying train data*
    (pass stats computed with the all-rows mask, NOT the split mask — the
    flag measures corpus membership that a pretrained model could have
    memorised, which the split's hiding of eval rows does not erase)."""
    out = np.zeros(len(truth_keys), dtype=bool)
    for i, k in enumerate(truth_keys):
        m = key_to_mol.get(str(k))
        out[i] = m is not None and stats["public_rows"][int(m)] > 0
    return out
