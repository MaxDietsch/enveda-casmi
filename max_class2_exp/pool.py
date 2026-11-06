"""The class-2 candidate pool: structures, not spectra.

Class 2 imitates "the structure is known (listed in PubChem / COCONUT), but
no public spectrum exists". A candidate library built from visible train rows
alone can never contain the answer (all of its rows are hidden), so the pool
is (simulator README §7.3):

    every molecule with >= 1 visible train row      the measured structures
  + split.library_additions()                        the class-2 structures
  + an optional external pool (e.g. a COCONUT extract) more class-2 structures
  - split.excluded_library_keys                      difficulty-3 structures

Structures are deduplicated by the scorer's key and sorted by exact mass for
±ppm window lookups. Every member carries its index into the fingerprint
universe (`struct`): train-derived members use their molecule index, external
members get fingerprint rows appended after `len(molecules)` (build_pool
returns those extra rows; np.vstack them onto the cached fps).

No spectra are stored here. The analogue channel (neighbours.py) supplies
spectra indirectly (a candidate's structural neighbours), the in-silico
channel (insilico.py) computes them from the structure.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "dataset-creation"))

import common  # noqa: E402


class Pool:
    """Unique structures sorted by exact mass (all arrays aligned to that order)."""

    def __init__(self, keys, smiles, mass, struct, visible_rows, is_addition, is_external):
        self.keys = np.asarray(keys)
        self.smiles = np.asarray(smiles, dtype=object)
        self.mass = np.asarray(mass, dtype=float)
        self.struct = np.asarray(struct, dtype=np.int64)
        self.visible_rows = np.asarray(visible_rows, dtype=np.int64)
        self.is_addition = np.asarray(is_addition, dtype=bool)
        self.is_external = np.asarray(is_external, dtype=bool)
        self._index = {k: i for i, k in enumerate(self.keys)}

    def __len__(self) -> int:
        return len(self.keys)

    def window(self, mass: float, ppm: float) -> tuple[np.ndarray, np.ndarray]:
        """(member indices, signed mass error in ppm) of all members within ±ppm."""
        if not np.isfinite(mass):
            return np.empty(0, dtype=np.int64), np.empty(0)
        lo, hi = np.searchsorted(self.mass, [mass * (1 - ppm * 1e-6), mass * (1 + ppm * 1e-6)])
        return np.arange(lo, hi, dtype=np.int64), (self.mass[lo:hi] - mass) / mass * 1e6

    def member_of_key(self, key: str) -> int:
        """Member index of a key (-1 if absent)."""
        return self._index.get(key, -1)


def external_rows(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Morgan r2/2048 fingerprints (packed uint64, 32 words) and exact masses.

    Unparsable SMILES get an all-zero row and a nan mass (such structures
    never become candidates because the mass window drops nan).
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = np.zeros((len(smiles), 2048 // 64), dtype=np.uint64)
    mass = np.full(len(smiles), np.nan)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s) if s else None
        if mol is not None:
            fps[i] = np.packbits(gen.GetFingerprintAsNumPy(mol).astype(np.uint8)).view(np.uint64)
            mass[i] = rdMolDescriptors.CalcExactMolWt(mol)
    return fps, mass


def build_pool(molecules: pd.DataFrame, row_key: np.ndarray, mask: np.ndarray,
               additions: pd.DataFrame | None = None, excluded_keys=(),
               external: pd.DataFrame | None = None) -> tuple[Pool, np.ndarray]:
    """The pool for one visibility mask; returns (pool, external fingerprint rows).

    additions: DataFrame with columns (key, smiles) - the class-2 structures
    (split.library_additions(), or the sampled pseudo-truths of a training
    run). external: DataFrame with columns (key, smiles[, mass]).
    """
    n_mol = len(molecules)
    counts = np.bincount(np.asarray(row_key)[np.asarray(mask, dtype=bool)], minlength=n_mol)
    keys, smi = molecules["key"].to_numpy(), molecules["smiles"].to_numpy()
    mass = molecules["exact_mass"].to_numpy(dtype=float)
    visible = (counts > 0) & np.isfinite(mass)
    excluded = set(excluded_keys)

    out = {"keys": [], "smiles": [], "mass": [], "struct": [],
           "visible_rows": [], "is_addition": [], "is_external": []}

    def _add(key, s, m, st, nvis, isadd, isext):
        out["keys"].append(key)
        out["smiles"].append(s)
        out["mass"].append(m)
        out["struct"].append(st)
        out["visible_rows"].append(nvis)
        out["is_addition"].append(isadd)
        out["is_external"].append(isext)

    for m in np.flatnonzero(visible):
        if keys[m] in excluded:
            continue
        _add(str(keys[m]), smi[m], mass[m], int(m), int(counts[m]), False, False)
    n_visible = len(out["keys"])
    seen = set(out["keys"])

    # additions: the class-2 structures (all rows hidden, so never in `visible`)
    if additions is not None and len(additions):
        mol_of_key = pd.Series(np.arange(n_mol), index=keys)
        for key, s in zip(additions["key"].to_numpy(), additions["smiles"].to_numpy()):
            key = str(key)
            if key in seen or key in excluded:
                continue
            m = mol_of_key.get(key, -1)
            if m < 0 or not np.isfinite(mass[m]):
                continue
            _add(key, smi[m] if smi[m] else s, mass[m], int(m), 0, True, False)
            seen.add(key)

    # external pool: extra structures with their own fingerprint rows
    fps_ext = np.zeros((0, 2048 // 64), dtype=np.uint64)
    if external is not None and len(external):
        ext = external.rename(columns=str.lower)
        for col in ("inchikey14", "key"):
            if col in ext.columns:
                ext = ext.rename(columns={col: "key"})
                break
        ext = ext.dropna(subset=["key", "smiles"])
        if len(ext):
            if "mass" not in ext.columns:
                _, ext_mass = external_rows(ext["smiles"].tolist())
            else:
                ext_mass = ext["mass"].to_numpy(dtype=float)
            fps_ext, _ = external_rows(ext["smiles"].tolist())
            for j, (key, s, m) in enumerate(zip(ext["key"].to_numpy(), ext["smiles"].to_numpy(), ext_mass)):
                key = str(key)
                if key in seen or key in excluded or not np.isfinite(m):
                    continue
                _add(key, s, float(m), n_mol + j, 0, False, True)
                seen.add(key)

    order = np.argsort(np.asarray(out["mass"], dtype=float), kind="stable")
    pool = Pool(
        np.asarray(out["keys"])[order],
        np.asarray(out["smiles"], dtype=object)[order],
        np.asarray(out["mass"], dtype=float)[order],
        np.asarray(out["struct"], dtype=np.int64)[order],
        np.asarray(out["visible_rows"], dtype=np.int64)[order],
        np.asarray(out["is_addition"], dtype=bool)[order],
        np.asarray(out["is_external"], dtype=bool)[order],
    )
    common.log(f"[pool] {len(pool):,} structures ({n_visible:,} with visible rows, "
               f"{int(pool.is_addition.sum()):,} additions, {int(pool.is_external.sum()):,} external; "
               f"{len(excluded)} excluded keys)")
    return pool, fps_ext
