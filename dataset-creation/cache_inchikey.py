"""RDKit-free molecule-key cache: group train rows by the shipped `inchikey14`.

The faithful cache builder (`common.build_keys`) canonicalises tautomers with
RDKit to merge variant SMILES of one skeleton into a single candidate. Kaggle's
kernel image has no RDKit, so this module rebuilds the same cache format from
the two columns train.parquet already ships:

  key         the row's `inchikey14` (standard InChI already normalises most
              tautomers, so grouping differs from the tautomer-canonical keys
              for ~1.6% of rows — measured locally, see README)
  exact_mass  the group's monoisotopic mass parsed from `molecular_formula`
              (charged formulas like "C12H19N2O2+" are parsed as written;
              measured: 15 of 274k molecules shift by more than the 10 ppm
              candidate window)
  smiles      the most frequent `normalized_smiles` per key (same rule as
              `build_keys`, ties by lowest factorize code)

The output (molecules.parquet + row_key.npy, with the same `casmi_cache`
metadata stamp) is what `common.ensure_cache` loads, so every consumer
(CandidateIndex, build_library, predict_test) works unchanged. Build it into a
SEPARATE cache dir and point --cache-dir there; never mix it with the tautomer
cache, whose row_key was computed with different keys.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

# monoisotopic masses of the elements that occur in the train structures
MASS = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740052, "O": 15.9949146221,
        "S": 31.97207074, "P": 30.97376163, "F": 18.9984031627, "Cl": 34.968852683,
        "Br": 78.9183371, "I": 126.9044719, "Si": 27.9769265327, "B": 11.0093054,
        "Se": 79.9165218, "As": 74.9215964, "Na": 22.989769282, "K": 38.9637064864,
        "Ca": 39.962590861, "Fe": 55.93493633, "Mg": 23.985041697, "Zn": 63.929146577,
        "Cu": 62.92959772, "Mn": 54.93805013, "Li": 7.016003436, "Al": 26.981538675,
        "Ni": 57.93534241, "Co": 58.93319504, "Ag": 106.9050916, "Sn": 119.90219963,
        "Pb": 207.9766525, "Hg": 201.970629, "Cr": 51.94051146, "Mo": 97.90540482,
        "V": 50.94395949, "Ti": 47.94794198, "Ge": 73.921177761, "Rb": 84.911789732,
        "Sr": 87.9056125, "Cs": 132.905451933, "Ba": 137.90524700}
TOK = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(formula) -> float:
    """Monoisotopic mass of a molecular formula string (nan if unparsable).

    A trailing '+' (charged forms shipped for some rows) is ignored — the H
    count is taken as written, which is the neutral-structure convention the
    column documents.
    """
    if not formula or not isinstance(formula, str):
        return float("nan")
    f = formula.rstrip("+")
    total, pos = 0.0, 0
    for el, n in TOK.findall(f):
        if el not in MASS:
            return float("nan")
        total += MASS[el] * (int(n) if n else 1)
        pos += len(el) + len(n)
    return total if pos == len(f) else float("nan")


def build_inchikey_cache(train_path: Path, cache_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Molecule table + row->molecule index from inchikey14 + molecular_formula.

    Writes molecules.parquet / row_key.npy with the same metadata stamp
    ensure_cache checks, then returns them.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tab = pq.read_table(train_path, columns=["normalized_smiles", "inchikey14", "molecular_formula"])
    smiles = tab.column("normalized_smiles").to_pylist()
    ik = np.asarray(tab.column("inchikey14").to_pylist(), dtype=object)
    formulas = tab.column("molecular_formula").to_pylist()
    del tab
    n_rows_total = len(ik)

    key_str = np.where(pd.isna(ik), "UNKNOWN", ik)              # same fallback as build_keys
    row_key, keys = pd.factorize(key_str, sort=True)
    row_key = row_key.astype(np.int32)
    nk = len(keys)

    smi_codes, smi_uniques = pd.factorize(pd.Series(smiles, dtype=object), use_na_sentinel=True)
    smi_list = [s if isinstance(s, str) else "" for s in smi_uniques]

    # representative SMILES per key: most frequent, ties by lowest code (= build_keys)
    pair = pd.DataFrame({"key": row_key, "code": smi_codes})
    pair = pair[pair["code"] >= 0]
    cnt = pair.groupby(["key", "code"]).size().reset_index(name="n")
    cnt = cnt.sort_values(["key", "n", "code"], ascending=[True, False, True]).drop_duplicates("key")
    rep_code = np.full(nk, -1, dtype=np.int64)
    rep_code[cnt["key"].to_numpy()] = cnt["code"].to_numpy()
    has_rep = rep_code >= 0

    # exact mass: median parsed formula mass over the representative SMILES' rows
    fm = np.array([formula_mass(f) for f in formulas])
    rep_mask = has_rep[row_key] & (smi_codes == rep_code[row_key]) & np.isfinite(fm)
    s = pd.Series(fm[rep_mask])
    s.index = row_key[rep_mask]
    g = s.groupby(level=0).median()
    exact_mass = np.full(nk, np.nan)
    exact_mass[g.index.to_numpy()] = g.to_numpy()

    molecules = pd.DataFrame({
        "key": keys.astype(str),
        "smiles": np.where(has_rep, np.array(smi_list, dtype=object)[np.maximum(rep_code, 0)], None),
        "exact_mass": exact_mass,
        "n_rows": np.bincount(row_key, minlength=nk).astype(np.int64),
    })
    stamp = {"train_size": Path(train_path).stat().st_size,
             "train_mtime": Path(train_path).stat().st_mtime,
             "n_keys": nk, "builder": "inchikey14 (RDKit-free)"}
    np.save(cache_dir / "row_key.npy", row_key)
    common._write_molecules(molecules, cache_dir / "molecules.parquet", stamp)
    common.log(f"[cache] inchikey14 cache: {nk:,} molecules from {n_rows_total:,} rows -> {cache_dir}")
    return molecules, row_key
