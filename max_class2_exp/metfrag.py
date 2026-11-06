"""The MetFrag-lite fragmentation channel (forum finding P1.1).

The repo Fragmentor (channel F, `insilico.py`) is chemistry-rich: single cuts
of acyclic bonds, ring fissions, neutral-loss cascades, the even-electron
rule, functional-group gating. The forum's best-evidenced class-2 feature
block is deliberately *dumber* — NeckBeard's paired t-tests on a 241-molecule
isomer panel (`742055`) and Aryan's 0.502 -> 0.627 result on the
no-public-spectra split (`749251`) both come from:

  * 1-bond and 2-bond cuts only (3-bond cuts: -0.031, rejected);
  * 2-bond cuts down-weighted x0.6 vs 1-bond;
  * H shifts -2..+3 on each fragment;
  * tolerance 0.005-0.01 Da;
  * LINEAR intensity weighting (not sqrt);
  * explained intensity expressed as a *gap to the best candidate* in the
    same molecule;
  * rarity weighting: a peak every candidate explains is weak evidence.

Parsimony is the signal ("how few cleavages does this candidate need"), not
hand-written cleavage chemistry (propensity weights: -0.002; reachability
penalties: -0.015; both rejected on the forum).

Enumeration convention: cutting a bond leaves each fragment as the radical
subgraph with its ORIGINAL hydrogen counts (mass = sum of the atoms' exact
monoisotopic masses + implicit/explicit H); the enumerated ion m/z values
are  frag_mass + h*H (h in -2..+3)  +/-  proton mass (positive/negative
polarity), i.e. every plausible H-transfer convention is covered by the
shift range, and no cleavage-propensity bookkeeping is attempted.

Weights: 1-bond-cut fragments score 1.0, 2-bond-cut fragments 0.6. Where
several cuts give the same m/z, the max weight is kept. The theoretical set
is cached per (SMILES, polarity) per process (adduct only flips the proton
sign, which the polarity covers for the protonated/deprotonated convention).
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

H_MASS = 1.00782503223       # exact mass of ^1H
PROTON = 1.007276466621       # [frag+H]+ / [frag-H]- charge carrier
MZ_FLOOR = 25.0               # drop junk tiny ions (H3O+, NH4+, ...)
H_SHIFTS = tuple(range(-2, 4))     # -2..+3 (NeckBeard: better than -2..+2, +0.010)
W_TWO_CUT = 0.6                    # 2-bond cuts down-weighted (NeckBeard: +0.030)
MAX_BONDS_FOR_PAIRS = 64           # cap O(B^2) enumeration (large molecules only)

# exact monoisotopic masses by atomic number (CHNOPS + halogens + a few);
# anything else falls back to RDKit's average atomic weight
_EXACT = {1: 1.00782503223, 5: 11.00930536, 6: 12.0, 7: 14.00307400443,
          8: 15.99491461957, 9: 18.99840322, 11: 22.98976928, 12: 24.305,
          14: 27.97692653465, 15: 30.97376199842, 16: 31.9720711744,
          17: 34.968852682, 19: 38.9637064864, 20: 39.962590983, 26: 55.93493633,
          30: 63.9291465, 33: 78.9183376, 34: 79.9165218, 35: 78.9183376,
          53: 126.9044719, 78: 151.904305}

_MTF_CACHE: dict[tuple[str, bool], tuple[np.ndarray, np.ndarray] | None] = {}
_MAX_CACHE = 100_000


def _atom_masses(mol) -> np.ndarray | None:
    """Per-atom exact mass incl. its hydrogens (radical-fragment baseline)."""
    from rdkit import Chem
    pt = Chem.GetPeriodicTable()
    out = np.zeros(mol.GetNumAtoms())
    for a in mol.GetAtoms():
        z = a.GetAtomicNum()
        if z == 0:
            return None                      # wildcard/dummy: give up on this SMILES
        heavy = _EXACT.get(z)
        if heavy is None:
            heavy = pt.GetAtomicWeight(z)    # approximate fallback (rare elements)
        out[a.GetIdx()] = heavy + a.GetTotalNumHs() * H_MASS
    return out


def _components_after_cut(mol, bond_indices: list[int]) -> list[np.ndarray]:
    """Heavy-atom connected components after cutting the given bonds."""
    from rdkit import Chem
    n = mol.GetNumAtoms()
    frag = Chem.FragmentOnBonds(mol, bond_indices, addDummies=False)
    comps = Chem.GetMolFrags(frag, asMols=False, sanitizeFrags=False)
    return [np.fromiter((i for i in c if i < n), dtype=np.int64, count=len(c)) for c in comps]


def metfrag_mzs(smiles: str, positive: bool) -> tuple[np.ndarray, np.ndarray] | None:
    """(sorted theoretical m/z, per-mz weight) of a SMILES for one polarity.

    Weight 1.0 = 1-bond cut (or the intact molecule), 0.6 = 2-bond cut.
    None if the SMILES cannot be parsed. Cached per (smiles, polarity).
    """
    key = (smiles, positive)
    if key in _MTF_CACHE:
        return _MTF_CACHE[key]
    out = None
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is not None and mol.GetNumAtoms() >= 2:
            masses = _atom_masses(mol)
            if masses is not None:
                total = float(masses.sum())
                bonds = [b.GetIdx() for b in mol.GetBonds()]
                mz, w = [], []
                ion = PROTON if positive else -PROTON

                def _add(frag_mass: float, weight: float):
                    for h in H_SHIFTS:
                        v = frag_mass + h * H_MASS + ion
                        if v >= MZ_FLOOR:
                            mz.append(v)
                            w.append(weight)

                _add(total, 1.0)              # the intact (pseudo-precursor) ion
                for b in bonds:               # 1-bond cuts
                    for comp in _components_after_cut(mol, [b]):
                        if len(comp):
                            _add(float(masses[comp].sum()), 1.0)
                if len(bonds) <= MAX_BONDS_FOR_PAIRS:
                    for i in range(len(bonds)):        # 2-bond cuts (down-weighted)
                        for j in range(i + 1, len(bonds)):
                            for comp in _components_after_cut(mol, [bonds[i], bonds[j]]):
                                if len(comp):
                                    _add(float(masses[comp].sum()), W_TWO_CUT)
                if mz:
                    mz_a, w_a = np.asarray(mz), np.asarray(w)
                    order = np.argsort(mz_a)            # merge m/z closer than 1e-4 Da,
                    mz_a, w_a = mz_a[order], w_a[order]  # keeping the max weight
                    keep = np.r_[True, np.diff(mz_a) > 1e-4]
                    uniq = mz_a[keep]
                    w_u = np.maximum.reduceat(w_a, np.flatnonzero(keep))
                    out = (uniq, w_u)
    except Exception:
        out = None
    if len(_MTF_CACHE) >= _MAX_CACHE:
        _MTF_CACHE.clear()
    _MTF_CACHE[key] = out
    return out


def metfrag_match(q_mz: np.ndarray, theo_mz: np.ndarray, w: np.ndarray, tol_da: float):
    """Match query peaks against the theoretical set.

    Returns (hit mask, weight of the matched theoretical peak, |Da| distance)
    per query peak; a peak matches if the nearest theoretical m/z lies within
    tol_da. If both neighbours lie within tol_da the higher weight wins.
    """
    q_mz = np.asarray(q_mz, dtype=float)
    if theo_mz is None or len(theo_mz) == 0 or len(q_mz) == 0:
        return (np.zeros(len(q_mz), dtype=bool), np.zeros(len(q_mz)), np.zeros(len(q_mz)))
    n = len(theo_mz)
    pos = np.searchsorted(theo_mz, q_mz)
    left = np.clip(pos - 1, 0, n - 1)
    right = np.clip(pos, 0, n - 1)
    # at/outside the range ends both clip onto the same element; widen the
    # pair toward the interior so a second within-tolerance theoretical peak
    # still takes part in the "both within tol -> higher weight" upgrade
    collapse = left == right
    right = np.where(collapse & (pos == 0), np.minimum(1, n - 1), right)
    left = np.where(collapse & (pos > 0), np.maximum(n - 2, 0), left)
    dl, dr = np.abs(q_mz - theo_mz[left]), np.abs(q_mz - theo_mz[right])
    take_l = dl <= dr
    d = np.where(take_l, dl, dr)
    hit = d <= tol_da
    # weight of the matched peak: the nearest one, upgraded to the other
    # neighbour's weight only when that one is also within tolerance
    near_w = np.where(take_l, w[left], w[right])
    other_w = np.where(take_l, w[right], w[left])
    other_d = np.where(take_l, dr, dl)
    wv = np.where(other_d <= tol_da, np.maximum(near_w, other_w), near_w)
    return hit, wv, d
