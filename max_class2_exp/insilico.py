"""The in-silico fragmentation channel (MetFrag / FingerID-style evidence).

The analogue channel borrows spectra from relatives; this channel asks the
question class 2 is really about: *would this candidate structure plausibly
produce the observed peaks?* For each (candidate SMILES, query adduct) the
repo's Fragmentor (src/enveda_casmi26/fragmentation.py - single cuts of
acyclic bonds, ring fissions, neutral-loss cascades, even-electron rule,
functional-group gating) enumerates theoretical fragment m/z values. The
observed query peaks are matched against them within --frag-ppm; the match
statistics (how many peaks, how much intensity, how close) become features.

Fragmentation is deterministic and cacheable per (smiles, adduct); the cache
is per process and survives across queries inside one worker, so a candidate
that appears in several shortlists is fragmented once per adduct.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_FRAG_CACHE: dict[tuple[str, str], np.ndarray | None] = {}
_MAX_CACHE = 200_000


def fragment_mzs(smiles: str, adduct: str) -> np.ndarray | None:
    """Sorted theoretical fragment m/z of (smiles, adduct), or None if the
    SMILES cannot be parsed (the features carry a frag_available flag)."""
    key = (smiles, adduct)
    if key in _FRAG_CACHE:
        return _FRAG_CACHE[key]
    try:
        from enveda_casmi26.fragmentation import Fragmentor
        theo = np.sort(np.array([f.mz for f in Fragmentor(smiles, adduct=adduct).enumerate_fragments()],
                                dtype=float))
    except Exception:
        theo = None
    if len(_FRAG_CACHE) >= _MAX_CACHE:
        _FRAG_CACHE.clear()
    _FRAG_CACHE[key] = theo
    return theo


def frag_match(q_mz: np.ndarray, q_p: np.ndarray, theo: np.ndarray | None, ppm: float) -> tuple[int, float, float]:
    """Match a query spectrum's peaks against theoretical fragment m/z values.

    Returns (matched peak count, matched query intensity share, mean |ppm| of
    the best-matching theoretical fragments). A peak matches if any
    theoretical fragment lies within the ppm tolerance (each query peak is
    counted at most once); tolerance scales with m/z.
    """
    q_mz = np.asarray(q_mz, dtype=float)
    q_p = np.asarray(q_p, dtype=float)
    if theo is None or len(theo) == 0 or len(q_mz) == 0:
        return 0, 0.0, 0.0
    pos = np.searchsorted(theo, q_mz)
    left = np.clip(pos - 1, 0, len(theo) - 1)
    right = np.clip(pos, 0, len(theo) - 1)
    dl = np.abs(q_mz - theo[left])
    dr = np.abs(q_mz - theo[right])
    d = np.minimum(dl, dr)
    hit = d <= np.maximum(q_mz * ppm * 1e-6, 1e-9)
    n = int(hit.sum())
    if n == 0:
        return 0, 0.0, 0.0
    return n, float(q_p[hit].sum()), float((d[hit] / q_mz[hit] * 1e6).mean())
