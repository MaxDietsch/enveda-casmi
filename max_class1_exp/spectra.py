"""Spectrum cleaning and similarity for the stage-1 library search.

Cleaning follows Li et al. (Nature Methods 2021, "Spectral entropy outperforms
MS/MS dot product similarity"): drop the precursor region, centroid peaks
closer than the m/z tolerance, remove noise below 1 % of the base peak,
normalise intensities to sum 1.

Similarities, all computed as one query spectrum against many library spectra
at once (vectorised with numpy):

  entropy    weighted spectral entropy similarity (the default score)
  cosine     plain cosine on the same cleaned peaks (reference baseline)

Entropy similarity of two spectra A, B (intensities summing to 1) is
    sum over matched peak pairs of [(a+b) ln(a+b) - a ln a - b ln b] / ln 4,
which equals 1 - (2 S(AB) - S(A) - S(B)) / ln 4 with S the Shannon entropy
and AB the 50/50 mix of both spectra. It is 1 for identical spectra and 0 for
spectra without shared peaks.
"""

from dataclasses import dataclass

import numpy as np

LN4 = np.log(4.0)


@dataclass(frozen=True)
class CleanParams:
    tol: float = 0.02               # m/z tolerance (Da) for centroiding and matching
    noise: float = 0.01             # drop peaks below this fraction of the base peak
    precursor_margin: float = 1.6   # drop peaks with m/z > precursor_mz - margin
    max_peaks: int = 0              # keep only the N most intense peaks (0 = all)


# ------------------------------------------------------------------ cleaning

def centroid(mz: np.ndarray, inten: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Merge peaks closer than tol, greedily from the most intense peak (Li et al.).

    mz must be sorted. Each merged peak gets the summed intensity and the
    intensity-weighted mean m/z. Only groups of peaks that are chained by
    gaps < tol are processed in Python; isolated peaks pass through.
    """
    if len(mz) < 2:
        return mz, inten
    gaps = np.diff(mz) < tol
    if not gaps.any():
        return mz, inten
    # single-linkage groups of close peaks
    group = np.concatenate([[0], np.cumsum(~gaps)])
    out_mz, out_int = [], []
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    ends = np.r_[starts[1:], len(mz)]
    for s, e in zip(starts, ends):
        if e - s == 1:
            out_mz.append(mz[s])
            out_int.append(inten[s])
            continue
        gm, gi = mz[s:e], inten[s:e]
        used = np.zeros(e - s, dtype=bool)
        for i in np.argsort(-gi, kind="stable"):
            if used[i]:
                continue
            near = (~used) & (np.abs(gm - gm[i]) <= tol)
            used |= near
            w = gi[near]
            out_mz.append(float((gm[near] * w).sum() / w.sum()))
            out_int.append(float(w.sum()))
    out_mz = np.asarray(out_mz, dtype=np.float64)
    out_int = np.asarray(out_int, dtype=np.float64)
    order = np.argsort(out_mz, kind="stable")
    return out_mz[order], out_int[order]


def clean_spectrum(mz, inten, precursor_mz: float, p: CleanParams = CleanParams()) -> tuple[np.ndarray, np.ndarray]:
    """Cleaned spectrum (sorted m/z, intensities summing to 1). May be empty."""
    mz = np.asarray(mz if mz is not None else [], dtype=np.float64)
    inten = np.asarray(inten if inten is not None else [], dtype=np.float64)
    keep = np.isfinite(mz) & np.isfinite(inten) & (inten > 0)
    if precursor_mz is not None and np.isfinite(precursor_mz) and p.precursor_margin is not None:
        keep &= mz <= precursor_mz - p.precursor_margin
    mz, inten = mz[keep], inten[keep]
    if len(mz) == 0:
        return mz, inten
    order = np.argsort(mz, kind="stable")
    mz, inten = centroid(mz[order], inten[order], p.tol)
    keep = inten >= p.noise * inten.max()
    mz, inten = mz[keep], inten[keep]
    if p.max_peaks and len(mz) > p.max_peaks:
        top = np.sort(np.argsort(-inten, kind="stable")[: p.max_peaks])
        mz, inten = mz[top], inten[top]
    return mz, inten / inten.sum()


def entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def entropy_weight(p: np.ndarray) -> np.ndarray:
    """Li et al. weighting: if S < 3, raise intensities to 0.25 + 0.25 S, renormalise."""
    if len(p) == 0:
        return p
    s = entropy(p)
    if s < 3:
        p = p ** (0.25 + 0.25 * s)
        p = p / p.sum()
    return p


def unit_norm(p: np.ndarray) -> np.ndarray:
    n = np.sqrt((p * p).sum())
    return p / n if n > 0 else p


# ---------------------------------------------------------------- similarity

def _match(q_mz: np.ndarray, l_mz: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """For every library peak, the index of the nearest query peak within tol.

    Returns (library peak indices, query peak indices) of the matches.
    """
    if len(q_mz) == 0 or len(l_mz) == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e
    pos = np.searchsorted(q_mz, l_mz)
    left = np.clip(pos - 1, 0, len(q_mz) - 1)
    right = np.clip(pos, 0, len(q_mz) - 1)
    dl = np.abs(l_mz - q_mz[left])
    dr = np.abs(l_mz - q_mz[right])
    q_idx = np.where(dr < dl, right, left)
    d = np.minimum(dl, dr)
    hit = np.flatnonzero(d <= tol)
    return hit, q_idx[hit]


def _one_to_one(l_idx, q_idx, l_spec, contrib):
    """Keep at most one library peak per (library spectrum, query peak): the best one."""
    if len(l_idx) == 0:
        return l_idx, contrib
    key = l_spec[l_idx].astype(np.int64) * (int(q_idx.max()) + 1) + q_idx
    order = np.lexsort((-contrib, key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    sel = order[first]
    return l_idx[sel], contrib[sel]


def entropy_one_vs_many(q_mz, q_p, l_mz, l_p, l_spec, n_spec: int, tol: float) -> np.ndarray:
    """Entropy similarity of one query against n_spec library spectra.

    q_p and l_p must already be entropy-weighted and sum to 1 per spectrum.
    l_spec gives, for every library peak, its spectrum index in [0, n_spec).
    """
    l_idx, q_idx = _match(q_mz, l_mz, tol)
    if len(l_idx) == 0:
        return np.zeros(n_spec)
    a, b = q_p[q_idx], l_p[l_idx]
    ab = a + b
    contrib = ab * np.log(ab) - a * np.log(a) - b * np.log(b)
    l_idx, contrib = _one_to_one(l_idx, q_idx, l_spec, contrib)
    return np.bincount(l_spec[l_idx], weights=contrib, minlength=n_spec) / LN4


def cosine_one_vs_many(q_mz, q_u, l_mz, l_u, l_spec, n_spec: int, tol: float) -> np.ndarray:
    """Cosine similarity; q_u and l_u must have unit L2 norm per spectrum."""
    l_idx, q_idx = _match(q_mz, l_mz, tol)
    if len(l_idx) == 0:
        return np.zeros(n_spec)
    contrib = q_u[q_idx] * l_u[l_idx]
    l_idx, contrib = _one_to_one(l_idx, q_idx, l_spec, contrib)
    return np.bincount(l_spec[l_idx], weights=contrib, minlength=n_spec)


def entropy_similarity(mz_a, p_a, mz_b, p_b, tol: float, weighted: bool = True) -> float:
    """Pairwise convenience wrapper (inputs: cleaned spectra summing to 1)."""
    if weighted:
        p_a, p_b = entropy_weight(p_a), entropy_weight(p_b)
    return float(entropy_one_vs_many(mz_a, p_a, mz_b, p_b, np.zeros(len(mz_b), dtype=np.int64), 1, tol)[0])


def cosine_similarity(mz_a, p_a, mz_b, p_b, tol: float) -> float:
    return float(cosine_one_vs_many(mz_a, unit_norm(p_a), mz_b, unit_norm(p_b),
                                    np.zeros(len(mz_b), dtype=np.int64), 1, tol)[0])


def merge_spectra(spectra: list[tuple[np.ndarray, np.ndarray]], tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Composite spectrum: average of cleaned spectra (each sums to 1),
    re-centroided within tol, normalised to sum 1."""
    spectra = [(m, p) for m, p in spectra if len(m)]
    if not spectra:
        return np.empty(0), np.empty(0)
    mz = np.concatenate([m for m, _ in spectra])
    p = np.concatenate([q for _, q in spectra]) / len(spectra)
    order = np.argsort(mz, kind="stable")
    mz, p = centroid(mz[order], p[order], tol)
    return mz, p / p.sum()
