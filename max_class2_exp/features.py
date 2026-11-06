"""Features for ranking structures, and the small model that fuses them.

One featurize function (rule R4) builds every (query molecule, candidate)
pair's features, for training and evaluation alike. Evidence channels plus
bookkeeping:

  analogue channel    the candidate's top-K nearest *visible* molecules stand
                       in for its missing spectra: entropy/cosine similarity
                       of the query against their spectra (best match, and a
                       Tanimoto-weighted merged spectrum per candidate), plus
                       neutral-loss matching. A visible candidate's own
                       molecule is among its neighbours at Tc 1.0 - evidence
                       that it *was* measured; the truth never has that
                       (all its rows are hidden), and the re-ranker learns
                       the asymmetry from training examples built the same way.
                       OFF by default (forum P1.7: saturated baseline; measure
                       its marginal contribution, don't lean on it).

  in-silico channel   would the candidate structure produce the observed
                       peaks? The repo Fragmentor enumerates its theoretical
                       fragments for the query's adduct; matched peaks,
                       explained intensity and ppm closeness become features.

  metfrag-lite         the forum's simplest and best-evidenced class-2 block
                       (P1.1): 1-bond cuts, 2-bond cuts down-weighted x0.6,
                       H shifts -2..+3, linear intensity, rarity-weighted
                       explained intensity, and the gap to the best candidate
                       in the same shortlist. See metfrag.py.

  fingerprint channel  spectrum -> predicted 2048-bit Morgan fingerprint
  (fpchannel.py)       (v1 linear on the fly, or the frozen de-novo encoder
                       + head), scored against each candidate's actual bits:
                       prior-normalised log-likelihood + Tanimoto. Built per
                       run in stage2 (fpchannel.build_fp_model), appended by
                       append_channel_blocks - leak-free by construction
                       (v1 trains on split-visible rows only) or by guard
                       (v2 checkpoints must prove their exclusions).

  neural in-silico     ICEBERG's predicted spectrum of the candidate
  (channel F2,         (adduct/CE conditioned, zero-shot public weights, run
  f2_bridge.py)        in the ms-pred venv via a subprocess + persistent
                       prediction cache) vs the observed spectrum: matched
                       cosine, best and mean over spectra. Off when the env
                       or weights are absent (features stay 0).

  context              mass error (median and worst-spectrum), candidate
                       visibility counts, source flags, same-formula
                       competition, shortlist size (context only).

The re-ranker is the class-1 recipe: z-score continuous features within each
shortlist (the contrast_within_set trick), then logistic regression and/or
HistGradientBoosting for P(candidate is the truth). No neural network.
HGB is trained with several seeds and averaged (forum `742088`: identical
pipelines differed 0.292 vs 0.298 by seed alone - pin and average).

Ranking/eval helpers shared with stage2.py: failure buckets (P0.3), the
same-formula panel rank (P0.2), rank-1 protection (P1.5) and the paired
bootstrap (P0.6).
"""

import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT / "dataset-creation"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import common  # noqa: E402
from fpchannel import ce_bucket  # noqa: E402
from spectra_utils import (centroid, cosine_one_vs_many, entropy_one_vs_many,  # noqa: E402
                           entropy_weight, matched_cosine_one_vs_many_ppm,
                           neutral_losses, unit_norm)
from insilico import frag_match, fragment_mzs  # noqa: E402
from metfrag import metfrag_match, metfrag_mzs  # noqa: E402

# feature columns; n_candidates is context only (constant per shortlist) and never a model feature
FEATURES = [
    "entropy_best", "entropy_composite", "cosine_best", "nl_entropy_best",
    "analogue_best_tc", "n_analogue_molecules", "n_analogue_spectra",
    "frag_matched_peaks", "frag_explained_intensity", "frag_mean_ppm",
    "n_theo_frags", "frag_available",
    "mtf_matched_peaks", "mtf_explained_linear", "mtf_rarity_explained",
    "mtf_gap_best", "mtf_mean_ppm", "mtf_available",
    "mass_error_ppm", "mass_error_ppm_max",
    "cand_visible_rows", "cand_tims_rows", "cand_public_rows",
    "cand_is_addition", "cand_is_external", "cand_n_same_formula",
    "fp_ll", "fp_tanimoto", "fp_available",
    "f2_cosine_best", "f2_cosine_mean", "f2_available",
]
FEATURE_COLUMNS = FEATURES + ["n_candidates"]
FIDX = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
RAW_FEATURES = ("frag_available", "mtf_available", "cand_is_addition", "cand_is_external",
                "fp_available", "f2_available")
TOP_K = 25
F2_COS_PPM = 20.0        # matched-cosine ppm tolerance vs predicted spectra (f2_eval default)
BUCKETS = ("rank_1", "rank_2_25", "in_window_rank_gt25", "outside_window")

# ablation (forum P1.3: source flags must be ablatable) - drop model features
# by name; they stay in X so saved feature frames remain comparable.
#
# The visibility/source block is dropped BY DEFAULT: within the local
# simulator every eval truth is a spectrum-less addition (all rows hidden,
# re-added via library_additions), so cand_is_addition / cand_visible_rows
# mark the truth with MRR ~1.0 - the ranker would measure the simulator's
# construction rule, not the chemistry, and the number would not transfer
# to the Kaggle pool where most decoys also lack spectra (FORUM_FINDINGS.md
# P1.3 "context, not a truth shortcut"; P2.2: such flags looked great on
# local holdouts and hurt on the LB). --keep-source-features opts back in.
SOURCE_FEATURES = ("cand_is_addition", "cand_visible_rows",
                   "cand_tims_rows", "cand_public_rows")
_DROPPED: frozenset[str] = frozenset(SOURCE_FEATURES)

# state shared with forked workers
_POOL = None
_NBRS = None
_LIB = None
_QUERIES = None
_TOL = 0.01
_FRAG_PPM = 10.0
_MTF_TOL = 0.01
_COMPOSITE: dict = {}


def set_dropped_features(names, keep_source: bool = False, replace: bool = False) -> None:
    """Ablate features by name for the model matrix (fit and predict alike).

    Defaults to also dropping SOURCE_FEATURES; keep_source includes them;
    replace sets exactly `names` (used when a saved model's training-time
    drops are authoritative)."""
    global _DROPPED
    names = [n for n in (names or []) if n]
    bad = [n for n in names if n not in FEATURES]
    if bad:
        raise ValueError(f"unknown features to drop: {bad}")
    d = set(names)
    if not keep_source and not replace:
        d |= set(SOURCE_FEATURES)
    _DROPPED = frozenset(d)


def dropped_features() -> frozenset[str]:
    return _DROPPED


# --------------------------------------------------------------- featurizing

def _composite(member: int, positive: bool, kept, lib, tol: float):
    """Tanimoto-weighted, merged, entropy-weighted spectrum of the candidate's
    kept neighbour spectra in one polarity (cached per worker, keyed by
    library identity because training and evaluation libraries differ).

    kept: (neighbour molecule indices with spectra, Tanimotos) of this member
    under the *current* library - callers pass the same kept lists that built
    the analogue block, so cache and block can never disagree."""
    k = (id(lib), member, positive)
    v = _COMPOSITE.get(k)
    if v is None:
        mols, tcs = kept
        mz_all, p_all = [], []
        for m, tc in zip(mols, tcs):
            s, e = lib.spectra_of(int(m))
            for i in range(s, e):
                if lib.positive[i] == positive:
                    mz, p = lib.get(i)
                    mz_all.append(mz)
                    p_all.append(p * tc)
        if mz_all:
            mz = np.concatenate(mz_all)
            p = np.concatenate(p_all)
            order = np.argsort(mz, kind="stable")
            mz, p = centroid(mz[order], p[order], tol)
            p = p / p.sum() if p.sum() > 0 else p
            v = (mz, entropy_weight(p))
        else:
            v = (np.empty(0), np.empty(0))
        _COMPOSITE[k] = v
    return v


def _spectrum_masses(spectra) -> np.ndarray:
    """Implied neutral mass per spectrum (nan where unusable), from the
    (…, precursor_mz, adduct[, ce_ev]) tuples; aligned with the spectra list."""
    from enveda_casmi26.pipeline import neutral_mass
    return np.array([neutral_mass(float(s[3]), s[4]) for s in spectra], dtype=float)


def featurize(spectra, cands, abs_ppm, pool, nbrs, lib, tol: float, frag_ppm: float,
              mtf_tol: float = 0.01) -> np.ndarray:
    """One feature row per candidate (R4: the one function for training and
    evaluation alike).

    spectra: list of (mz, p, positive, precursor_mz, adduct[, ce_ev]) with
    mz/p from clean_spectrum (p sums to 1); the optional 6th element
    (collision energy, eV) is read by the channel-P/F2 block, not here.
    cands: pool member indices. abs_ppm: signed
    mass error per candidate (vs the query's median implied mass). nbrs:
    {member: (visible neighbour molecule indices, Tanimotos)} - only
    neighbours that have spectra count. Returns (n_cands, len(FEATURE_COLUMNS)).
    """
    n = len(cands)
    X = np.zeros((n, len(FEATURE_COLUMNS)))
    if n == 0:
        return X
    X[:, FIDX["n_candidates"]] = n
    X[:, FIDX["mass_error_ppm"]] = abs_ppm
    X[:, FIDX["cand_visible_rows"]] = np.log1p(pool.visible_rows[cands])
    tims_rows = getattr(pool, "tims_rows", None)
    public_rows = getattr(pool, "public_rows", None)
    if tims_rows is not None:
        X[:, FIDX["cand_tims_rows"]] = np.log1p(np.maximum(tims_rows[cands], 0.0))
    if public_rows is not None:
        X[:, FIDX["cand_public_rows"]] = np.log1p(np.maximum(public_rows[cands], 0.0))
    X[:, FIDX["cand_is_addition"]] = pool.is_addition[cands]
    X[:, FIDX["cand_is_external"]] = pool.is_external[cands]
    formulas = getattr(pool, "formulas", None)
    if formulas is not None:
        # same-formula competition: how many OTHER shortlist candidates share
        # this candidate's formula (the class-2 deciding axis, P0.2/P1.3)
        _, inv = np.unique(formulas[cands].astype(str), return_inverse=True)
        X[:, FIDX["cand_n_same_formula"]] = np.bincount(inv)[inv] - 1
    # worst-spectrum mass error: the candidate must fit EVERY spectrum's mass
    sm = _spectrum_masses(spectra)
    ok = np.isfinite(sm)
    if ok.any():
        errs = (pool.mass[cands][:, None] - sm[None, :]) / sm[None, :] * 1e6
        X[:, FIDX["mass_error_ppm_max"]] = np.abs(errs[:, ok]).max(axis=1)

    # ---- analogue neighbourhood: keep only neighbours that have spectra
    empty = np.empty(0, dtype=np.int64)
    kept: dict = {}                               # member -> (mols with spectra, tcs)
    n_mol_feat, n_spec_feat, tc_feat = np.zeros(n), np.zeros(n), np.zeros(n)
    ranges = []                                   # spectrum slots per candidate
    for j, c in enumerate(cands):
        mols, tcs = nbrs.get(int(c), (empty, np.empty(0)))
        rm, rt, rr = [], [], []
        for m, tc in zip(mols, tcs):
            s, e = lib.spectra_of(int(m))
            if e > s:
                rm.append(int(m))
                rt.append(float(tc))
                rr.append((s, e))
        n_mol_feat[j] = len(rm)
        n_spec_feat[j] = sum(e - s for s, e in rr)
        tc_feat[j] = max(rt) if rt else 0.0
        kept[int(c)] = (rm, rt)
        ranges.append(rr)
    X[:, FIDX["n_analogue_molecules"]] = n_mol_feat
    X[:, FIDX["n_analogue_spectra"]] = n_spec_feat
    X[:, FIDX["analogue_best_tc"]] = tc_feat

    # ---- one contiguous peak block over all candidates' neighbour spectra.
    # Candidates without any (the truth never has) get no slots and stay at 0.
    # n_slots counts SPECTRA (a neighbour molecule can own many), matching the
    # spec_idx construction, so the reduceat boundaries below are aligned.
    n_slots = np.array([sum(e - s for s, e in r) for r in ranges], dtype=np.int64)
    flat = [(s, e) for r in ranges for s, e in r]
    spec_idx = np.concatenate([np.arange(s, e) for s, e in flat]) if flat \
        else np.empty(0, dtype=np.int64)
    peak_idx = np.concatenate([np.arange(lib.peak_start[s], lib.peak_start[e]) for s, e in flat]) \
        if flat else np.empty(0, dtype=np.int64)
    local_spec = np.repeat(np.arange(len(spec_idx)), np.diff(lib.peak_start)[spec_idx]) \
        if len(spec_idx) else np.empty(0, dtype=np.int64)
    l_mz, l_pw, l_pu, l_nl = lib.mz[peak_idx], lib.pw[peak_idx], lib.pu[peak_idx], lib.nlmz[peak_idx]
    with_slots = np.flatnonzero(n_slots)                              # reduceat over real slots
    starts = np.cumsum(n_slots) - n_slots

    def _best(per_slot: np.ndarray) -> np.ndarray:
        out = np.zeros(n)
        if len(with_slots):
            out[with_slots] = np.maximum.reduceat(per_slot, starts[with_slots])
        return out

    acc = {k: np.zeros(n) for k in ("entropy_best", "entropy_composite", "cosine_best",
                                    "nl_entropy_best", "frag_matched_peaks",
                                    "frag_explained_intensity", "frag_mean_ppm",
                                    "n_theo_frags", "frag_available",
                                    "mtf_matched_peaks", "mtf_explained_linear",
                                    "mtf_rarity_explained", "mtf_gap_best",
                                    "mtf_mean_ppm", "mtf_available")}
    used = 0
    for spec in spectra:                          # tolerant: a 6th ce_ev element is allowed
        mz, p, positive, prec, adduct = spec[:5]
        if len(mz) == 0:
            continue
        used += 1
        pw, pu = entropy_weight(p), unit_norm(p)
        if len(spec_idx):
            se = entropy_one_vs_many(mz, pw, l_mz, l_pw, local_spec, len(spec_idx), tol)
            sc = cosine_one_vs_many(mz, pu, l_mz, l_pu, local_spec, len(spec_idx), tol)
            nl, nl_ord = neutral_losses(mz, prec)
            nle = entropy_one_vs_many(nl, pw[nl_ord], l_nl, l_pw, local_spec, len(spec_idx), tol)
            acc["entropy_best"] += _best(se)
            acc["cosine_best"] += _best(sc)
            acc["nl_entropy_best"] += _best(nle)
        comps = [_composite(int(c), positive, kept[int(c)], lib, tol) for c in cands]
        c_lens = np.array([len(x[0]) for x in comps])
        if c_lens.sum():
            c_mz = np.concatenate([x[0] for x in comps])
            c_pw = np.concatenate([x[1] for x in comps])
            acc["entropy_composite"] += entropy_one_vs_many(
                mz, pw, c_mz, c_pw, np.repeat(np.arange(n), c_lens), n, tol)
        # in-silico channel: per candidate, for this spectrum's adduct
        for j, c in enumerate(cands):
            theo = fragment_mzs(pool.smiles[c], adduct)
            nm, ex, mp = frag_match(mz, p, theo, frag_ppm)
            acc["frag_matched_peaks"][j] += nm
            acc["frag_explained_intensity"][j] += ex
            acc["frag_mean_ppm"][j] += mp
            if theo is not None and len(theo):
                acc["frag_available"][j] += 1
                acc["n_theo_frags"][j] += len(theo)
        # metfrag-lite channel (P1.1): linear intensity, rarity across the
        # shortlist, per-cut weights; gap-to-best is accumulated after the loop
        masks = np.zeros((n, len(mz)), dtype=bool)
        wbest = np.zeros((n, len(mz)))
        dbest = np.zeros((n, len(mz)))
        for j, c in enumerate(cands):
            theo = metfrag_mzs(pool.smiles[c], positive)
            if theo is not None and len(theo[0]):
                hit, wv, d = metfrag_match(mz, theo[0], theo[1], mtf_tol)
                masks[j], wbest[j], dbest[j] = hit, wv, d
                acc["mtf_available"][j] += 1
        acc["mtf_matched_peaks"] += masks.sum(axis=1)
        explained = masks * wbest @ p                     # linear intensity x cut weight
        acc["mtf_explained_linear"] += explained
        rarity = 1.0 / (1.0 + masks.sum(axis=0))          # peak explained by everyone: weak
        acc["mtf_rarity_explained"] += (masks * (wbest * rarity)) @ p
        cnt = np.maximum(masks.sum(axis=1), 1)
        acc["mtf_mean_ppm"] += np.where(masks, dbest / np.maximum(mz, 1e-9) * 1e6, 0.0).sum(axis=1) / cnt
        acc["mtf_gap_best"] += explained - explained.max()
    if used:
        for k, v in acc.items():
            X[:, FIDX[k]] = v / used
        X[:, FIDX["n_theo_frags"]] = np.log1p(X[:, FIDX["n_theo_frags"]])
    return X


def _featurize_one(i: int) -> np.ndarray:
    q = _QUERIES[i]
    return featurize(q["spectra"], q["cands"], np.abs(q["ppm_err"]), _POOL, _NBRS,
                     _LIB, _TOL, _FRAG_PPM, _MTF_TOL)


def featurize_queries(queries, pool, nbrs, lib, tol, frag_ppm, workers, label,
                      mtf_tol: float = 0.01) -> list[np.ndarray]:
    """Featurize many queries in parallel (fork: pool/nbrs/lib are shared COW)."""
    global _POOL, _NBRS, _LIB, _QUERIES, _TOL, _FRAG_PPM, _MTF_TOL
    _POOL, _NBRS, _LIB, _QUERIES = pool, nbrs, lib, queries
    _TOL, _FRAG_PPM, _MTF_TOL = tol, frag_ppm, mtf_tol
    _COMPOSITE.clear()
    out = []
    if not queries:
        return out
    with Pool(min(workers, len(queries))) as p:
        prog = common.Progress(label, len(queries))
        for X in p.imap(_featurize_one, range(len(queries)), chunksize=4):
            out.append(X)
            prog.update()
        prog.close()
    return out


# ---------------------------------------- channels P (fingerprint) and F2

def channel_block(q, pool, fp_mean, priors, fps_all, f2_preds, cos_ppm: float):
    """The six channel-P/F2 columns for one query's shortlist (R4: the same
    function for training shortlists, eval queries and the blind control).

    fp_mean: the query's mean predicted fingerprint probabilities (2048,)
    from the run's channel-P model, or None. priors: the model's per-bit
    training-molecule on-rates. f2_preds: {(smiles, adduct, ce_bucket):
    (mz, p, ||p||^2)} from the F2 bridge. Returns (n_candidates, 6):
    [fp_ll, fp_tanimoto, fp_available, f2_cosine_best, f2_cosine_mean,
    f2_available] — all zero when a channel is off (the fusion keeps
    working on the remaining channels)."""
    n = len(q["cands"])
    out = np.zeros((n, 6))
    if n == 0:
        return out
    cands = np.asarray(q["cands"], dtype=np.int64)
    # ---- channel P: predicted fingerprint vs each candidate's actual bits
    if fp_mean is not None and priors is not None and len(fps_all):
        bits = np.unpackbits(np.asarray(fps_all[pool.struct[cands]]).view(np.uint8),
                             axis=1).astype(bool)          # (n, 2048)
        p = np.clip(fp_mean, 1e-4, 1.0 - 1e-4)
        pri = np.clip(priors, 1e-3, 1.0 - 1e-3)
        out[:, 0] = np.where(bits, np.log(p / pri),
                             np.log((1.0 - p) / (1.0 - pri))).sum(axis=1)
        pred = p > 0.5
        inter = (bits & pred).sum(axis=1)
        union = (bits | pred).sum(axis=1)
        out[:, 1] = inter / np.maximum(union, 1)
        out[:, 2] = 1.0
    # ---- channel F2: observed spectrum vs each candidate's predicted spectrum
    if f2_preds:
        best = np.zeros(n)
        tot = np.zeros(n)
        avail = np.zeros(n)
        n_spec = sum(1 for s in q["spectra"] if len(s[0]))
        if n_spec:
            smiles = [str(pool.smiles[c]) for c in cands]
            for spec in q["spectra"]:
                mz, p = spec[0], spec[1]
                if len(mz) == 0:
                    continue
                adduct = str(spec[4])
                ce = ce_bucket(spec[5]) if len(spec) > 5 else ce_bucket(None)
                which, pm, pp, pn = [], [], [], []
                for j, s in enumerate(smiles):
                    pr = f2_preds.get((s, adduct, ce))
                    if pr is not None and len(pr[0]):
                        which.append(j)
                        pm.append(pr[0])
                        pp.append(pr[1])
                        pn.append(pr[2])
                if which:
                    local = np.repeat(np.arange(len(which)),
                                     [len(m) for m in pm]).astype(np.int64)
                    cos = matched_cosine_one_vs_many_ppm(
                        mz, p, np.concatenate(pm), np.concatenate(pp),
                        local, np.asarray(pn), len(which), cos_ppm)
                    tot[which] += cos
                    best[which] = np.maximum(best[which], cos)
                    avail[which] += 1
            out[:, 3] = best
            out[:, 4] = tot / n_spec
            out[:, 5] = avail / n_spec
    return out


def append_channel_blocks(queries, Xs, pool, fps_all, fp_model, f2_preds, label,
                          cos_ppm: float = F2_COS_PPM) -> None:
    """Fill the channel-P/F2 columns of already-featurized shortlists, in
    place. R4 holds by construction: training shortlists, eval queries and
    the blind control all go through `channel_block`.

    The fingerprint predictions are batched up front — one encoder pass per
    spectrum for v2 — then the per-candidate scoring is cheap numpy. No
    forking: torch CPU inference after fork is unsafe, and the batched pass
    is faster than per-query workers anyway."""
    if not queries or (fp_model is None and not f2_preds):
        return
    t0 = time.time()
    priors = fp_model.priors if fp_model is not None else None
    probs = fp_model.predict_proba_batch([q["spectra"] for q in queries]) \
        if fp_model is not None else [None] * len(queries)
    prog = common.Progress(label, len(queries))
    for i, (q, X) in enumerate(zip(queries, Xs)):
        block = channel_block(q, pool, probs[i], priors, fps_all, f2_preds, cos_ppm)
        X[:, FIDX["fp_ll"]] = block[:, 0]
        X[:, FIDX["fp_tanimoto"]] = block[:, 1]
        X[:, FIDX["fp_available"]] = block[:, 2]
        X[:, FIDX["f2_cosine_best"]] = block[:, 3]
        X[:, FIDX["f2_cosine_mean"]] = block[:, 4]
        X[:, FIDX["f2_available"]] = block[:, 5]
        prog.update()
    prog.close()
    common.log(f"{label}: {len(queries)} shortlists in {time.time() - t0:.0f}s")


# ------------------------------------------------------- model matrix / fit

def zscore_within_sets(X: np.ndarray, sizes) -> np.ndarray:
    """Z-score rows within each shortlist (the contrast_within_set trick):
    the model sees how a candidate compares to its competitors. Constant
    columns within a set become all zeros."""
    X = np.asarray(X, dtype=float).copy()
    sizes = np.asarray(sizes, dtype=np.int64)
    ends, starts = np.cumsum(sizes), np.cumsum(sizes) - sizes
    mu = np.add.reduceat(X, starts, axis=0) / sizes[:, None]
    m2 = np.add.reduceat(X * X, starts, axis=0) / sizes[:, None]
    sd = np.sqrt(np.maximum(m2 - mu * mu, 0.0))
    Z = np.zeros_like(X)
    for g, (s, e) in enumerate(zip(starts, ends)):
        d = X[s:e] - mu[g]
        Z[s:e] = np.divide(d, sd[g], out=np.zeros_like(d), where=sd[g] > 1e-12)
    return Z


def model_matrix(X: np.ndarray, sizes) -> np.ndarray:
    """Model input for one or many shortlists: FEATURES columns (n_candidates
    dropped, ablated features dropped), continuous features z-scored per
    shortlist, the raw flags left as they are (R4: same transform for fit and
    predict)."""
    cols = [c for c in FEATURES if c not in _DROPPED]
    Z = np.asarray(X, dtype=float)[:, [FIDX[c] for c in cols]].copy()
    zcols = [k for k, f in enumerate(cols) if f not in RAW_FEATURES]
    sizes = np.asarray(sizes, dtype=np.int64)
    ends, starts = np.cumsum(sizes), np.cumsum(sizes) - sizes
    sub = np.zeros((len(Z), len(zcols)))
    for g, (s, e) in enumerate(zip(starts, ends)):
        sub[s:e] = zscore_within_sets(Z[s:e][:, zcols], [e - s])
    Z[:, zcols] = sub
    return Z


class ModelEnsemble:
    """Seed-averaged predictions (forum: pin seeds and average). Uniform to a
    single sklearn model from predict()'s point of view."""

    def __init__(self, members: list):
        self.members = members

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        return np.mean([m.predict_proba(Z) for m in self.members], axis=0)

    def first(self):
        return self.members[0]

    def __len__(self) -> int:
        return len(self.members)


def fit_models(Z: np.ndarray, y: np.ndarray, seed: int, kinds, n_seeds: int = 1) -> dict:
    models = {}
    if "logistic" in kinds:
        from sklearn.linear_model import LogisticRegression
        models["logistic"] = ModelEnsemble([
            LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced").fit(Z, y)])
    if "hgb" in kinds:
        from sklearn.ensemble import HistGradientBoostingClassifier
        # early stopping needs a validation split with both classes; tiny runs
        # (unit tests, --limit smoke tests) fall back to plain boosting
        early = len(y) >= 200
        base = dict(max_depth=3, max_iter=300, early_stopping=early,
                    validation_fraction=0.1 if early else None,
                    n_iter_no_change=20 if early else None)
        base = {k: v for k, v in base.items() if v is not None}
        members = []
        for s in range(max(1, n_seeds)):
            try:
                members.append(HistGradientBoostingClassifier(
                    class_weight="balanced", random_state=seed + s, **base).fit(Z, y))
            except TypeError:       # sklearn without class_weight in HGB
                w = np.where(y == 1, len(y) / max(1, 2 * y.sum()),
                             len(y) / max(1, 2 * (len(y) - y.sum())))
                members.append(HistGradientBoostingClassifier(
                    random_state=seed + s, **base).fit(Z, y, sample_weight=w))
        models["hgb"] = ModelEnsemble(members)
    return models


def predict(model, X: np.ndarray, sizes) -> np.ndarray:
    return model.predict_proba(model_matrix(X, sizes))[:, 1]


def rank_order(scores: np.ndarray, abs_ppm: np.ndarray, cands: np.ndarray) -> np.ndarray:
    """Candidate order: score desc, then mass error asc, then member index."""
    return np.lexsort((np.asarray(cands), np.abs(abs_ppm), -scores))


def truth_rank(scores, q, t) -> tuple[int, float, float, float]:
    """Truth `t` (pool member index) vs its shortlist under `scores`.

    Returns (rank, rr@TOP_K, truth_score, top_score) with rank_order's
    tie-breaks (score desc, |ppm_err| asc, candidate index). rank is 0 and
    truth_score NaN when the truth is absent (t < 0 or not in q["cands"]);
    rr is 1/rank for rank <= TOP_K else 0."""
    cands = np.asarray(q["cands"])
    s = np.asarray(scores, dtype=float)
    if len(cands) == 0:
        return 0, 0.0, float("nan"), float("nan")
    order = rank_order(s, np.abs(np.asarray(q["ppm_err"], dtype=float)), cands)
    top = float(s[order[0]])
    if t is None or t < 0:
        return 0, 0.0, float("nan"), top
    hit = np.flatnonzero(cands == t)
    if len(hit) == 0:
        return 0, 0.0, float("nan"), top
    pos = int(hit[0])
    rank = int(np.flatnonzero(order == pos)[0]) + 1
    rr = 1.0 / rank if rank <= TOP_K else 0.0
    return rank, rr, float(s[pos]), top


# ------------------------------------------------------ ranking / eval helpers

def failure_bucket(rank: int, truth_in_candidates: bool) -> str:
    """Per-molecule failure ledger bucket (P0.3, after Pathik's `752735`):
    outside_window / in_window_rank_gt25 / rank_2_25 / rank_1."""
    if not truth_in_candidates:
        return "outside_window"
    if rank == 1:
        return "rank_1"
    if rank <= TOP_K:
        return "rank_2_25"
    return "in_window_rank_gt25"


def rank_within(scores: np.ndarray, positions: np.ndarray, truth_pos: int,
                tiebreak: np.ndarray | None = None) -> int:
    """Truth's rank when only the `positions` candidates compete (P0.2 panel).

    positions: sorted shortlist positions that form the panel (they must
    include truth_pos, else 0). Ties break like rank_order (score desc, then
    tiebreak asc, then position) so panel ranks are consistent with the main
    ranking."""
    if truth_pos < 0 or len(positions) == 0:
        return 0
    k = int(np.searchsorted(positions, truth_pos))
    if k >= len(positions) or int(positions[k]) != int(truth_pos):
        return 0                                     # truth not part of the panel
    s = np.asarray(scores)[positions]
    tb = np.zeros(len(positions)) if tiebreak is None else np.asarray(tiebreak)[positions]
    order = np.lexsort((positions, tb, -s))
    return int(np.flatnonzero(order == k)[0]) + 1


def protect_top1_order(fused_order: np.ndarray, protect_pos: int) -> np.ndarray:
    """Rank-1 protection (P1.5, `749412`'s cheapest validated trick): keep the
    protect channel's top-1 at rank 1, then the fused order (whose top-1 lands
    at rank 2 when it differs) - it cannot lose a rank-1 hit of the protect
    channel, at the price of never promoting the fused pick above it."""
    if protect_pos < 0 or len(fused_order) == 0 or fused_order[0] == protect_pos:
        return fused_order
    return np.array([protect_pos] + [int(i) for i in fused_order if i != protect_pos])


def rank_in_order(order: np.ndarray, tpos: int, restrict: np.ndarray | None = None) -> int:
    """Truth's rank within an explicit candidate ordering (e.g. the protected
    order), optionally restricted to a panel of positions. 0 if absent."""
    if tpos < 0 or len(order) == 0:
        return 0
    sub = np.asarray(order) if restrict is None else np.asarray(order)[np.isin(order, restrict)]
    k = np.flatnonzero(sub == tpos)
    return int(k[0]) + 1 if len(k) else 0


def paired_bootstrap(rr_a, rr_b, n_boot: int, seed: int) -> dict | None:
    """Paired-by-molecule bootstrap of mean(rr_a) - mean(rr_b) (P0.6)."""
    a, b = np.asarray(rr_a, dtype=float), np.asarray(rr_b, dtype=float)
    if len(a) == 0 or n_boot <= 0 or len(a) != len(b):
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return {"mean_delta": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4),
                     round(float(np.percentile(d, 97.5)), 4)],
            "p_delta_gt_0": round(float((d > 0).mean()), 3)}
