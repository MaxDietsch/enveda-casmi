"""Stage 2 of the class-1 experiment: a learned re-ranker over stage-1 shortlists.

For every (query molecule, candidate) pair a fixed set of ~16 numeric features
is computed by deterministic peak matching (entropy / cosine / neutral-loss
similarities, match statistics, mass error and setting flags); a small tabular
model (logistic regression and/or gradient-boosted trees from scikit-learn)
maps them to P(candidate is the truth) and the shortlist is re-ranked by that
probability. No neural network, no deep learning.

Training data is generated on the fly, in memory, from the regular train
dataset with the simulator building blocks (`make_difficulty1.qualifying` /
`choose_rows`): for each sampled molecule one setting is held out as the
pseudo-query, exactly like the eval split does. Leakage rules:

  R1  no eval molecule is ever a training query (no positive label for a
      molecule the model is evaluated on) and no split-hidden row is ever
      visible; eval molecules only appear in the roles they also play at
      evaluation time (label-0 decoys, visible library rows);
  R2  every training example holds the pseudo-query's rows out of the library,
      so the truth is reachable only through its other settings;
  R3  `change` / `difficulty` are never features (stratification only);
  R4  one shared `featurize` function serves training and evaluation;
  R5  all randomness from one seeded np.random.default_rng(--train-seed).

Usage (from the repo root):
    uv run python max_class1_exp/stage2.py \
        [--split dataset-creation/out/difficulty1.parquet] [--ppm 10] [--tol 0.02] [--noise 0.01]
        [--n-train 3000] [--train-mix instrument=0.4,adduct=0.3,ce=0.3] [--train-seed 2027]
        [--model logistic|hgb|both] [--limit N] [--workers N] [--no-cache]
        [--leak-train] [--train-on-eval] [--out-dir max_class1_exp/out]

Outputs (max_class1_exp/out/): <split>_stage2_results.parquet,
<split>_stage2_features.parquet, <split>_stage2_metrics.json,
<split>_stage2_model.joblib; a comparison table (stage 1 vs stage 2) is printed.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from library import CandidateIndex, build_library, query_mass  # noqa: E402
import common  # noqa: E402  (dataset-creation, put on sys.path by library)
import make_difficulty1  # noqa: E402
from dataloader import clean_peaks, load_split, read_rows  # noqa: E402
from spectra import (CleanParams, clean_spectrum, cosine_one_vs_many, entropy_one_vs_many,  # noqa: E402
                     entropy_weight, match_stats, merge_spectra, neutral_losses, unit_norm)
from stage1 import TOP_K, rank_order  # noqa: E402

CHANGES = ("instrument", "adduct", "ce")           # quota fill order (instrument is the rarest)
STAGE1_METHODS = ["cosine_best", "entropy_best", "entropy_composite", "entropy_combined"]
# feature columns; n_candidates is context only (constant per shortlist) and never a model feature
FEATURES = ["entropy_best", "entropy_composite", "cosine_best", "nl_entropy_best", "nl_cosine_best",
            "matched_peaks", "explained_intensity", "mass_error_ppm", "n_library_spectra",
            "has_same_adduct", "has_same_polarity", "same_instrument_family", "min_ce_gap",
            "n_source_libraries"]
FEATURE_COLUMNS = FEATURES + ["n_candidates"]
FIDX = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
RAW_FEATURES = ("has_same_adduct", "has_same_polarity", "same_instrument_family")  # no z-scoring
CE_GAP_IMPUTE = 100.0                              # min_ce_gap imputation when CE is unknown
QUERY_ROW_COLUMNS = ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct",
                    "ionization_mode", "instrument_type", "collision_energy_ev"]

# state shared with forked workers
_LIB = None
_QUERIES = None
_PARAMS = None
_COMPOSITE: dict = {}


# ---------------------------------------------------------------- featurizing

def _composite(m: int, positive: bool, lib, tol: float):
    """Merged, entropy-weighted spectrum of molecule m in one polarity.

    Cached per worker and keyed by library identity: the training and the
    evaluation libraries are different objects (see STAGE2_PLAN.md pitfalls).
    """
    k = (id(lib), m, positive)
    v = _COMPOSITE.get(k)
    if v is None:
        s, e = lib.spectra_of(m)
        spectra = [lib.get(i) for i in range(s, e) if lib.positive[i] == positive]
        mz, p = merge_spectra(spectra, tol)
        v = (mz, entropy_weight(p))
        _COMPOSITE[k] = v
    return v


def featurize(spectra, cands, abs_ppm, lib, tol: float) -> np.ndarray:
    """One feature row per candidate (R4: the one function used for training
    and evaluation alike).

    spectra: list of (mz, p, positive, precursor_mz, adduct, is_tims, ce)
    with mz/p from clean_spectrum (p sums to 1). Spectra-based features are the
    mean over the query's spectra, "best" = max over the candidate's library
    spectra, exactly like stage 1. Returns (n_cands, len(FEATURE_COLUMNS)).
    """
    n = len(cands)
    X = np.zeros((n, len(FEATURE_COLUMNS)))
    if n == 0:
        return X
    ranges = [lib.spectra_of(int(m)) for m in cands]
    n_spec = np.array([e - s for s, e in ranges])
    assert (n_spec > 0).all(), "a candidate without library spectra"
    X[:, FIDX["n_candidates"]] = n
    X[:, FIDX["mass_error_ppm"]] = abs_ppm
    X[:, FIDX["n_library_spectra"]] = n_spec
    adducts, pol_any, tim_any, ces, n_src = [], [], [], [], []
    for s, e in ranges:
        adducts.append(set(lib.table["adduct"].to_numpy()[s:e]))
        pol = lib.positive[s:e]
        pol_any.append((bool(pol.any()), bool((~pol).any())))
        tim = lib.tims[s:e]
        tim_any.append((bool(tim.any()), bool((~tim).any())))
        ces.append(lib.ce[s:e])
        n_src.append(len(set(lib.ingest[s:e])))
    X[:, FIDX["n_source_libraries"]] = n_src
    # library spectra of all candidates, contiguous per candidate (stage-1 pattern)
    spec_idx = np.concatenate([np.arange(s, e) for s, e in ranges])
    peak_idx = np.concatenate([np.arange(lib.peak_start[s], lib.peak_start[e]) for s, e in ranges])
    local_spec = np.repeat(np.arange(len(spec_idx)), np.diff(lib.peak_start)[spec_idx])
    l_mz, l_pw, l_pu, l_nl = lib.mz[peak_idx], lib.pw[peak_idx], lib.pu[peak_idx], lib.nlmz[peak_idx]
    cand_start = np.r_[0, np.cumsum(n_spec)[:-1]]
    cand_end = np.cumsum(n_spec)

    acc = {k: np.zeros(n) for k in ("entropy_best", "entropy_composite", "cosine_best",
                                    "nl_entropy_best", "nl_cosine_best", "matched_peaks",
                                    "explained_intensity", "has_same_adduct", "has_same_polarity",
                                    "same_instrument_family")}
    ce_sum, ce_cnt, used = np.zeros(n), np.zeros(n), 0
    for mz, p, positive, prec, adduct, is_tims, ce in spectra:
        if len(mz) == 0:
            continue
        used += 1
        pw, pu = entropy_weight(p), unit_norm(p)
        nl, nl_ord = neutral_losses(mz, prec)
        se = entropy_one_vs_many(mz, pw, l_mz, l_pw, local_spec, len(spec_idx), tol)
        sc = cosine_one_vs_many(mz, pu, l_mz, l_pu, local_spec, len(spec_idx), tol)
        nle = entropy_one_vs_many(nl, pw[nl_ord], l_nl, l_pw, local_spec, len(spec_idx), tol)
        nlc = cosine_one_vs_many(nl, pu[nl_ord], l_nl, l_pu, local_spec, len(spec_idx), tol)
        acc["entropy_best"] += np.maximum.reduceat(se, cand_start)
        acc["cosine_best"] += np.maximum.reduceat(sc, cand_start)
        acc["nl_entropy_best"] += np.maximum.reduceat(nle, cand_start)
        acc["nl_cosine_best"] += np.maximum.reduceat(nlc, cand_start)
        comps = [_composite(int(m), positive, lib, tol) for m in cands]
        c_lens = np.array([len(c[0]) for c in comps])
        if c_lens.sum():
            c_mz = np.concatenate([c[0] for c in comps])
            c_pw = np.concatenate([c[1] for c in comps])
            acc["entropy_composite"] += entropy_one_vs_many(
                mz, pw, c_mz, c_pw, np.repeat(np.arange(n), c_lens), n, tol)
        for j in range(n):
            s0, s1 = cand_start[j], cand_end[j]
            b = int(spec_idx[s0 + np.argmax(se[s0:s1])])     # best-entropy library spectrum
            p0, p1 = lib.peak_start[b], lib.peak_start[b + 1]
            n_matched, share = match_stats(mz, p, lib.mz[p0:p1], lib.p[p0:p1], tol)
            acc["matched_peaks"][j] += n_matched
            acc["explained_intensity"][j] += share
            if adduct in adducts[j]:
                acc["has_same_adduct"][j] += 1
            if (positive and pol_any[j][0]) or ((not positive) and pol_any[j][1]):
                acc["has_same_polarity"][j] += 1
            if (is_tims and tim_any[j][0]) or ((not is_tims) and tim_any[j][1]):
                acc["same_instrument_family"][j] += 1
            if np.isfinite(ce):
                gaps = np.abs(ces[j] - ce)
                gaps = gaps[np.isfinite(gaps)]
                if len(gaps):
                    ce_sum[j] += gaps.min()
                    ce_cnt[j] += 1
    if used:
        for k, v in acc.items():
            X[:, FIDX[k]] = v / used
        X[:, FIDX["min_ce_gap"]] = np.where(ce_cnt > 0, ce_sum / np.maximum(ce_cnt, 1), np.nan)
    else:
        X[:, FIDX["min_ce_gap"]] = np.nan
    return X


def _featurize_one(i: int) -> np.ndarray:
    q = _QUERIES[i]
    return featurize(q["spectra"], q["cands"], np.abs(q["ppm_err"]), _LIB, _PARAMS.tol)


def featurize_queries(queries, lib, params, workers, label) -> list[np.ndarray]:
    global _LIB, _QUERIES, _PARAMS
    _LIB, _QUERIES, _PARAMS = lib, queries, params
    _COMPOSITE.clear()
    with Pool(workers) as pool:
        prog = common.Progress(label, len(queries))
        out = []
        for X in pool.imap(_featurize_one, range(len(queries)), chunksize=4):
            out.append(X)
            prog.update()
        prog.close()
    return out


# ------------------------------------------------------- model matrix / fit

def zscore_within_sets(X: np.ndarray, sizes) -> np.ndarray:
    """Z-score rows within each shortlist (the contrast_within_set trick from
    CandidateScorer): the model sees how a candidate compares to its
    competitors. Constant columns within a set become all zeros."""
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
    dropped), min_ce_gap imputed, continuous features z-scored per shortlist,
    the raw flags left at 0/1 (R4: same transform for fit and predict)."""
    X = np.asarray(X, dtype=float)
    Z = X[:, [FIDX[c] for c in FEATURES]].copy()
    j = FEATURES.index("min_ce_gap")
    Z[:, j] = np.where(np.isnan(Z[:, j]), CE_GAP_IMPUTE, Z[:, j])
    zcols = [k for k, f in enumerate(FEATURES) if f not in RAW_FEATURES]
    sizes = np.asarray(sizes, dtype=np.int64)
    ends, starts = np.cumsum(sizes), np.cumsum(sizes) - sizes
    sub = np.zeros((len(Z), len(zcols)))
    for g, (s, e) in enumerate(zip(starts, ends)):
        sub[s:e] = zscore_within_sets(Z[s:e][:, zcols], [e - s])
    Z[:, zcols] = sub
    return Z


def fit_models(Z: np.ndarray, y: np.ndarray, seed: int, kinds) -> dict:
    models = {}
    if "logistic" in kinds:
        from sklearn.linear_model import LogisticRegression
        models["logistic"] = LogisticRegression(max_iter=5000, C=1.0,
                                                class_weight="balanced").fit(Z, y)
    if "hgb" in kinds:
        from sklearn.ensemble import HistGradientBoostingClassifier
        base = dict(max_depth=3, max_iter=300, early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=20, random_state=seed)
        try:
            models["hgb"] = HistGradientBoostingClassifier(class_weight="balanced", **base).fit(Z, y)
        except TypeError:                       # sklearn without class_weight in HGB
            w = np.where(y == 1, len(y) / max(1, 2 * y.sum()), len(y) / max(1, 2 * (len(y) - y.sum())))
            models["hgb"] = HistGradientBoostingClassifier(**base).fit(Z, y, sample_weight=w)
    return models


def predict(model, X: np.ndarray, sizes) -> np.ndarray:
    return model.predict_proba(model_matrix(X, sizes))[:, 1]


# ------------------------------------------------------------- ranking / mrr

def truth_rank(scores: np.ndarray, q, truth_mol: int) -> tuple[int, float, float, float]:
    """(rank, rr@25, score of the truth, best score) for one shortlist."""
    cands = q["cands"]
    pos = np.flatnonzero(cands == truth_mol)
    if len(cands) == 0 or len(pos) == 0:
        return 0, 0.0, np.nan, float(scores.max()) if len(cands) else np.nan
    order = rank_order(scores, np.abs(q["ppm_err"]), cands)
    rank = int(np.flatnonzero(order == pos[0])[0]) + 1
    return rank, (1.0 / rank if rank <= TOP_K else 0.0), float(scores[pos[0]]), float(scores.max())


def method_scores(X: np.ndarray, q) -> dict:
    """Stage-1 scores from the feature columns (they are features already)."""
    e, c = X[:, FIDX["entropy_best"]], X[:, FIDX["entropy_composite"]]
    return {"cosine_best": X[:, FIDX["cosine_best"]], "entropy_best": e,
            "entropy_composite": c, "entropy_combined": (e + c) / 2}


# ------------------------------------------------- training data on the fly

def split_hidden_by_mol(split, key_index) -> dict:
    """molecule index -> the split's hidden + eval rows (for --train-on-eval)."""
    out = {}
    for r in split.truth.itertuples():
        rows = np.concatenate([np.asarray(r.eval_row_ids, dtype=np.int64),
                               np.asarray(r.hidden_row_ids, dtype=np.int64)])
        out[int(key_index[r.key])] = rows
    return out


def sample_training_examples(n_train: int, train_mix: str, train_on_eval: bool, rng, meta, order,
                             start, eval_mols: set, hidden_by_split: dict):
    """Sampled pseudo-queries (R1, R2, R5): molecules, change types, held-out
    query rows and the rows that stay visible. Deterministic for one seed
    (fixed change order, sorted pools, rng consumed in sampling order)."""
    qual = make_difficulty1.qualifying(meta)
    meta_cols = {c: meta[c].to_numpy() for c in ("eligible", "is_tims", "adduct", "ce")}
    mix = common.parse_mix(train_mix, CHANGES)
    quotas = dict(zip(CHANGES, common.split_counts(n_train, [mix.get(c, 0.0) for c in CHANGES])))
    common.log(f"[stage2] training quotas {quotas} from mix {train_mix}")
    examples, shortfall = [], {}
    for change in CHANGES:
        pool = sorted(set(qual[change]) - (set() if train_on_eval else eval_mols))
        perm = rng.permutation(len(pool))
        picked = [pool[i] for i in perm[: quotas[change]]]
        shortfall[change] = max(0, quotas[change] - len(picked))
        for m in picked:
            rows = order[start[m]:start[m + 1]]
            q_rows, visible = make_difficulty1.choose_rows(rng, meta_cols, rows, change, qual[change][m])
            if len(q_rows) == 0:
                shortfall["empty_query"] = shortfall.get("empty_query", 0) + 1
                continue
            if train_on_eval and m in eval_mols and np.isin(q_rows, hidden_by_split.get(m, [])).any():
                shortfall["eval_row_overlap"] = shortfall.get("eval_row_overlap", 0) + 1
                continue
            examples.append({"mol": int(m), "change": change,
                             "query_rows": np.sort(q_rows),
                             "visible_rows": np.asarray(visible, dtype=np.int64),
                             "hidden_rows": np.setdiff1d(rows, visible)})
    if not train_on_eval:
        assert not (set(e["mol"] for e in examples) & eval_mols), "R1: eval molecule as training query"
    common.log(f"[stage2] sampled {len(examples)} training molecules; shortfalls {shortfall}")
    return examples, shortfall


def training_mask(split, examples) -> np.ndarray:
    """split.train_mask minus every sampled molecule's rows that are neither
    pseudo-query nor visible (R2: the truth is reachable only through its
    other settings, exactly like the eval split does)."""
    mask = split.train_mask.copy()
    for ex in examples:
        mask[ex["hidden_rows"]] = False
    return mask


def load_query_spectra(df, params: CleanParams) -> list[tuple]:
    """Cleaned query spectra (test-like: dataloader.clean_peaks, then
    clean_spectrum) with (mz, p, positive, precursor_mz, adduct, is_tims, ce)."""
    out = []
    for mz, it, prec, adduct, mode, inst, cev in zip(
            df["ms2_mzs"], df["ms2_normalized_intensities"], df["precursor_mz"], df["adduct"],
            df["ionization_mode"], df["instrument_type"], df["collision_energy_ev"]):
        m, i = clean_peaks(mz, it, float(prec))
        if m is None:
            continue
        cm, cp = clean_spectrum(m, i, float(prec), params)
        ce = np.nan if cev is None or len(cev) == 0 else float(np.mean(cev))
        out.append((cm, cp, mode == "positive", float(prec), adduct, inst == common.TIMS, ce))
    return out


def build_training_queries(examples, read_rows_fn, train_path, meta, cidx, ppm, params):
    """Candidate windows + cleaned query spectra per sampled molecule; molecules
    whose truth falls outside the mass window (or that yield no usable query
    spectrum) are dropped and counted, never padded."""
    prec, add = meta["precursor_mz"].to_numpy(), meta["adduct"].to_numpy()
    kept, dropped = [], {"truth_outside_window": 0, "no_query_mass": 0, "empty_spectra": 0}
    for ex in examples:
        mass = query_mass(prec[ex["query_rows"]], add[ex["query_rows"]])
        if not np.isfinite(mass):
            dropped["no_query_mass"] += 1
            continue
        cands, err = cidx.window(mass, ppm)
        if ex["mol"] not in set(cands.tolist()):
            dropped["truth_outside_window"] += 1
            continue
        kept.append((ex, cands.astype(np.int64), err, mass))
    queries = []
    all_rows = np.unique(np.concatenate([ex["query_rows"] for ex, _, _, _ in kept])) if kept \
        else np.empty(0, dtype=np.int64)
    if len(all_rows):
        table = read_rows_fn(all_rows, QUERY_ROW_COLUMNS, train_path)
        df = table.to_pandas() if hasattr(table, "to_pandas") else table
        df.index = all_rows
    for ex, cands, err, mass in kept:
        spectra = load_query_spectra(df.loc[ex["query_rows"]], params)
        if not spectra:
            dropped["empty_spectra"] += 1
            continue
        queries.append({"mol": ex["mol"], "change": ex["change"], "cands": cands, "ppm_err": err,
                        "spectra": spectra, "mass": mass})
    return queries, dropped


# --------------------------------------------------------------------- main

def main() -> None:
    global _LIB, _QUERIES, _PARAMS
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, default=common.OUT / "difficulty1.parquet",
                        help="split file from dataset-creation")
    parser.add_argument("--ppm", type=float, default=10.0, help="candidate mass window (± ppm)")
    parser.add_argument("--tol", type=float, default=0.02, help="m/z tolerance for peaks (Da)")
    parser.add_argument("--noise", type=float, default=0.01, help="drop peaks below this share of the base peak")
    parser.add_argument("--n-train", type=int, default=3000, help="number of training molecules")
    parser.add_argument("--train-mix", default="instrument=0.4,adduct=0.3,ce=0.3",
                        help="share of training molecules per change type")
    parser.add_argument("--train-seed", type=int, default=2027, help="seed of the training sampler")
    parser.add_argument("--model", choices=("logistic", "hgb", "both"), default="both",
                        help="which re-ranker to fit (both: the better one on internal validation is primary)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N eval molecules (quick test)")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="skip training: load a model saved by a previous run "
                             "(out/<split>_stage2_model.joblib) and only evaluate")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes (default: CPUs - 2)")
    parser.add_argument("--no-cache", action="store_true", help="rebuild the library cache")
    parser.add_argument("--leak-train", action="store_true",
                        help="self-test: train on the eval molecules with their spectra in the "
                             "library; MRR(stage2) must be ~1. Never use for real numbers")
    parser.add_argument("--train-on-eval", action="store_true",
                        help="diagnostic: also draw training queries from the eval molecules "
                             "(held-out setting disjoint from the split's hidden rows); "
                             "the metric delta vs the clean run measures the label-leakage dispute")
    parser.add_argument("--train", type=Path, default=common.TRAIN, help="train parquet")
    parser.add_argument("--cache-dir", type=Path, default=common.CACHE, help="dataset cache directory")
    parser.add_argument("--ppm-tol", type=float, default=10.0,
                        help="eligibility ppm for the training sampler (as in make_difficulty1.py)")
    parser.add_argument("--rebuild-cache", action="store_true", help="recompute the dataset key cache")
    parser.add_argument("--out-dir", type=Path, default=HERE / "out")
    args = parser.parse_args()
    if args.model_path and (args.leak_train or args.train_on_eval):
        parser.error("--model-path cannot be combined with --leak-train or --train-on-eval")
    t_start = time.time()
    params = CleanParams(tol=args.tol, noise=args.noise)
    workers = args.workers or max(1, (os.cpu_count() or 2) - 2)
    kinds = ["logistic", "hgb"] if args.model == "both" else [args.model]

    # ---- split, molecules, eval queries (stage-1 flow, with per-spectrum meta)
    split = load_split(args.split, args.train, args.cache_dir)
    info = split.info
    cache_dir = Path(info["params"]["cache_dir"])
    molecules = pd.read_parquet(cache_dir / "molecules.parquet")
    row_key = np.load(cache_dir / "row_key.npy")
    truth = split.truth.reset_index(drop=True)
    if args.limit:
        truth = truth.iloc[: args.limit].reset_index(drop=True)
    key_index = pd.Series(np.arange(len(molecules)), index=molecules["key"])
    eval_mols = {int(key_index[k]) for k in split.truth["key"]}          # the FULL split (R1)
    common.log(f"[stage2] {args.split.name}: {len(truth)} eval molecules (split has "
               f"{len(split.truth)}; all excluded from clean training)")
    ev = split.eval_spectra()
    ev = ev[ev["molecule_id"].isin(truth["molecule_id"])]
    cidx_eval = CandidateIndex(molecules, row_key, split.train_mask)
    queries_eval, need = [], set()
    groups = dict(tuple(ev.groupby("molecule_id", sort=False)))
    for r in truth.itertuples():
        g = groups.get(r.molecule_id)
        mol = int(key_index[r.key])
        if g is None:
            queries_eval.append({"mol": mol, "cands": np.empty(0, dtype=np.int64),
                                 "ppm_err": np.empty(0), "spectra": [], "mass": np.nan})
            continue
        mass = query_mass(g["precursor_mz"], g["adduct"])
        cands, err = cidx_eval.window(mass, args.ppm)
        spectra = []
        for mz, it, pr, mode, add, inst, cev in zip(
                g["ms2_mzs"], g["ms2_normalized_intensities"], g["precursor_mz"],
                g["ionization_mode"], g["adduct"], g["instrument_type"], g["collision_energy_ev"]):
            cm, cp = clean_spectrum(mz, it, pr, params)
            ce = np.nan if cev is None or len(cev) == 0 else float(np.mean(cev))
            spectra.append((cm, cp, mode == "positive", float(pr), add, inst == common.TIMS, ce))
        queries_eval.append({"mol": mol, "cands": cands.astype(np.int64), "ppm_err": err,
                             "spectra": spectra, "mass": mass})
        need.update(cands.tolist())
    need = np.array(sorted(need), dtype=np.int64)
    lib_rows_eval = np.flatnonzero(split.train_mask & np.isin(row_key, need)) if len(need) \
        else np.empty(0, dtype=np.int64)
    if args.leak_train:
        lib_rows_eval = np.union1d(lib_rows_eval, split.eval_row_ids)
    else:
        # leakage guards (stage-1 pattern): library = visible rows only, never an eval row
        assert split.train_mask[lib_rows_eval].all()
        assert not np.isin(split.eval_row_ids, lib_rows_eval).any()
    lib_eval = build_library(lib_rows_eval, row_key, split.train_path, params,
                             {**info, "leak": args.leak_train}, workers, use_cache=not args.no_cache)
    t_eval_lib = time.time()
    X_eval = featurize_queries(queries_eval, lib_eval, params, workers,
                               "[stage2] eval features")
    t_eval_feat = time.time() - t_eval_lib
    n_lib_eval = len(lib_eval)
    del lib_eval                    # eval features are computed; the library is dead weight below
    _LIB = None                     # featurize_queries stashed it globally; drop the last reference

    # ---- phase 1-3: training data on the fly, then fit (or load a saved model)
    if args.model_path:
        training = {"mode": "eval_only"}
        train_queries, X_train = [], []
    elif args.leak_train:
        training = {"mode": "leak_train"}
        train_queries = queries_eval
        X_train = X_eval
    else:
        training = {"mode": "train_on_eval" if args.train_on_eval else "clean"}
        prep = SimpleNamespace(train=split.train_path, cache_dir=cache_dir,
                               rebuild_cache=args.rebuild_cache, ppm_tol=args.ppm_tol)
        meta, mol2, rk2 = common.prepare(prep)
        assert len(mol2) == len(molecules) and (rk2 == row_key).all(), "cache changed under us"
        order, start = common.mol_rows(meta)
        rng = np.random.default_rng(args.train_seed)
        examples, shortfall = sample_training_examples(
            args.n_train, args.train_mix, args.train_on_eval, rng, meta, order, start,
            eval_mols, split_hidden_by_mol(split, key_index) if args.train_on_eval else {})
        mask = training_mask(split, examples)
        cidx_train = CandidateIndex(molecules, row_key, mask)
        train_queries, dropped = build_training_queries(
            examples, read_rows, split.train_path, meta, cidx_train, args.ppm, params)
        training.update(shortfalls=shortfall, dropped=dropped)
        del meta, order, start, mol2, rk2   # the 2.5M-row metadata frame is not needed past
                                           # sampling; free it before the memory-hungry library build
        # R1 / R2 asserts: no split-hidden row and no pseudo-query row in the training library
        all_query_rows = np.unique(np.concatenate([ex["query_rows"] for ex in examples])) if examples \
            else np.empty(0, dtype=np.int64)
        need_train = np.array(sorted({int(m) for q in train_queries for m in q["cands"]}), dtype=np.int64)
        lib_rows_train = np.flatnonzero(mask & np.isin(row_key, need_train)) if len(need_train) \
            else np.empty(0, dtype=np.int64)
        assert split.train_mask[lib_rows_train].all(), "R1: split-hidden row in the training library"
        assert not np.isin(all_query_rows, lib_rows_train).any(), "R2: pseudo-query row in the library"
        common.log(f"[stage2] training library: {len(lib_rows_train):,} spectra rows")
        lib_train = build_library(lib_rows_train, row_key, split.train_path, params,
                                  {**info, "tag_extra": {"stage2": "train", "seed": args.train_seed,
                                                         "n_train": args.n_train,
                                                         "mix": args.train_mix,
                                                         "train_on_eval": bool(args.train_on_eval)}},
                                  workers, use_cache=not args.no_cache)
        t0 = time.time()
        X_train = featurize_queries(train_queries, lib_train, params, workers,
                                    "[stage2] training features")
        training["featurize_seconds"] = round(time.time() - t0, 1)
        del lib_train
        _LIB = None
    # shortlists without candidates cannot be trained on (truth outside the window)
    pairs = [(q, X) for q, X in zip(train_queries, X_train) if len(q["cands"]) > 0]
    train_queries, X_train = [p[0] for p in pairs], [p[1] for p in pairs]
    y_per = [q["cands"] == q["mol"] for q in train_queries]
    for y in y_per:
        assert y.sum() == 1, "exactly one positive per shortlist"
    sizes = [len(q["cands"]) for q in train_queries]
    X_all = np.concatenate(X_train) if len(X_train) else np.empty((0, len(FEATURE_COLUMNS)))
    y_all = np.concatenate(y_per) if y_per else np.empty(0)
    training.update(n_molecules=len(train_queries), n_rows=len(y_all),
                    class_balance=round(float(y_all.mean()), 4) if len(y_all) else None,
                    median_candidates=float(np.median(sizes)) if sizes else None)
    if not args.model_path:
        common.log(f"[stage2] training set: {len(train_queries)} shortlists, {len(y_all):,} rows, "
                   f"{100 * y_all.mean():.1f}% positive" if len(y_all) else "[stage2] empty training set")

    # internal validation: 10% of the training molecules (rng, R5)
    n = len(train_queries)
    is_val = np.zeros(n, dtype=bool)
    if n > 1 and not args.leak_train:
        is_val = rng.random(n) < 0.1
        if not is_val.any():
            is_val = np.zeros(n, dtype=bool)
            is_val[rng.choice(n, size=max(1, n // 10), replace=False)] = True
        if is_val.all():
            is_val = np.zeros(n, dtype=bool)
            is_val[rng.choice(n, size=max(1, n // 10), replace=False)] = True
    if args.leak_train:
        common.log("[stage2] LEAK SELF-TEST: training = the eval shortlists themselves")
    Z_all = model_matrix(X_all, sizes)
    tr = np.flatnonzero(~is_val)                      # shortlist indices
    ends, starts = np.cumsum(sizes), np.cumsum(sizes) - np.asarray(sizes)

    def _rows(sel) -> np.ndarray:                    # row indices of the given shortlists
        return np.concatenate([np.arange(starts[i], ends[i]) for i in sel]) if len(sel) \
            else np.empty(0, dtype=np.int64)

    models = fit_models(Z_all[_rows(tr)], y_all[_rows(tr)], args.train_seed, kinds) if len(tr) else {}

    def val_mrr(model=None) -> float:
        rr = []
        i = 0
        for q, X, val in zip(train_queries, X_train, is_val):
            if not val or len(q["cands"]) == 0:
                continue
            s = predict(model, X, [len(q["cands"])]) if model is not None \
                else method_scores(X, q)["entropy_best"]
            rr.append(truth_rank(s, q, q["mol"])[1])
        return round(float(np.mean(rr)), 4) if rr else None

    internal = {"entropy_best_baseline": val_mrr()}
    primary = None
    for kind in kinds:
        if kind in models:
            internal[f"stage2_{kind}"] = val_mrr(models[kind])
    fitted = [k for k in kinds if k in models]
    if fitted:
        scores = {k: internal[f"stage2_{k}"] for k in fitted}
        primary = max(fitted, key=lambda k: (scores[k] is not None, scores[k] or 0.0))
    training.update(internal_validation=internal, primary_model=primary)

    # feature importances (logistic coefficients; permutation importance for the trees)
    importances = {}
    if "logistic" in models:
        importances["logistic"] = dict(zip(FEATURES, models["logistic"].coef_[0].round(6).tolist()))
    if "hgb" in models and is_val.any():
        from sklearn.inspection import permutation_importance
        va = np.flatnonzero(is_val)
        Xv = np.concatenate([X_train[i] for i in va])
        yv = np.concatenate([y_per[i] for i in va])
        sv = [len(train_queries[i]["cands"]) for i in va]
        r = permutation_importance(models["hgb"], model_matrix(Xv, sv), yv, n_repeats=5,
                                   random_state=args.train_seed)
        importances["hgb"] = dict(zip(FEATURES, r.importances_mean.round(6).tolist()))
    training["feature_importances"] = importances

    if args.model_path:
        import joblib
        blob = joblib.load(args.model_path)
        assert blob.get("features") == FEATURES, \
            f"{args.model_path.name} was trained with a different feature set; retrain first"
        models = {blob["kind"]: blob["model"]}
        fitted, primary = [blob["kind"]], blob["kind"]
        training = {"mode": "eval_only", "model_path": str(args.model_path),
                   "train_seed_of_model": blob.get("train_seed"), "primary_model": primary,
                   "internal_validation": None, "feature_importances": None}
        common.log(f"[stage2] loaded the {blob['kind']} model from {args.model_path.name}; "
                   f"training skipped (features match)")
    elif models:
        import joblib
        args.out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": models[primary], "kind": primary, "features": FEATURES,
                     "raw_features": list(RAW_FEATURES), "impute": {"min_ce_gap": CE_GAP_IMPUTE},
                     "train_seed": args.train_seed},
                    args.out_dir / f"{args.split.stem}_stage2_model.joblib")
    training["seconds_total_train"] = round(time.time() - t_eval_lib - t_eval_feat, 1)

    # ---- phase 4: evaluate on the untouched split
    def stage2_scores(name, X, q):
        return predict(models[name], X, [len(q["cands"])]) if len(q["cands"]) else np.empty(0)

    rows, feat_rows = [], []
    for r, q, X in zip(truth.itertuples(), queries_eval, X_eval):
        t = int(key_index[r.key])
        rec = {"molecule_id": r.molecule_id, "key": r.key, "change": r.change,
               "difficulty": int(r.difficulty), "n_eval_spectra": len(q["spectra"]),
               "n_candidates": len(q["cands"]), "truth_in_candidates": bool((q["cands"] == t).any()),
               "query_mass": q.get("mass", np.nan)}
        if len(q["cands"]):
            fr = pd.DataFrame(X, columns=FEATURE_COLUMNS)
            fr.insert(0, "molecule_id", r.molecule_id)
            fr.insert(1, "change", r.change)
            fr.insert(2, "cand_key", molecules["key"].to_numpy()[q["cands"]])
            fr.insert(3, "label", (q["cands"] == t).astype(np.int8))
            feat_rows.append(fr)
        for meth, s in method_scores(X, q).items():
            rank, rr, ts, bs = truth_rank(s, q, t) if len(q["cands"]) else (0, 0.0, np.nan, np.nan)
            rec[f"rank_{meth}"], rec[f"rr_{meth}"] = rank, rr
            rec[f"truth_score_{meth}"], rec[f"top_score_{meth}"] = ts, bs
            if meth == "entropy_combined" and len(q["cands"]):
                top = rank_order(s, np.abs(q["ppm_err"]), q["cands"])[:TOP_K]
                rec["top25_keys"] = molecules["key"].to_numpy()[q["cands"][top]].tolist()
                rec["top25_smiles"] = molecules["smiles"].to_numpy()[q["cands"][top]].tolist()
                rec["top25_scores"] = s[top].round(4).tolist()
        for name in fitted:
            s = stage2_scores(name, X, q)
            rank, rr, ts, bs = truth_rank(s, q, t) if len(q["cands"]) else (0, 0.0, np.nan, np.nan)
            rec[f"rank_stage2_{name}"], rec[f"rr_stage2_{name}"] = rank, rr
            rec[f"truth_score_stage2_{name}"], rec[f"top_score_stage2_{name}"] = ts, bs
            if len(q["cands"]):
                rec[f"p_truth_stage2_{name}"] = ts
            if name == primary:
                rec["rank_stage2"], rec["rr_stage2"] = rank, rr
                rec["p_truth_stage2"], rec["truth_score_stage2"] = ts, ts
                rec["top_score_stage2"] = bs
        rows.append(rec)
    res = pd.DataFrame(rows)

    def summary(df: pd.DataFrame) -> dict:
        d = {"n": len(df), "candidate_recall": round(float(df["truth_in_candidates"].mean()), 4),
             "median_candidates": float(df["n_candidates"].median())}
        methods = list(STAGE1_METHODS) + [f"stage2_{k}" for k in fitted]
        if primary:
            methods.append("stage2")
        for meth in methods:
            rk = df[f"rank_{meth}"]
            d[meth] = {"mrr@25": round(float(df[f"rr_{meth}"].mean()), 4),
                       "top1": round(float((rk == 1).mean()), 4),
                       "top10": round(float(((rk >= 1) & (rk <= 10)).mean()), 4),
                       "top25": round(float(((rk >= 1) & (rk <= 25)).mean()), 4)}
        return d

    metrics = {"split": str(args.split), "params": {k: str(v) for k, v in vars(args).items()},
               "training": training,
               "overall": summary(res),
               "by_change": {c: summary(g) for c, g in res.groupby(res["change"].fillna("none"))},
               "library_spectra": n_lib_eval,
               "seconds": {"eval_features": round(t_eval_feat, 1), "total": round(time.time() - t_start, 1)}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.split.stem + ("_leak" if args.leak_train else "") \
        + ("_on_eval" if args.train_on_eval else "") + (f"_limit{args.limit}" if args.limit else "")
    res.to_parquet(args.out_dir / f"{stem}_stage2_results.parquet")
    if feat_rows:
        pd.concat(feat_rows, ignore_index=True).to_parquet(
            args.out_dir / f"{stem}_stage2_features.parquet")
    (args.out_dir / f"{stem}_stage2_metrics.json").write_text(json.dumps(metrics, indent=1))

    # ---- printed table (stage 1 vs stage 2), stage-1 layout
    table_methods = list(STAGE1_METHODS) + [f"stage2_{k}" for k in fitted]
    groups = [("overall", metrics["overall"])] + list(metrics["by_change"].items())
    common.log("")
    common.log(f"MRR@25 (top-1 / top-10)   {args.split.name}"
               + ("   [LEAK SELF-TEST]" if args.leak_train else "")
               + ("   [TRAIN-ON-EVAL, diagnostic]" if args.train_on_eval else ""))
    common.log(f"{'':20s}" + "".join(f"{g:>22s}" for g, _ in groups))
    common.log(f"{'n molecules':20s}" + "".join(f"{m['n']:>22d}" for _, m in groups))
    common.log(f"{'candidate recall':20s}" + "".join(f"{m['candidate_recall']:>22.3f}" for _, m in groups))
    for meth in table_methods:
        label = meth + (" *" if meth == f"stage2_{primary}" else "")
        common.log(f"{label:20s}" + "".join(
            f"{m[meth]['mrr@25']:>10.3f} ({m[meth]['top1']:.2f}/{m[meth]['top10']:.2f})"
            for _, m in groups))
    common.log(f"\n[stage2] model: {primary} (internal validation "
               f"{json.dumps(internal)}); results in {args.out_dir}/{stem}_stage2_*; "
               f"total {time.time() - t_start:.0f}s")

    # --train-on-eval: report the metric delta against the clean run's metrics
    # file (same split, same limit, same seed) — the memorization effect (R1 label half)
    if args.train_on_eval:
        clean_stem = args.split.stem + (f"_limit{args.limit}" if args.limit else "")
        p = args.out_dir / f"{clean_stem}_stage2_metrics.json"
        if p.exists():
            clean = json.loads(p.read_text())
            common.log("")
            common.log(f"[stage2] leakage delta (train-on-eval - clean, MRR@25), clean run: {p.name}")
            for name in [f"stage2_{k}" for k in fitted]:
                parts = []
                for c, g in metrics["by_change"].items():
                    base = clean["by_change"].get(c, {}).get(name, {}).get("mrr@25")
                    if base is None:
                        continue
                    parts.append(f"{c} {g[name]['mrr@25'] - base:+.3f}")
                if parts:
                    common.log(f"  {name:18s} " + "   ".join(parts))
        else:
            common.log(f"[stage2] no clean-run metrics found at {p.name}; run "
                       f"`stage2.py` with the same --train-seed and --limit for the delta")


if __name__ == "__main__":
    main()
