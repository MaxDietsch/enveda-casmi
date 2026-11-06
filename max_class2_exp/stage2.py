"""Stage 2 of the class-2 experiment: rank *structures* for never-measured molecules.

Class 2 = "the structure is known (in PubChem / COCONUT), but no public
spectrum exists". The eval molecules of the split (default:
dataset-creation/out/difficulty2.parquet) have every train row hidden, so
spectral library search cannot see them at all; their structures enter the
candidate pool through split.library_additions() (simulator §7.3).

For every (query molecule, candidate structure) pair a fixed set of numeric
features is computed (features.py):

  analogue     OPTIONAL (--enable-analogues, off by default - forum P1.7):
               the candidate's top-K nearest visible molecules stand in for
               its missing spectra. A visible candidate's own spectra are its
               best "neighbour" - evidence that it *was* measured, which the
               truth never has;
  in-silico    the repo Fragmentor enumerates the candidate's theoretical
               fragments for the query's adduct; matched peaks, explained
               intensity and ppm closeness (MetFrag / FingerID-style);
  metfrag-lite the forum's simple block (P1.1): 1-bond / 2-bond (x0.6) cuts,
               H shifts -2..+3, linear intensity, rarity weighting, gap to
               the best candidate;
  fingerprint  channel P (fpchannel.py, ON by default, no flag): predict the
               query's 2048-bit Morgan fingerprint from its spectra and score
               each candidate's actual bits (prior-normalised log-likelihood
               + Tanimoto). v1 = linear model trained inside the run on
               split-visible rows only (leak-free by construction); v2 =
               frozen de-novo encoder + head when the artifacts exist, and
               only if their checkpoint metadata PROVES no eval-split
               molecule was trained on (else the run refuses them);
  neural       channel F2 (f2_bridge.py -> ICEBERG in the ms-pred venv, ON
               when the env + weights exist, no flag): the candidate's
               predicted spectrum (adduct/CE conditioned, zero-shot public
               weights) vs the observed one, matched cosine; predictions
               are disk-cached across runs. Zero-shot weights never saw any
               Enveda spectrum, so no eval or pseudo-truth can be memorised;
  context      mass error (median/worst spectrum), visibility and source
               flags, same-formula competition.

A small tabular model (logistic regression and/or gradient-boosted trees,
seed-averaged) maps the features to P(candidate is the truth) and re-ranks
the mass-window shortlists. The re-ranker itself stays tabular; the neural
models live in the channels (fingerprint prediction, ICEBERG), never in
the fusion.

Training data is generated on the fly, in memory, from the regular train
dataset: sampled molecules have ALL their rows hidden (exactly what the eval
split does to its molecules), serve as the pseudo-query, and are added to
the training pool as library additions. Leakage rules:

  R1  no eval molecule is ever a training query and no split-hidden row is
      ever visible; training queries are drawn only from molecules that are
      visible under the split;
  R2  every training example hides all of the pseudo-truth's rows, so the
      truth is reachable only as a structure;
  R3  `difficulty` / `neighbours_removed` / `max_visible_tc` are simulator
      metadata, never features (stratification only);
  R4  one shared `featurize` function serves training and evaluation;
  R5  all randomness from one seeded np.random.default_rng(--train-seed).

Evaluation protocol additions from the forum findings
(FORUM_FINDINGS.md §8; "the simulator is right; the metrics must target
same-formula ranking"):

  P0.2 same-formula isomer panel (truth + same-formula candidates; and
       truth + top-10 same-formula decoys) - the sensitive class-2 metric;
  P0.3 per-molecule failure ledger (outside_window / >25 / 2-25 / rank 1)
       and MRR conditional on the truth being in the window;
  P0.4 stratification by has_public_spectra (memorisation-risk slice for
       any pretrained channel);
  P0.5 --blind-control: swap query spectra across mass-adjacent molecules
       and re-score; channels must collapse to ~0 (mass_error stays);
  P0.6 --n-boot paired-by-molecule bootstrap CIs; --model-seeds seed-averaged
       HGB (forum: pin seeds and average);
  P1.2 defaults follow the forum: --ppm 8.5, --tol 0.01;
  P1.5 --protect-top1: keep a chosen channel's top-1 at rank 1 and insert
       the fused top-1 at rank 2 (cannot lose that channel's rank-1 hits).

Usage (from the repo root):
    uv run python max_class2_exp/stage2.py \
        [--split dataset-creation/out/difficulty2.parquet] [--ppm 8.5]
        [--tol 0.01] [--noise 0.01] [--frag-ppm 10] [--mtf-tol 0.01]
        [--enable-analogues] [--k-neighbours 8] [--min-neighbour-tc 0.30]
        [--n-train 2000] [--train-seed 2027] [--model logistic|hgb|both]
        [--model-seeds 1] [--n-boot 200] [--protect-top1 CHANNEL] [--blind-control]
        [--drop-features f1,f2] [--limit N] [--workers N] [--no-cache] [--leak-train]
        [--external-pool data/coconut_pool.parquet]
        [--out-dir max_class2_exp/out]

Outputs (max_class2_exp/out/): <split>_stage2_results.parquet,
<split>_stage2_features.parquet, <split>_stage2_metrics.json,
<split>_stage2_model.joblib; a comparison table (channels vs re-ranker,
panel metrics, blind control) is printed.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "dataset-creation"))

from annotations import (annotate_pool_formulas, annotate_pool_sources, formulas_for,  # noqa: E402
                         load_row_sources, truth_public_flags, visible_source_stats)
import f2_bridge  # noqa: E402
from features import (BUCKETS, FEATURES, FEATURE_COLUMNS, FIDX, TOP_K,  # noqa: E402
                      append_channel_blocks, dropped_features, failure_bucket,
                      featurize_queries, fit_models, model_matrix, paired_bootstrap,
                      predict, protect_top1_order, rank_in_order, rank_order,
                      rank_within, set_dropped_features, truth_rank)
import fpchannel  # noqa: E402
from neighbours import topk_neighbours  # noqa: E402
from pool import build_pool  # noqa: E402
from spectra_utils import CleanParams, clean_spectrum  # noqa: E402
from speclib import build_library, query_mass  # noqa: E402
import common  # noqa: E402
from dataloader import clean_peaks, load_split, read_rows  # noqa: E402
from enveda_casmi26 import tracking  # noqa: E402

# channel baselines printed/evaluated straight from the feature columns
BASELINES = ["mass_error", "entropy_best", "entropy_composite", "frag_explained", "mtf_explained"]
PROTECT_CHOICES = ["mass_error", "entropy_best", "entropy_composite", "frag_explained",
                   "mtf_explained"]
QUERY_ROW_COLUMNS = ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz",
                     "adduct", "ionization_mode", "collision_energy_ev"]
TC_BANDS = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]
# P0.1: the class shares are NOT known; report gains under all three estimates
CLASS_SHARE_SCENARIOS = {
    "16/45/39 (741597 LB probe)": 0.45,
    "28/20/52 (748524 LB fit)": 0.20,
    "20/60/20 (simulator default)": 0.60,
}


# ------------------------------------------------------------------ queries

def load_query_spectra(df, params: CleanParams) -> list[tuple]:
    """Cleaned query spectra (test-like: dataloader.clean_peaks, then
    clean_spectrum) as (mz, p, positive, precursor_mz, adduct, ce_ev)."""
    out = []
    for mz, it, prec, adduct, mode, cev in zip(df["ms2_mzs"], df["ms2_normalized_intensities"],
                                               df["precursor_mz"], df["adduct"],
                                               df["ionization_mode"], df["collision_energy_ev"]):
        m, i = clean_peaks(mz, it, float(prec))
        if m is None:
            continue
        cm, cp = clean_spectrum(m, i, float(prec), params)
        out.append((cm, cp, mode == "positive", float(prec), adduct, fpchannel.ce_of(cev)))
    return out


def _query_context(g) -> dict:
    """Query-level context (never model features, R3-adjacent): adducts,
    polarities, precursor presence, and the spread of the per-spectrum
    implied neutral masses (forum P1.3: multi-adduct agreement ~1.3 ppm)."""
    prec = np.asarray(g["precursor_mz"], dtype=float)
    raw = g["ms2_mzs"].tolist()
    has_prec = any(any(abs(float(m) - p) < 0.1 for m in (peaks or []))
                   for peaks, p in zip(raw, prec))
    masses = [m for m in (query_mass([p], [a]) for p, a in zip(prec, g["adduct"]))
              if np.isfinite(m)]
    agreement = float("nan")
    if len(masses) >= 2:
        med = np.median(masses)
        agreement = float(np.median(np.abs(np.asarray(masses) - med)) / med * 1e6)
    return {"n_query_adducts": int(g["adduct"].nunique()),
            "n_query_polarities": int(g["ionization_mode"].nunique()),
            "has_precursor_peak": bool(has_prec),
            "mass_agreement_ppm": agreement}


def build_eval_queries(split, truth, pool, params: CleanParams, ppm: float):
    """Queries for the eval molecules: candidate window, cleaned spectra,
    truth member index (-1 if the truth is not in the pool at all)."""
    ev = split.eval_spectra()
    ev = ev[ev["molecule_id"].isin(truth["molecule_id"])]
    groups = dict(tuple(ev.groupby("molecule_id", sort=False)))
    ctx0 = {"n_query_adducts": 0, "n_query_polarities": 0,
            "has_precursor_peak": False, "mass_agreement_ppm": np.nan}
    queries, cands_union = [], set()
    for r in truth.itertuples():
        member = pool.member_of_key(r.key)
        g = groups.get(r.molecule_id)
        if g is None:
            queries.append({"truth": member, "cands": np.empty(0, dtype=np.int64),
                            "ppm_err": np.empty(0), "spectra": [], "mass": np.nan,
                            "key": r.key, **ctx0})
            continue
        mass = query_mass(g["precursor_mz"], g["adduct"])
        cands, err = pool.window(mass, ppm)
        spectra = []
        for mz, it, pr, mode, add, cev in zip(g["ms2_mzs"], g["ms2_normalized_intensities"],
                                             g["precursor_mz"], g["ionization_mode"],
                                             g["adduct"], g["collision_energy_ev"]):
            cm, cp = clean_spectrum(mz, it, float(pr), params)
            spectra.append((cm, cp, mode == "positive", float(pr), add, fpchannel.ce_of(cev)))
        queries.append({"truth": member, "cands": cands.astype(np.int64), "ppm_err": err,
                        "spectra": spectra, "mass": mass, "key": r.key, **_query_context(g)})
        cands_union.update(cands.tolist())
    return queries, cands_union


# ------------------------------------------------------- neighbours / library

def visible_allowed(n_struct: int, visible_mols: np.ndarray) -> np.ndarray:
    """Bool mask over the fingerprint universe: which structures may serve as
    analogue neighbours (molecules with >= 1 visible row under the mask in force)."""
    allowed = np.zeros(n_struct, dtype=bool)
    allowed[np.asarray(visible_mols, dtype=np.int64)] = True
    return allowed


def neighbour_lists(pool, members, allowed, fps, k, min_tc, workers, label):
    """{pool member index: (visible neighbour molecule indices, Tanimotos)}."""
    if k <= 0:
        return {}
    struct_to_member = {int(pool.struct[m]): int(m) for m in set(int(x) for x in members)}
    nb = topk_neighbours(fps, sorted(struct_to_member), allowed, k, min_tc, workers, label)
    return {mm: nb.get(st, (np.empty(0, dtype=np.int64), np.empty(0)))
            for st, mm in struct_to_member.items()}


def analogue_library(nbrs, mask, row_key, train_path, params, info, tag_extra, workers,
                     use_cache: bool, extra_rows=None):
    """The visible train spectra of every neighbour of every candidate (the
    analogue channel's library; empty when the channel is off).
    extra_rows (--leak-train): eval rows added to the library."""
    mols = np.unique(np.concatenate([idx for idx, _ in nbrs.values()])) if nbrs \
        else np.empty(0, dtype=np.int64)
    rows = np.flatnonzero(np.asarray(mask) & np.isin(row_key, mols)) if len(mols) \
        else np.empty(0, dtype=np.int64)
    if extra_rows is not None and len(extra_rows):
        rows = np.union1d(rows, extra_rows)
    else:
        assert np.asarray(mask)[rows].all(), "a hidden row entered the analogue library"
    return build_library(rows, row_key, train_path, params, {**info, "tag_extra": tag_extra},
                         workers, use_cache)


# ------------------------------------------------- training data on the fly

def sample_training_examples(n_train: int, rng, molecules, meta, visible_mols, eval_mols: set):
    """Pseudo-class-2 molecules (R1, R5): drawn only from molecules that are
    *visible* under the split (so no split-hidden row is ever touched) and
    are not eval molecules. Their rows will all be hidden (R2)."""
    cand = np.array(sorted(set(visible_mols.tolist())
                           & set(common.eval_candidates(molecules).tolist()) - eval_mols), dtype=np.int64)
    perm = rng.permutation(len(cand))
    picked = cand[perm[: n_train]]
    shortfalls = {"molecules": max(0, n_train - len(picked))}
    order, start = common.mol_rows(meta)
    eligible = meta["eligible"].to_numpy()
    examples = []
    for m in picked:
        rows = order[start[m]: start[m + 1]]
        elig = np.sort(rows[eligible[rows]])
        if len(elig) == 0:
            shortfalls["no_eligible_rows"] = shortfalls.get("no_eligible_rows", 0) + 1
            continue
        k = common.draw_n_spectra(rng, len(elig))
        q_rows = np.sort(rng.choice(elig, size=k, replace=False))
        examples.append({"mol": int(m), "key": str(molecules["key"].iat[m]),
                         "query_rows": q_rows, "hidden_rows": np.sort(rows)})
    common.log(f"[stage2] sampled {len(examples)} training molecules; shortfalls {shortfalls}")
    return examples, shortfalls


def build_training_queries(examples, molecules, pool_train, meta, ppm: float, params: CleanParams,
                           train_path: Path):
    """Candidate windows + cleaned query spectra per sampled molecule. The
    pseudo-truth is a pool member through the additions; examples whose truth
    falls outside the mass window (or with no usable spectrum) are dropped
    and counted, never padded."""
    prec, add = meta["precursor_mz"].to_numpy(), meta["adduct"].to_numpy()
    kept, dropped = [], {"truth_outside_window": 0, "no_query_mass": 0, "empty_spectra": 0}
    for ex in examples:
        mass = query_mass(prec[ex["query_rows"]], add[ex["query_rows"]])
        if not np.isfinite(mass):
            dropped["no_query_mass"] += 1
            continue
        cands, err = pool_train.window(mass, ppm)
        member = pool_train.member_of_key(ex["key"])
        if member < 0 or member not in set(cands.tolist()):
            dropped["truth_outside_window"] += 1
            continue
        kept.append((ex, member, cands.astype(np.int64), err, mass))
    queries = []
    if kept:
        all_rows = np.unique(np.concatenate([ex["query_rows"] for ex, _, _, _, _ in kept]))
        df = read_rows(all_rows, QUERY_ROW_COLUMNS, train_path).to_pandas()
        df.index = all_rows
    for ex, member, cands, err, mass in kept:
        spectra = load_query_spectra(df.loc[ex["query_rows"]], params)
        if not spectra:
            dropped["empty_spectra"] += 1
            continue
        queries.append({"mol": ex["mol"], "truth": member, "cands": cands, "ppm_err": err,
                        "spectra": spectra, "mass": mass, "key": ex["key"]})
    return queries, dropped


# ------------------------------------------------------------------- ranking

def method_scores(X: np.ndarray) -> dict:
    """Channel baselines straight from the feature columns."""
    return {"mass_error": -X[:, FIDX["mass_error_ppm"]],
            "entropy_best": X[:, FIDX["entropy_best"]],
            "entropy_composite": X[:, FIDX["entropy_composite"]],
            "frag_explained": X[:, FIDX["frag_explained_intensity"]],
            "mtf_explained": X[:, FIDX["mtf_explained_linear"]]}


def blind_queries(queries, offset: int):
    """Spectrum-blind control (P0.5): each query receives the spectra of the
    query `offset` places away in mass order (mass-adjacent molecules, as
    the forum suggests). Candidate windows, truth membership and mass
    features are untouched - only the spectral evidence is falsified."""
    usable = [i for i, q in enumerate(queries) if len(q["spectra"])]
    if len(usable) < 2:
        return []
    order = sorted(usable, key=lambda i: queries[i]["mass"])
    out = [None] * len(queries)
    for pos, i in enumerate(order):
        src = queries[order[(pos + offset) % len(order)]]
        q = dict(queries[i])
        q["spectra"] = src["spectra"]
        out[i] = q
    return [out[i] for i in usable]


# ---------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", type=Path, default=common.OUT / "difficulty2.parquet",
                        help="split file from dataset-creation")
    parser.add_argument("--ppm", type=float, default=8.5,
                        help="candidate mass window (+/- ppm; forum P1.2: 8.5 is the "
                             "community sweet spot, sweep 5/8.5/10)")
    parser.add_argument("--tol", type=float, default=0.01,
                        help="m/z tolerance for peaks (Da; forum P1.2: 0.01, sweep "
                             "0.005/0.01/0.02)")
    parser.add_argument("--noise", type=float, default=0.01, help="drop peaks below this share of the base peak")
    parser.add_argument("--frag-ppm", type=float, default=10.0,
                        help="m/z tolerance (ppm) for matching in-silico fragments")
    parser.add_argument("--mtf-tol", type=float, default=0.01,
                        help="m/z tolerance (Da) for the metfrag-lite channel (NeckBeard: "
                             "0.005 > 0.01, small effect)")
    parser.add_argument("--enable-analogues", action="store_true",
                        help="turn the analogue channel on (forum P1.7: off by default; "
                             "measure its marginal contribution, don't lean on it)")
    parser.add_argument("--k-neighbours", type=int, default=0,
                        help="analogue channel: neighbours per candidate (0 = channel off; "
                             "--enable-analogues sets 8)")
    parser.add_argument("--min-neighbour-tc", type=float, default=0.30,
                        help="analogue channel: minimum Tanimoto of a neighbour")
    parser.add_argument("--n-train", type=int, default=2000, help="number of training molecules")
    parser.add_argument("--train-seed", type=int, default=2027, help="seed of the training sampler")
    parser.add_argument("--model", choices=("logistic", "hgb", "both"), default="both",
                        help="which re-ranker to fit (both: the better one on internal validation is primary)")
    parser.add_argument("--model-seeds", type=int, default=1,
                        help="HGB seeds to fit and average (forum 742088: pin seeds and average)")
    parser.add_argument("--n-boot", type=int, default=200,
                        help="paired-by-molecule bootstrap resamples for stage2-vs-baseline "
                             "deltas (0 = off)")
    parser.add_argument("--protect-top1", choices=PROTECT_CHOICES, default=None,
                        help="keep this channel's top-1 at rank 1, insert the fused top-1 "
                             "at rank 2 (P1.5, forum 749412)")
    parser.add_argument("--blind-control", action="store_true",
                        help="also featurize and score a spectrum-blind control (P0.5): "
                             "query spectra swapped across mass-adjacent molecules")
    parser.add_argument("--drop-features", default="",
                        help="comma-separated FEATURES to ablate from the model (they stay "
                             "in the saved feature frames); e.g. cand_is_addition,cand_public_rows")
    parser.add_argument("--keep-source-features", action="store_true",
                        help="let the model use the visibility/source block (cand_is_addition, "
                             "cand_visible_rows, cand_tims_rows, cand_public_rows). It is dropped "
                             "by default: within the simulator these flags mark the truth (every "
                             "eval truth is a spectrum-less addition, MRR ~1.0 from the flag "
                             "alone), so they measure the simulator, not the chemistry "
                             "(FORUM_FINDINGS.md P1.3/P2.2)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N eval molecules (quick test)")
    parser.add_argument("--model-path", type=Path, default=None,
                        help="skip training: load a model saved by a previous run and only evaluate")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes (default: CPUs - 2)")
    parser.add_argument("--no-cache", action="store_true", help="rebuild the library cache")
    parser.add_argument("--leak-train", action="store_true",
                        help="self-test: train on the eval shortlists with the eval spectra leaked "
                             "into the truths' analogue spectra; MRR(stage2) must be ~1. "
                             "Never use for real numbers")
    parser.add_argument("--external-pool", type=Path, default=None,
                        help="optional extra structures (key/inchikey14, smiles[, mass]) to add to "
                             "the candidate pool, e.g. a COCONUT extract")
    parser.add_argument("--train", type=Path, default=None,
                        help="train parquet (default: data/train_clean.parquet when present, "
                             "else data/train.parquet)")
    parser.add_argument("--cache-dir", type=Path, default=common.CACHE, help="dataset cache directory")
    parser.add_argument("--ppm-tol", type=float, default=10.0,
                        help="eligibility ppm for the training sampler (as in the make scripts)")
    parser.add_argument("--rebuild-cache", action="store_true", help="recompute the dataset key cache")
    parser.add_argument("--out-dir", type=Path, default=HERE / "out")
    tracking.add_arguments(parser)
    args = parser.parse_args()
    if args.model_path and args.leak_train:
        parser.error("--model-path cannot be combined with --leak-train")
    stem = args.split.stem + ("_leak" if args.leak_train else "") \
        + (f"_limit{args.limit}" if args.limit else "")
    with tracking.session(args, run_name=f"class2-stage2-{stem}",
                          tags={"script": "max_class2_exp/stage2.py", "stage": "stage2"}) as tracker:
        run(args, tracker, stem)


def run(args, tracker: tracking.Tracker, stem: str) -> None:
    t_start = time.time()
    params = CleanParams(tol=args.tol, noise=args.noise)
    workers = args.workers or max(1, (os.cpu_count() or 2) - 2)
    kinds = ["logistic", "hgb"] if args.model == "both" else [args.model]
    set_dropped_features([s.strip() for s in args.drop_features.split(",") if s.strip()],
                         keep_source=args.keep_source_features)
    k_neighbours = args.k_neighbours
    if args.enable_analogues and k_neighbours <= 0:
        k_neighbours = 8
    if args.leak_train and k_neighbours <= 0:      # the leak self-test exercises the analogue path
        k_neighbours = 8

    # ---- split, universe, eval pool
    split = load_split(args.split, args.train, args.cache_dir)
    info = split.info
    cache_dir = Path(info["params"]["cache_dir"])
    molecules = pd.read_parquet(cache_dir / "molecules.parquet")
    row_key = np.load(cache_dir / "row_key.npy")
    fps = common.fingerprint_matrix(molecules, cache_dir)
    truth = split.truth.reset_index(drop=True)
    if args.limit:
        truth = truth.iloc[: args.limit].reset_index(drop=True)
    key_index = pd.Series(np.arange(len(molecules)), index=molecules["key"])
    eval_mols = {int(key_index[k]) for k in split.truth["key"]}          # the FULL split (R1)
    visible_mols = np.unique(row_key[split.train_mask])
    external = pd.read_parquet(args.external_pool) if args.external_pool else None
    pool_eval, fps_ext = build_pool(molecules, row_key, split.train_mask, split.library_additions(),
                                    split.excluded_library_keys, external)
    fps_all = np.vstack([fps, fps_ext]) if len(fps_ext) else fps
    allowed_eval = visible_allowed(len(fps_all), visible_mols)
    pool_stats = {"n_members": len(pool_eval),
                  "n_visible_members": int((pool_eval.visible_rows > 0).sum()),
                  "n_additions": int(pool_eval.is_addition.sum()),
                  "n_external": int(pool_eval.is_external.sum())}
    common.log(f"[stage2] {args.split.name}: {len(truth)} eval molecules; pool {pool_stats['n_members']:,} "
              f"({pool_stats['n_visible_members']:,} visible / {pool_stats['n_additions']:,} additions); "
              f"all {len(split.truth)} split molecules excluded from clean training; "
              f"analogue channel {'ON' if k_neighbours > 0 else 'OFF'}")

    # ---- annotations (P0.2 formulas, P0.4 source stats)
    lib_codes, tims_flags, lib_names = load_row_sources(split.train_path, len(row_key))
    stats_eval = visible_source_stats(row_key, split.train_mask, lib_codes, tims_flags,
                                      lib_names, len(molecules))
    stats_all = visible_source_stats(row_key, np.ones(len(row_key), dtype=bool), lib_codes,
                                     tims_flags, lib_names, len(molecules))
    f_cache = formulas_for(np.concatenate([pool_eval.keys, truth["key"].to_numpy(dtype=object)]),
                           np.concatenate([pool_eval.smiles, truth["smiles"].to_numpy(dtype=object)]))
    annotate_pool_formulas(pool_eval, f_cache)
    annotate_pool_sources(pool_eval, stats_eval)
    truth_public = truth_public_flags(truth["key"].to_numpy(), stats_all, key_index)

    # ---- eval queries, neighbours, analogue library, features
    queries_eval, cands_union = build_eval_queries(split, truth, pool_eval, params, args.ppm)
    nbrs_eval = neighbour_lists(pool_eval, cands_union, allowed_eval, fps_all,
                                k_neighbours, args.min_neighbour_tc, workers, "[neighbours] eval")
    if args.leak_train:
        # the self-test pretends the truths *were* measured: their eval spectra
        # enter the library and each truth becomes its own top analogue (Tc 1.0)
        for q in queries_eval:
            if q["truth"] >= 0:
                m = int(key_index[q["key"]])
                nbrs_eval[q["truth"]] = (np.array([m], dtype=np.int64), np.array([1.0], dtype=np.float32))
        lib_eval = analogue_library(nbrs_eval, split.train_mask, row_key, split.train_path, params,
                                    info, {"stage2": "leak"}, workers, not args.no_cache,
                                    extra_rows=split.eval_row_ids)
    else:
        lib_eval = analogue_library(nbrs_eval, split.train_mask, row_key, split.train_path, params,
                                    info, {"stage2": "eval"}, workers, not args.no_cache)
        lib_rows_eval = lib_eval.table["row_id"].to_numpy(dtype=np.int64) if len(lib_eval) \
            else np.empty(0, dtype=np.int64)
        assert not np.isin(split.eval_row_ids, lib_rows_eval).any(), \
            "an eval row entered the analogue library"
    t0 = time.time()
    X_eval = featurize_queries(queries_eval, pool_eval, nbrs_eval, lib_eval, args.tol,
                               args.frag_ppm, workers, "[stage2] eval features", args.mtf_tol)
    t_eval_feat = time.time() - t0
    n_lib_eval = len(lib_eval)
    tracker.log_params({"n_molecules": len(truth), "n_analogue_spectra": n_lib_eval,
                        **pool_stats})
    tracker.log_metrics({"seconds/eval_features": round(t_eval_feat, 1)})

    # ---- training data on the fly, then fit (or load a saved model)
    if args.model_path:
        training = {"mode": "eval_only"}
        train_queries, X_train = [], []
        rng = np.random.default_rng(args.train_seed)
    elif args.leak_train:
        training = {"mode": "leak_train", "dropped_features": sorted(dropped_features())}
        train_queries, X_train = queries_eval, X_eval
        rng = np.random.default_rng(args.train_seed)
    else:
        training = {"mode": "clean", "dropped_features": sorted(dropped_features())}
        prep = SimpleNamespace(train=split.train_path, cache_dir=cache_dir,
                               rebuild_cache=args.rebuild_cache, ppm_tol=args.ppm_tol)
        meta, mol2, rk2 = common.prepare(prep)
        assert len(mol2) == len(molecules) and (rk2 == row_key).all(), "cache changed under us"
        rng = np.random.default_rng(args.train_seed)
        examples, shortfalls = sample_training_examples(args.n_train, rng, molecules, meta,
                                                        visible_mols, eval_mols)
        mask_train = split.train_mask.copy()
        for ex in examples:
            mask_train[ex["hidden_rows"]] = False          # R2: all rows of the pseudo-truth hidden
        additions_train = molecules.iloc[[e["mol"] for e in examples]][["key", "smiles"]]
        pool_train, _ = build_pool(molecules, row_key, mask_train, additions_train,
                                   split.excluded_library_keys, external)
        # the eval-time formula cache only covers pool_eval + truth; the
        # training pool also spans (most of) the visible corpus and the
        # sampled additions - extend it (incremental, disk-cached) so
        # cand_n_same_formula is computed from real formulas, not ""s
        f_cache = formulas_for(pool_train.keys, pool_train.smiles)
        annotate_pool_formulas(pool_train, f_cache)
        annotate_pool_sources(pool_train, visible_source_stats(
            row_key, mask_train, lib_codes, tims_flags, lib_names, len(molecules)))
        train_queries, dropped = build_training_queries(examples, molecules, pool_train, meta,
                                                       args.ppm, params, split.train_path)
        training.update(shortfalls=shortfalls, dropped=dropped)
        del meta, mol2, rk2                               # the 2.5M-row metadata frame is dead weight below
        # R1 / R2 asserts happen after the training library is built below
        cands_union_train = {int(c) for q in train_queries for c in q["cands"]}
        nbrs_train = neighbour_lists(pool_train, cands_union_train,
                                    visible_allowed(len(fps_all), np.unique(row_key[mask_train])),
                                    fps_all, k_neighbours, args.min_neighbour_tc, workers,
                                    "[neighbours] train")
        all_query_rows = np.unique(np.concatenate([e["query_rows"] for e in examples])) if examples \
            else np.empty(0, dtype=np.int64)
        lib_train = analogue_library(nbrs_train, mask_train, row_key, split.train_path,
                                    params, info,
                                    {"stage2": "train", "seed": args.train_seed,
                                     "n_train": args.n_train, "k": k_neighbours,
                                     "min_tc": args.min_neighbour_tc, "frag_ppm": args.frag_ppm},
                                    workers, not args.no_cache)
        # an empty library (analogue channel off) has an object-dtype row_id
        # column - normalise before the leakage asserts
        lib_rows = lib_train.table["row_id"].to_numpy(dtype=np.int64) if len(lib_train) \
            else np.empty(0, dtype=np.int64)
        assert split.train_mask[lib_rows].all(), \
            "R1: split-hidden row in the training library"
        assert not (set(e["mol"] for e in examples) & eval_mols), "R1: eval molecule as training query"
        assert not np.isin(all_query_rows, lib_rows).any(), \
            "R2: pseudo-query row in the training library"
        common.log(f"[stage2] training library: {len(lib_train):,} spectra rows")
        t0 = time.time()
        X_train = featurize_queries(train_queries, pool_train, nbrs_train, lib_train, args.tol,
                                    args.frag_ppm, workers, "[stage2] training features",
                                    args.mtf_tol)
        training["featurize_seconds"] = round(time.time() - t0, 1)
        del lib_train

    # ---- channels P (spectrum -> fingerprint) and F2 (ICEBERG in-silico):
    # one model / one prediction set per run, then their feature columns are
    # appended to every shortlist — eval, training and blind control go
    # through the same functions (R4). Leak-free is the DEFAULT, not a flag:
    #   * P v1 trains inside this run on split-visible rows only, minus the
    #     sampled pseudo-truths' rows (R1/R2 extended to channel P, asserted
    #     in fpchannel.build_fp_model);
    #   * P v2 (frozen artifacts) and any future self-trained F2 must prove
    #     via checkpoint metadata that no eval-split molecule was trained on
    #     — otherwise the run refuses to use them;
    #   * F2 here is the zero-shot public checkpoint (no Enveda spectrum in
    #     its training corpus), so no eval or pseudo-truth can be memorised.
    fp_mask = mask_train if training["mode"] == "clean" else split.train_mask
    fp_guard_rows = all_query_rows if training["mode"] == "clean" \
        else np.empty(0, dtype=np.int64)
    fp_model, fp_info = fpchannel.build_fp_model(
        fp_mask, row_key, split.train_path, fps, params, args.train_seed, workers,
        eval_row_ids=split.eval_row_ids, eval_keys={str(k) for k in split.truth["key"]},
        query_rows=fp_guard_rows)
    f2_reqs = f2_bridge.collect_requests(queries_eval, pool_eval)
    if training["mode"] == "clean":
        f2_reqs |= f2_bridge.collect_requests(train_queries, pool_train)
    f2_preds, f2_info = f2_bridge.ensure_predictions(f2_reqs, workers)
    append_channel_blocks(queries_eval, X_eval, pool_eval, fps_all, fp_model, f2_preds,
                          "[stage2] eval fp/f2 features")
    if training["mode"] == "clean":
        append_channel_blocks(train_queries, X_train, pool_train, fps_all, fp_model,
                              f2_preds, "[stage2] train fp/f2 features")
    tracker.log_params({"fp_channel": fp_info.get("kind"),
                        "f2_predictions_ready": f2_info.get("n_ready", 0)})

    # a shortlist can only train the ranker if its truth is inside the window:
    # at --ppm 8.5 a molecule's own rows can fall outside its window (they
    # become outside_window eval buckets but carry no positive label)
    keep = [bool(len(q["cands"])) and q["truth"] >= 0 and bool((q["cands"] == q["truth"]).any())
            for q in train_queries]
    n_outside = len(train_queries) - sum(keep)
    if n_outside:
        common.log(f"[stage2] {n_outside} training shortlists dropped (truth outside window)")
    training["n_truth_outside_window"] = n_outside
    train_queries = [q for q, k in zip(train_queries, keep) if k]
    X_train = [X for X, k in zip(X_train, keep) if k]
    y_per = [q["cands"] == q["truth"] for q in train_queries]
    for y in y_per:
        assert y.sum() == 1, "exactly one positive per shortlist"
    sizes = [len(q["cands"]) for q in train_queries]
    X_all = np.concatenate(X_train) if len(X_train) else np.empty((0, len(FEATURE_COLUMNS)))
    y_all = np.concatenate(y_per) if y_per else np.empty(0)
    training.update(n_molecules=len(train_queries), n_rows=len(y_all),
                    class_balance=round(float(y_all.mean()), 4) if len(y_all) else None,
                    median_candidates=float(np.median(sizes)) if sizes else None)
    if not args.model_path and len(y_all):
        common.log(f"[stage2] training set: {len(train_queries)} shortlists, {len(y_all):,} rows, "
                  f"{100 * y_all.mean():.2f}% positive")
    tracker.log_metrics({"training": training})

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
    Z_all = model_matrix(X_all, sizes) if len(y_all) else np.empty((0, len(FEATURES) - len(dropped_features())))
    tr = np.flatnonzero(~is_val)
    ends, starts = np.cumsum(sizes), np.cumsum(sizes) - np.asarray(sizes)

    def _rows(sel) -> np.ndarray:
        return np.concatenate([np.arange(starts[i], ends[i]) for i in sel]) if len(sel) \
            else np.empty(0, dtype=np.int64)

    models = fit_models(Z_all[_rows(tr)], y_all[_rows(tr)], args.train_seed, kinds,
                        args.model_seeds) \
        if len(tr) and len(y_all) else {}

    def val_mrr(model=None) -> float | None:
        rr = []
        for q, X, val in zip(train_queries, X_train, is_val):
            if not val or len(q["cands"]) == 0:
                continue
            s = predict(model, X, [len(q["cands"])]) if model is not None \
                else method_scores(X)["entropy_best"]
            rr.append(truth_rank(s, q, q["truth"])[1])
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

    importances = {}
    if "logistic" in models:
        importances["logistic"] = dict(zip(FEATURES, models["logistic"].first().coef_[0].round(6).tolist()))
    if "hgb" in models and is_val.any():
        from sklearn.inspection import permutation_importance
        va = np.flatnonzero(is_val)
        Xv = np.concatenate([X_train[i] for i in va])
        yv = np.concatenate([y_per[i] for i in va])
        sv = [len(train_queries[i]["cands"]) for i in va]
        r = permutation_importance(models["hgb"].first(), model_matrix(Xv, sv), yv, n_repeats=5,
                                   random_state=args.train_seed)
        importances["hgb"] = dict(zip(FEATURES, r.importances_mean.round(6).tolist()))
    training["feature_importances"] = importances

    if args.model_path:
        import joblib
        blob = joblib.load(args.model_path)
        assert blob.get("features") == FEATURES, \
            f"{args.model_path.name} was trained with a different feature set; retrain first"
        set_dropped_features(blob.get("dropped", []), replace=True)
        models = {blob["kind"]: blob["model"]}
        fitted, primary = [blob["kind"]], blob["kind"]
        training = {"mode": "eval_only", "model_path": str(args.model_path),
                    "train_seed_of_model": blob.get("train_seed"), "primary_model": primary,
                    "internal_validation": None, "feature_importances": None,
                    "dropped_features": sorted(dropped_features())}
        common.log(f"[stage2] loaded the {blob['kind']} model from {args.model_path.name}; "
                   f"training skipped (features match)")
    elif models:
        import joblib
        args.out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": models[primary], "kind": primary, "features": FEATURES,
                     "dropped": sorted(dropped_features()), "train_seed": args.train_seed},
                    args.out_dir / f"{stem}_stage2_model.joblib")

    # ---- evaluate on the untouched split
    def stage2_scores(name, X, q):
        return predict(models[name], X, [len(q["cands"])]) if len(q["cands"]) else np.empty(0)

    tc_band = pd.cut(truth["max_visible_tc"].astype(float), TC_BANDS,
                     labels=[f"{a:g}-{b:g}" for a, b in zip(TC_BANDS[:-1], TC_BANDS[1:])],
                     include_lowest=True)
    formulas_members = pool_eval.formulas
    rows, feat_rows = [], []
    for i, (r, q, X, band) in enumerate(zip(truth.itertuples(), queries_eval, X_eval, tc_band)):
        t = q["truth"]
        in_c = bool(t >= 0 and len(q["cands"]) and (q["cands"] == t).any())
        rec = {"molecule_id": r.molecule_id, "key": r.key, "difficulty": int(r.difficulty),
               "neighbours_removed": bool(r.neighbours_removed),
               "max_visible_tc": float(r.max_visible_tc), "tc_band": str(band),
               "has_public_spectra": bool(truth_public[i]),
               "n_eval_spectra": len(q["spectra"]),
               "n_query_adducts": q.get("n_query_adducts", 0),
               "n_query_polarities": q.get("n_query_polarities", 0),
               "has_precursor_peak": bool(q.get("has_precursor_peak", False)),
               "mass_agreement_ppm": q.get("mass_agreement_ppm", np.nan),
               "n_candidates": len(q["cands"]),
               "truth_in_candidates": in_c,
               "query_mass": q.get("mass", np.nan)}
        # ---- the same-formula isomer panel (P0.2)
        panel_pos, panel_top_pos, tpos, formula_share = np.empty(0, dtype=np.int64), \
            np.empty(0, dtype=np.int64), -1, np.nan
        if len(q["cands"]):
            fa = formulas_members[q["cands"]].astype(str)
            _, cnts = np.unique(fa, return_counts=True)
            formula_share = float(cnts.max()) / len(q["cands"])
        if in_c:
            tpos = int(np.flatnonzero(q["cands"] == t)[0])
            tf = str(formulas_members[t])
            panel_pos = np.flatnonzero(fa == tf)
            same = np.array([p for p in panel_pos if p != tpos], dtype=np.int64)
            if len(same):
                ref = stage2_scores(primary, X, q) if primary else -np.abs(q["ppm_err"])
                top10 = same[np.argsort(-ref[same], kind="stable")[:10]]
                panel_top_pos = np.sort(np.concatenate([[tpos], top10]))
            else:
                panel_top_pos = np.array([tpos], dtype=np.int64)
        rec.update(panel_size=len(panel_pos), paneltop_size=len(panel_top_pos),
                   formula_share=formula_share)
        if len(q["cands"]):
            fr = pd.DataFrame(X, columns=FEATURE_COLUMNS)
            fr.insert(0, "molecule_id", r.molecule_id)
            fr.insert(1, "difficulty", int(r.difficulty))
            fr.insert(2, "cand_key", pool_eval.keys[q["cands"]])
            fr.insert(3, "cand_smiles", pool_eval.smiles[q["cands"]])
            fr.insert(4, "label", (q["cands"] == t).astype(np.int8) if t >= 0 else 0)
            feat_rows.append(fr)

        def _record(meth: str, s: np.ndarray):
            rank, rr, ts, bs = truth_rank(s, q, t) if len(q["cands"]) else (0, 0.0, np.nan, np.nan)
            rec[f"rank_{meth}"], rec[f"rr_{meth}"] = rank, rr
            rec[f"truth_score_{meth}"], rec[f"top_score_{meth}"] = ts, bs
            rec[f"bucket_{meth}"] = failure_bucket(rank, in_c)
            if in_c:
                rec[f"panel_rank_{meth}"] = rank_within(s, panel_pos, tpos, np.abs(q["ppm_err"]))
                rec[f"paneltop_rank_{meth}"] = rank_within(s, panel_top_pos, tpos, np.abs(q["ppm_err"]))
            else:
                rec[f"panel_rank_{meth}"] = 0
                rec[f"paneltop_rank_{meth}"] = 0

        for meth, s in method_scores(X).items():
            _record(meth, s)
        for name in fitted:
            _record(f"stage2_{name}", stage2_scores(name, X, q))
        if primary:
            s = stage2_scores(primary, X, q)
            _record("stage2", s)                     # the plain fused ranking
            if len(q["cands"]):
                order = rank_order(s, np.abs(q["ppm_err"]), q["cands"])
                rec["top25_keys"] = pool_eval.keys[q["cands"][order[:TOP_K]]].tolist()
                rec["top25_smiles"] = pool_eval.smiles[q["cands"][order[:TOP_K]]].tolist()
                rec["top25_scores"] = s[order[:TOP_K]].round(4).tolist()
                if args.protect_top1:
                    ps = method_scores(X)[args.protect_top1]
                    protect_pos = int(rank_order(ps, np.abs(q["ppm_err"]), q["cands"])[0])
                    prot = protect_top1_order(order, protect_pos)
                    rank = rank_in_order(prot, tpos) if in_c else 0
                    rr = 1.0 / rank if 1 <= rank <= TOP_K else 0.0
                    rec["rank_stage2_protect"], rec["rr_stage2_protect"] = rank, rr
                    rec["truth_score_stage2_protect"] = float(s[tpos]) if in_c else np.nan
                    rec["top_score_stage2_protect"] = float(s[prot[0]])
                    rec["bucket_stage2_protect"] = failure_bucket(rank, in_c)
                    if in_c:
                        rec["panel_rank_stage2_protect"] = rank_in_order(prot, tpos, panel_pos)
                        rec["paneltop_rank_stage2_protect"] = rank_in_order(prot, tpos, panel_top_pos)
                    else:
                        rec["panel_rank_stage2_protect"] = 0
                        rec["paneltop_rank_stage2_protect"] = 0
        rows.append(rec)
    res = pd.DataFrame(rows)

    def _panel_block(df: pd.DataFrame, meth: str, col: str) -> dict:
        pr = df.get(f"{col}_{meth}")
        if pr is None:
            return {"n": 0}
        has = np.asarray(pr) > 0
        out = {"n": int(has.sum())}
        if has.any():
            ph = np.asarray(pr)[has]
            prr = np.where(ph <= TOP_K, 1.0 / ph, 0.0)
            out.update(top1=round(float((ph == 1).mean()), 4),
                       **{"mrr@25": round(float(prr.mean()), 4)})
            size_col = "panel_size" if col == "panel_rank" else "paneltop_size"
            out["median_size"] = float(df.loc[has, size_col].median())
        return out

    def summary(df: pd.DataFrame) -> dict:
        if not len(df):
            return {"n": 0}
        d = {"n": len(df), "candidate_recall": round(float(df["truth_in_candidates"].mean()), 4),
             "median_candidates": float(df["n_candidates"].median())}
        methods = list(BASELINES) + [f"stage2_{k}" for k in fitted]
        if primary:
            methods.append("stage2")
        if args.protect_top1 and primary:
            methods.append("stage2_protect")
        methods = [m for m in methods if f"rank_{m}" in df.columns]
        in_c = df["truth_in_candidates"].astype(bool)
        for meth in methods:
            rk = df[f"rank_{meth}"]
            rr = df[f"rr_{meth}"]
            blk = {"mrr@25": round(float(rr.mean()), 4),
                   "top1": round(float((rk == 1).mean()), 4),
                   "top10": round(float(((rk >= 1) & (rk <= 10)).mean()), 4),
                   "top25": round(float(((rk >= 1) & (rk <= 25)).mean()), 4)}
            if in_c.any():
                blk["mrr_conditional_on_recall"] = round(float(rr[in_c].mean()), 4)
            blk["buckets"] = {b: int((df[f"bucket_{meth}"] == b).sum()) for b in BUCKETS}
            blk["panel"] = _panel_block(df, meth, "panel_rank")
            blk["panel_top"] = _panel_block(df, meth, "paneltop_rank")
            d[meth] = blk
        return d

    metrics = {"split": str(args.split),
               "params": {k: str(v) for k, v in tracking.public_args(args).items()},
               "training": training, "pool": pool_stats,
               "class_share_scenarios": {
                   "note": "class shares are unknown; expected overall LB gain = "
                           "share_class2 x delta_MRR_class2 (FORUM_FINDINGS.md P0.1)",
                   **{k: {"class2_share": v} for k, v in CLASS_SHARE_SCENARIOS.items()}},
               "overall": summary(res),
               "by_tc_band": {b: summary(g) for b, g in res.groupby("tc_band", observed=False)},
               "by_neighbours_removed": {str(b): summary(g) for b, g in
                                         res.groupby("neighbours_removed", observed=False)},
               "by_public_spectra": {str(b): summary(g) for b, g in
                                     res.groupby("has_public_spectra", observed=False)},
               "analogue_spectra": n_lib_eval,
               "fp_channel": fp_info,
               "f2_channel": f2_info,
               "seconds": {"eval_features": round(t_eval_feat, 1),
                           "total": round(time.time() - t_start, 1)}}
    if res["difficulty"].nunique() > 1:
        metrics["by_difficulty"] = {str(d): summary(g) for d, g in res.groupby("difficulty")}
    if res["has_public_spectra"].nunique() > 1:
        metrics["public_spectra_note"] = ("has_public_spectra = the truth structure has rows in "
                                          "a public library (approximate MassSpecGym overlap; "
                                          "the no-public-spectra slice is the honest headline "
                                          "for pretrained channels - FORUM_FINDINGS.md P0.4)")
    # P0.6: paired-by-molecule bootstrap of stage2 vs each channel baseline
    if args.n_boot and primary and len(res):
        boot = {}
        for meth in BASELINES:
            b = paired_bootstrap(res["rr_stage2"].to_numpy(), res[f"rr_{meth}"].to_numpy(),
                                 args.n_boot, args.train_seed)
            if b:
                boot[meth] = b
        if boot:
            metrics["bootstrap_stage2_vs"] = boot

    # ---- P0.5: spectrum-blind control (same windows, swapped spectra)
    if args.blind_control:
        rng_b = np.random.default_rng(args.train_seed + 1)
        offset = 1 + int(rng_b.integers(max(1, len(queries_eval) - 1)))
        blind = blind_queries(queries_eval, offset)
        if len(blind) >= 2:
            X_blind = featurize_queries(blind, pool_eval, nbrs_eval, lib_eval, args.tol,
                                        args.frag_ppm, workers, "[stage2] blind features",
                                        args.mtf_tol)
            # the swapped spectra re-use the eval prediction universe (same
            # candidates, same (adduct, ce) combos) - no new F2 predictions
            append_channel_blocks(blind, X_blind, pool_eval, fps_all, fp_model, f2_preds,
                                  "[stage2] blind fp/f2 features")
            blind_rows = []
            for q, X in zip(blind, X_blind):
                t = q["truth"]
                in_c = bool(t >= 0 and len(q["cands"]) and (q["cands"] == t).any())
                rec_b = {"n_candidates": len(q["cands"]), "truth_in_candidates": in_c}

                def _brec(meth: str, s: np.ndarray):
                    rank, rr = (truth_rank(s, q, t)[:2] if len(q["cands"]) else (0, 0.0))
                    rec_b[f"rank_{meth}"], rec_b[f"rr_{meth}"] = rank, rr
                    rec_b[f"bucket_{meth}"] = failure_bucket(rank, in_c)

                for meth, s in method_scores(X).items():
                    _brec(meth, s)
                for name in fitted:
                    _brec(f"stage2_{name}", stage2_scores(name, X, q))
                if primary:
                    _brec("stage2", stage2_scores(primary, X, q))
                blind_rows.append(rec_b)
            bres = pd.DataFrame(blind_rows)
            bsum = {"offset": offset, "overall": summary(bres)}
            if bsum["overall"]["candidate_recall"] == 0:
                bsum["vacuous"] = "candidate coverage is 0; the blind control says nothing"
                common.log("[stage2] WARNING: blind control is vacuous (coverage 0)")
            metrics["blind_control"] = bsum
            common.log(f"[stage2] blind control scored {len(bres)} queries "
                       f"(mass-rotation offset {offset})")
        else:
            metrics["blind_control"] = {"skipped": "fewer than 2 usable queries"}
    del lib_eval

    args.out_dir.mkdir(parents=True, exist_ok=True)
    res.to_parquet(args.out_dir / f"{stem}_stage2_results.parquet")
    if feat_rows:
        pd.concat(feat_rows, ignore_index=True).to_parquet(
            args.out_dir / f"{stem}_stage2_features.parquet")
    (args.out_dir / f"{stem}_stage2_metrics.json").write_text(json.dumps(metrics, indent=1))

    # ---- printed table
    table_methods = list(BASELINES) + [f"stage2_{k}" for k in fitted]
    if primary:
        table_methods.append("stage2")
    if args.protect_top1 and primary:
        table_methods.append("stage2_protect")
    groups = [(g, m) for g, m in [("overall", metrics["overall"])] + list(metrics["by_tc_band"].items())
              if m.get("n")]
    common.log("")
    common.log(f"MRR@25 (top-1 / top-10)   {args.split.name}"
               + ("   [LEAK SELF-TEST]" if args.leak_train else ""))
    common.log(f"{'':20s}" + "".join(f"{g:>22s}" for g, _ in groups))
    common.log(f"{'n molecules':20s}" + "".join(f"{m['n']:>22d}" for _, m in groups))
    common.log(f"{'candidate recall':20s}" + "".join(f"{m['candidate_recall']:>22.3f}" for _, m in groups))
    for meth in table_methods:
        if not all(meth in m for _, m in groups):    # keep the table rectangular
            continue
        label = meth + (" *" if meth == "stage2" else "")
        common.log(f"{label:20s}" + "".join(
            f"{m[meth]['mrr@25']:>10.3f} ({m[meth]['top1']:.2f}/{m[meth]['top10']:.2f})"
            for _, m in groups))
    ov = metrics["overall"]
    common.log("")
    common.log("same-formula isomer panel (P0.2) - the metric class 2 is actually about:")
    for meth in table_methods:
        p, pt = ov[meth]["panel"], ov[meth]["panel_top"]
        if p.get("n"):
            common.log(f"  {meth:20s} top1={p['top1']:.3f} mrr@25={p['mrr@25']:.3f} "
                       f"median={p['median_size']:.0f} | truth+top10 decoys: "
                       f"top1={pt.get('top1', float('nan')):.3f} "
                       f"mrr@25={pt.get('mrr@25', float('nan')):.3f}")
    if "bootstrap_stage2_vs" in metrics:
        common.log("")
        common.log("paired bootstrap, stage2 vs channel (mean delta [95% CI], p(delta>0)):")
        for meth, b in metrics["bootstrap_stage2_vs"].items():
            common.log(f"  vs {meth:18s} {b['mean_delta']:+.4f} "
                       f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] p={b['p_delta_gt_0']:.3f}")
    if "by_public_spectra" in metrics and primary:
        common.log("")
        for flag, m in metrics["by_public_spectra"].items():
            if not m.get("n"):
                continue
            label = "public spectra" if flag == "True" else "no public spectra"
            common.log(f"{label:20s} n={m['n']} stage2 mrr@25={m['stage2']['mrr@25']:.3f} "
                       f"(panel {m['stage2']['panel'].get('mrr@25', float('nan')):.3f})")
    if isinstance(metrics.get("blind_control"), dict) and "overall" in metrics["blind_control"]:
        b = metrics["blind_control"]["overall"]
        line = f"blind control (spectra swapped, same windows): mass_error {b['mass_error']['mrr@25']:.3f} " \
               f"(sighted {ov['mass_error']['mrr@25']:.3f})"
        if primary:
            line += f"; stage2 {b['stage2']['mrr@25']:.3f} (sighted {ov['stage2']['mrr@25']:.3f})"
        common.log("")
        common.log(line)
    common.log(f"\n[stage2] model: {primary} (internal validation "
               f"{json.dumps(internal)}); results in {args.out_dir}/{stem}_stage2_*; "
               f"total {time.time() - t_start:.0f}s")

    artifacts = [args.out_dir / f"{stem}_stage2_results.parquet",
                 args.out_dir / f"{stem}_stage2_metrics.json"]
    model_file = Path(args.model_path) if args.model_path else \
        (args.out_dir / f"{args.split.stem}_stage2_model.joblib")
    if model_file.exists():
        artifacts.append(model_file)
    tracker.log_params({"primary_model": str(primary)})
    tracker.log_metrics(metrics)
    for artifact in artifacts:
        tracker.log_artifact(artifact)


if __name__ == "__main__":
    main()
