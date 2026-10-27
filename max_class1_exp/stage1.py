"""Stage 1 of the class-1 experiment: mass window + spectral library search.

For every eval molecule of a split (default: dataset-creation/out/difficulty1.parquet):
  1. candidates = visible train molecules within --ppm of the query's neutral mass;
  2. every eval spectrum is compared with every visible library spectrum of every
     candidate (cleaned as in Li et al. 2021);
  3. candidate score = mean over the query's spectra of the best match, ranked
     descending (ties: smaller mass error, then key).

Four scoring variants are evaluated side by side:
  cosine_best        plain cosine, best library spectrum
  entropy_best       weighted entropy similarity, best library spectrum
  entropy_composite  weighted entropy vs. one merged spectrum per candidate and
                     polarity (all its library spectra averaged, CE-robust)
  entropy_combined   mean of entropy_best and entropy_composite

Outputs (max_class1_exp/out/): <split>_stage1_results.parquet (one row per
molecule: candidates, rank of the answer per variant, top 25) and
<split>_stage1_metrics.json; a table is printed.

Usage:
    uv run python max_class1_exp/stage1.py [--split dataset-creation/out/difficulty1.parquet]
        [--ppm 10] [--tol 0.02] [--noise 0.01] [--limit N] [--workers N] [--no-cache]
        [--leak-eval]     # self-test: put the eval spectra into the library (expect MRR ~1)
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from library import CandidateIndex, build_library, query_mass  # noqa: E402
from enveda_casmi26 import tracking  # noqa: E402
import common  # noqa: E402  (dataset-creation, put on sys.path by library)
from dataloader import load_split  # noqa: E402
from spectra import (CleanParams, clean_spectrum, cosine_one_vs_many, entropy_one_vs_many,  # noqa: E402
                     entropy_weight, merge_spectra, unit_norm)

METHODS = ["cosine_best", "entropy_best", "entropy_composite", "entropy_combined"]
TOP_K = 25

# state shared with forked workers
_LIB = None
_QUERIES = None
_PARAMS = None
_COMPOSITE: dict = {}


# ------------------------------------------------------------------ scoring

def _composite(m: int, positive: bool):
    """Merged, entropy-weighted spectrum of molecule m in one polarity (cached per worker)."""
    k = (m, positive)
    if k not in _COMPOSITE:
        s, e = _LIB.spectra_of(m)
        spectra = [_LIB.get(i) for i in range(s, e) if _LIB.positive[i] == positive]
        mz, p = merge_spectra(spectra, _PARAMS.tol)
        _COMPOSITE[k] = (mz, entropy_weight(p))
    return _COMPOSITE[k]


def score_molecule(i: int) -> dict:
    """Scores of all candidates of query molecule i for every method."""
    q = _QUERIES[i]
    cands = q["cands"]
    n = len(cands)
    out = {m: np.zeros(n) for m in METHODS}
    if n == 0:
        return out
    # library spectra of all candidates, contiguous per candidate
    ranges = [_LIB.spectra_of(int(m)) for m in cands]
    n_spec = np.array([e - s for s, e in ranges])
    assert (n_spec > 0).all(), "a candidate without library spectra"
    spec_idx = np.concatenate([np.arange(s, e) for s, e in ranges])
    peak_idx = np.concatenate([np.arange(_LIB.peak_start[s], _LIB.peak_start[e]) for s, e in ranges])
    local_spec = np.repeat(np.arange(len(spec_idx)), np.diff(_LIB.peak_start)[spec_idx])
    l_mz, l_pw, l_pu = _LIB.mz[peak_idx], _LIB.pw[peak_idx], _LIB.pu[peak_idx]
    cand_start = np.r_[0, np.cumsum(n_spec)[:-1]]

    best_e, best_c, comp, used = np.zeros(n), np.zeros(n), np.zeros(n), 0
    for mz, p, positive in q["spectra"]:
        if len(mz) == 0:
            continue
        used += 1
        pw, pu = entropy_weight(p), unit_norm(p)
        se = entropy_one_vs_many(mz, pw, l_mz, l_pw, local_spec, len(spec_idx), _PARAMS.tol)
        sc = cosine_one_vs_many(mz, pu, l_mz, l_pu, local_spec, len(spec_idx), _PARAMS.tol)
        best_e += np.maximum.reduceat(se, cand_start)
        best_c += np.maximum.reduceat(sc, cand_start)
        # composite: one merged spectrum per candidate in the query's polarity
        cm = [_composite(int(m), positive) for m in cands]
        c_lens = np.array([len(x[0]) for x in cm])
        if c_lens.sum():
            c_mz = np.concatenate([x[0] for x in cm])
            c_pw = np.concatenate([x[1] for x in cm])
            c_spec = np.repeat(np.arange(n), c_lens)
            comp += entropy_one_vs_many(mz, pw, c_mz, c_pw, c_spec, n, _PARAMS.tol)
    if used:
        out["cosine_best"] = best_c / used
        out["entropy_best"] = best_e / used
        out["entropy_composite"] = comp / used
        out["entropy_combined"] = (best_e + comp) / (2 * used)
    return out


def rank_order(scores: np.ndarray, abs_ppm: np.ndarray, mols: np.ndarray) -> np.ndarray:
    """Candidate order: score desc, then mass error asc, then molecule index (= key order)."""
    return np.lexsort((mols, abs_ppm, -scores))


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
    parser.add_argument("--limit", type=int, default=0, help="only the first N molecules (quick test)")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes (default: CPUs - 2)")
    parser.add_argument("--no-cache", action="store_true", help="rebuild the library cache")
    parser.add_argument("--leak-eval", action="store_true",
                        help="self-test: add the eval spectra to the library; MRR must be ~1")
    parser.add_argument("--out-dir", type=Path, default=HERE / "out")
    tracking.add_arguments(parser)
    args = parser.parse_args()
    t_start = time.time()
    _PARAMS = CleanParams(tol=args.tol, noise=args.noise)
    workers = args.workers or max(1, (os.cpu_count() or 2) - 2)

    # ---- split, cache, eval spectra
    split = load_split(args.split)
    info = split.info
    cache_dir = Path(info["params"]["cache_dir"])
    molecules = pd.read_parquet(cache_dir / "molecules.parquet")
    row_key = np.load(cache_dir / "row_key.npy")
    truth = split.truth.reset_index(drop=True)
    if args.limit:
        truth = truth.iloc[: args.limit].reset_index(drop=True)
    key_index = pd.Series(np.arange(len(molecules)), index=molecules["key"])
    common.log(f"[stage1] {args.split.name}: {len(truth)} molecules; building eval spectra ...")
    ev = split.eval_spectra()
    ev = ev[ev["molecule_id"].isin(truth["molecule_id"])]

    # ---- queries and candidates
    cidx = CandidateIndex(molecules, row_key, split.train_mask)
    queries, need = [], set()
    groups = dict(tuple(ev.groupby("molecule_id", sort=False)))
    for r in truth.itertuples():
        g = groups.get(r.molecule_id)
        if g is None:
            queries.append({"cands": np.empty(0, dtype=np.int64), "ppm_err": np.empty(0), "spectra": []})
            continue
        mass = query_mass(g["precursor_mz"], g["adduct"])
        cands, err = cidx.window(mass, args.ppm)
        if args.leak_eval:
            t = int(key_index[r.key])
            if t not in set(cands.tolist()):
                cands = np.r_[cands, t]
                err = np.r_[err, (molecules["exact_mass"].iat[t] - mass) / mass * 1e6]
        spectra = []
        for mz, it, pr, mode in zip(g["ms2_mzs"], g["ms2_normalized_intensities"], g["precursor_mz"],
                                    g["ionization_mode"]):
            cm, cp = clean_spectrum(mz, it, pr, _PARAMS)
            spectra.append((cm, cp, mode == "positive"))
        queries.append({"cands": cands.astype(np.int64), "ppm_err": err, "spectra": spectra, "mass": mass})
        need.update(cands.tolist())

    # ---- library rows: every visible row of every candidate
    need = np.array(sorted(need), dtype=np.int64)
    lib_rows = np.flatnonzero(split.train_mask & np.isin(row_key, need))
    if args.leak_eval:
        lib_rows = np.union1d(lib_rows, split.eval_row_ids)
    else:
        # leakage guards: library = visible rows only, never an eval row
        assert split.train_mask[lib_rows].all()
        assert not np.isin(split.eval_row_ids, lib_rows).any()
    _LIB = build_library(lib_rows, row_key, split.train_path, _PARAMS,
                         {**info, "leak": args.leak_eval}, workers, use_cache=not args.no_cache)
    _QUERIES = queries

    # ---- score
    t0 = time.time()
    with Pool(workers) as pool:
        prog = common.Progress("[stage1] molecules", len(queries))
        scores = []
        for s in pool.imap(score_molecule, range(len(queries)), chunksize=4):
            scores.append(s)
            prog.update()
        prog.close()
    t_score = time.time() - t0

    # ---- rank and evaluate
    rows = []
    for r, q, s in zip(truth.itertuples(), queries, scores):
        t = int(key_index[r.key])
        cands = q["cands"]
        pos = np.flatnonzero(cands == t)
        rec = {"molecule_id": r.molecule_id, "key": r.key, "change": r.change, "difficulty": int(r.difficulty),
               "n_eval_spectra": len(q["spectra"]), "n_candidates": len(cands),
               "n_library_spectra": int(sum(e - b for b, e in (_LIB.spectra_of(int(m)) for m in cands))),
               "truth_in_candidates": bool(len(pos)), "query_mass": q.get("mass", np.nan)}
        for meth in METHODS:
            order = rank_order(s[meth], np.abs(q["ppm_err"]), cands)
            rank = int(np.flatnonzero(order == pos[0])[0]) + 1 if len(pos) else 0
            rec[f"rank_{meth}"] = rank
            rec[f"rr_{meth}"] = 1.0 / rank if 0 < rank <= TOP_K else 0.0
            rec[f"truth_score_{meth}"] = float(s[meth][pos[0]]) if len(pos) else np.nan
            rec[f"top_score_{meth}"] = float(s[meth].max()) if len(cands) else np.nan
            if meth == "entropy_combined":
                top = order[:TOP_K]
                rec["top25_keys"] = molecules["key"].to_numpy()[cands[top]].tolist()
                rec["top25_smiles"] = molecules["smiles"].to_numpy()[cands[top]].tolist()
                rec["top25_scores"] = s[meth][top].round(4).tolist()
        rows.append(rec)
    res = pd.DataFrame(rows)

    def summary(df: pd.DataFrame) -> dict:
        d = {"n": len(df), "candidate_recall": round(float(df["truth_in_candidates"].mean()), 4),
             "median_candidates": float(df["n_candidates"].median())}
        for meth in METHODS:
            rk = df[f"rank_{meth}"]
            d[meth] = {"mrr@25": round(float(df[f"rr_{meth}"].mean()), 4),
                       "top1": round(float(((rk >= 1) & (rk <= 1)).mean()), 4),
                       "top10": round(float(((rk >= 1) & (rk <= 10)).mean()), 4),
                       "top25": round(float(((rk >= 1) & (rk <= 25)).mean()), 4)}
        return d

    metrics = {"split": str(args.split),
               "params": {k: str(v) for k, v in tracking.public_args(args).items()},
               "overall": summary(res),
               "by_change": {c: summary(g) for c, g in res.groupby(res["change"].fillna("none"))},
               "library_spectra": len(_LIB), "seconds": {"scoring": round(t_score, 1),
                                                         "total": round(time.time() - t_start, 1)}}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.split.stem + ("_leak" if args.leak_eval else "") + (f"_limit{args.limit}" if args.limit else "")
    res.to_parquet(args.out_dir / f"{stem}_stage1_results.parquet")
    (args.out_dir / f"{stem}_stage1_metrics.json").write_text(json.dumps(metrics, indent=1))

    # ---- table
    groups = [("overall", metrics["overall"])] + list(metrics["by_change"].items())
    common.log("")
    common.log(f"MRR@25 (top-1 / top-10)   {args.split.name}" + ("   [LEAK SELF-TEST]" if args.leak_eval else ""))
    common.log(f"{'':20s}" + "".join(f"{g:>22s}" for g, _ in groups))
    common.log(f"{'n molecules':20s}" + "".join(f"{m['n']:>22d}" for _, m in groups))
    common.log(f"{'candidate recall':20s}" + "".join(f"{m['candidate_recall']:>22.3f}" for _, m in groups))
    common.log(f"{'median candidates':20s}" + "".join(f"{m['median_candidates']:>22.0f}" for _, m in groups))
    for meth in METHODS:
        common.log(f"{meth:20s}" + "".join(
            f"{m[meth]['mrr@25']:>10.3f} ({m[meth]['top1']:.2f}/{m[meth]['top10']:.2f})" for _, m in groups))
    common.log(f"\n[stage1] scoring {t_score:.0f}s, total {time.time() - t_start:.0f}s; "
               f"results in {args.out_dir}/{stem}_stage1_*.")
    tracking.log_run(
        args,
        params={"n_molecules": len(res), "n_library_spectra": len(_LIB)},
        metrics=metrics,
        artifacts=[args.out_dir / f"{stem}_stage1_results.parquet",
                   args.out_dir / f"{stem}_stage1_metrics.json"],
        run_name=f"stage1-{stem}",
        tags={"script": "max_class1_exp/stage1.py", "stage": "stage1"},
    )


if __name__ == "__main__":
    main()
