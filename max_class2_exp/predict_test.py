"""Run the class-2 fusion re-ranker on the competition test set; build submission.csv.

Serving path for the real test set (no eval split, no simulator): every train
row is visible, the candidate pool is the train structures themselves — no
library additions, because a test truth that is absent from train can never be
added by name — and each test molecule's shortlist is featurized with the same
`featurize` as training and evaluation (R4):

  in-silico / metfrag-lite   live (RDKit; the Kaggle kernel installs it from
                             a wheel shipped in the attached dataset);
  channel P                  the frozen v2 artifacts from --artifacts (the
                             guarded load trivially passes with no eval
                             keys). --artifacts MUST contain them: the fusion
                             model was trained on v2 fp features, so a silent
                             v1 fallback would skew serving — predict refuses
                             instead (--allow-v1-fp opts into the fallback);
  channel F2                 always off in serving (no ms-pred env); its
                             columns stay 0 with f2_available=0 — serve a
                             fusion model trained in the same configuration
                             (the box's difficulty2 model was trained F2-off)
                             or measure the skew with --drop-features first;
  analogue channel           off (an empty library, like stage2's default).

The fusion model comes from --model (a <split>_stage2_model.joblib saved by
stage2.py); its `dropped` feature set is restored exactly like stage2's
--model-path mode (the visibility/source block is dropped by default, so the
absence of library additions at serve time cannot influence the ranking).

The molecule-key cache is READ, never rebuilt: serving needs the very cache
the model was trained with (byte-identical molecules.parquet -> row_key.npy ->
fps.npy chain), and an upload mtime change would otherwise trigger a silent
rebuild inside common.ensure_cache. In the Kaggle kernel all of train, cache,
artifacts and model arrive in the attached dataset.

Every test molecule must appear exactly once with at least one guess, so
shortlists shorter than --top (and molecules with no candidates or an
unusable query mass) are padded with the pool structures nearest in exact mass
(the pool is mass-sorted), falling back to the most-spectra structures when
the query mass is unusable (a wrong guess costs only its rank; a missing row
would be rejected).

Works locally and inside the Kaggle kernel (same code path):
    uv run python max_class2_exp/predict_test.py --test data/test.parquet

Outputs:
    --out    submission.csv (molecule_id, smiles; up to --top ';'-joined guesses)
    --diag   per-molecule diagnostics parquet
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "dataset-creation"))

from annotations import (annotate_pool_formulas, annotate_pool_sources,  # noqa: E402
                          formulas_for, load_row_sources, visible_source_stats)
import common  # noqa: E402
from dataloader import clean_peaks, resolve_train_path  # noqa: E402
from features import (FEATURES, append_channel_blocks, featurize_queries,  # noqa: E402
                      predict, rank_order, set_dropped_features)
import fpchannel  # noqa: E402
from pool import build_pool  # noqa: E402
from spectra_utils import CleanParams, clean_spectrum  # noqa: E402
from speclib import build_library, query_mass  # noqa: E402


def query_spectra(g, params: CleanParams) -> list[tuple]:
    """Cleaned query spectra of one test molecule, test-like cleaning
    (dataloader.clean_peaks, then clean_spectrum — exactly the class-1
    predict path and the stage2 training rows).

    (mz, p, positive, precursor_mz, adduct, ce_ev) per spectrum."""
    out = []
    cev = g["collision_energy_ev"] if "collision_energy_ev" in g.columns \
        else [None] * len(g)
    for mz, it, prec, mode, adduct, ce in zip(g["ms2_mzs"], g["ms2_normalized_intensities"],
                                              g["precursor_mz"], g["ionization_mode"],
                                              g["adduct"], cev):
        m, i = clean_peaks(mz, it, float(prec))
        if m is None:
            continue
        cm, cp = clean_spectrum(m, i, float(prec), params)
        ce = np.nan if ce is None or len(ce) == 0 else ce      # class-1's normalization
        out.append((cm, cp, mode == "positive", float(prec), adduct, fpchannel.ce_of(ce)))
    return out


def pad_from_pool(q, pool, have: list, top: int) -> list:
    """Fill `have` up to `top` guesses from the mass-nearest pool structures
    (the pool is mass-sorted; dedup by SMILES). Unusable query mass falls
    back to the structures with the most visible rows."""
    if len(have) >= top:
        return have
    used = set(have)
    if np.isfinite(q["mass"]):
        pos = int(np.searchsorted(pool.mass, q["mass"]))
        lo, hi = max(0, pos - 2500), min(len(pool), pos + 2500)
        cand = lo + np.argsort(np.abs(pool.mass[lo:hi] - q["mass"]), kind="stable")
    else:
        cand = np.argsort(-pool.visible_rows, kind="stable")
    for j in cand:
        if len(have) >= top:
            break
        s = pool.smiles[int(j)]
        if s is None or s in used:
            continue
        used.add(s)
        have.append(s)
    return have


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", type=Path, default=None,
                        help="test parquet (default: data/test_clean.parquet when present, "
                             "else data/test.parquet)")
    parser.add_argument("--train", type=Path, default=None,
                        help="train parquet (default: cleaned copy when present, else raw; "
                             "the kernel passes the dataset's train_clean.parquet)")
    parser.add_argument("--cache-dir", type=Path, default=common.CACHE,
                        help="the molecule-key cache the model was trained with (READ only, "
                             "never rebuilt here)")
    parser.add_argument("--artifacts", type=Path, default=HERE / "out",
                        help="channel-P v2 artifacts dir (denovo_v1.pt + denovo_fp_head.pt)")
    parser.add_argument("--model", type=Path,
                        default=HERE / "out" / "difficulty2_stage2_model.joblib",
                        help="fusion model saved by stage2.py (<split>_stage2_model.joblib)")
    parser.add_argument("--allow-v1-fp", action="store_true",
                        help="let channel P fall back to a v1 linear model when the v2 "
                             "artifacts are absent (skews serving vs training; off by default)")
    parser.add_argument("--ppm", type=float, default=8.5, help="candidate mass window (+/- ppm)")
    parser.add_argument("--tol", type=float, default=0.01, help="m/z tolerance for peaks (Da)")
    parser.add_argument("--noise", type=float, default=0.01,
                        help="drop peaks below this share of the base peak")
    parser.add_argument("--frag-ppm", type=float, default=10.0,
                        help="m/z tolerance (ppm) for matching in-silico fragments")
    parser.add_argument("--mtf-tol", type=float, default=0.01,
                        help="m/z tolerance (Da) for the metfrag-lite channel")
    parser.add_argument("--top", type=int, default=25, help="guesses per molecule (competition: 25)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N test molecules")
    parser.add_argument("--external-pool", type=Path, default=None,
                        help="optional extra structures (key/inchikey14, smiles[, mass]) to add "
                             "to the candidate pool")
    parser.add_argument("--out", type=Path, default=common.PROJECT_ROOT / "submissions"
                        / "class2_stage2_submission.csv", help="output submission csv")
    parser.add_argument("--diag", type=Path, default=HERE / "out" / "test_stage2_diag.parquet",
                        help="per-molecule diagnostics parquet ('' to skip)")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes")
    args = parser.parse_args()
    if args.test is None:
        args.test = common.prefer_cleaned(common.TRAIN.parent / "test.parquet")
    if args.train is None:
        args.train = resolve_train_path()
    common.log(f"[predict] train {args.train} | test {args.test} | cache {args.cache_dir}")
    t_start = time.time()
    params = CleanParams(tol=args.tol, noise=args.noise)
    workers = args.workers or max(1, (os.cpu_count() or 2) - 2)

    # ---- the fusion model first: fail fast on a stale feature set
    import joblib
    blob = joblib.load(args.model)
    assert blob.get("features") == FEATURES, \
        f"{args.model.name} was trained with a different feature set; retrain first"
    model = blob["model"]
    set_dropped_features(blob.get("dropped") or [], replace=True)
    common.log(f"[predict] {blob['kind']} model from {args.model.name} "
               f"(seed {blob.get('train_seed')}, dropped {sorted(blob.get('dropped') or [])})")

    # ---- channel-P v2 artifacts must exist (no silent v1 fallback in serving)
    art = Path(args.artifacts)
    enc_path = next((art / f for f in ("denovo_v1.pt", "denovo_v1.pt.last.pt")
                     if (art / f).exists()), None)
    head_path = art / "denovo_fp_head.pt"
    if (enc_path is None or not head_path.exists()) and not args.allow_v1_fp:
        raise SystemExit(f"[predict] channel-P v2 artifacts missing in {art} "
                         f"(need denovo_v1.pt[.last.pt] + denovo_fp_head.pt); the fusion model "
                         "was trained on v2 fp features - pass --allow-v1-fp to serve with a "
                         "v1 model anyway (not recommended)")

    # ---- the training cache, read as shipped (never rebuilt: an upload mtime
    # change would otherwise trigger a silent ensure_cache rebuild)
    cache = Path(args.cache_dir)
    if not (cache / "molecules.parquet").exists() or not (cache / "row_key.npy").exists():
        raise SystemExit(f"[predict] cache files missing in {cache} — serve with the cache "
                         "the model was trained with (dataset-creation/out/cache)")
    molecules = pd.read_parquet(cache / "molecules.parquet")
    row_key = np.load(cache / "row_key.npy")
    fps = common.fingerprint_matrix(molecules, cache)
    common.log(f"[predict] {len(molecules):,} molecules / {len(row_key):,} rows "
               f"(cache sha {common.cache_hash(cache)[:12]}, read-only)")
    mask = np.ones(len(row_key), dtype=bool)

    # ---- the pool: every train structure with >= 1 row (all rows visible;
    # no additions — a test truth absent from train cannot be added by name)
    external = pd.read_parquet(args.external_pool) if args.external_pool else None
    pool, fps_ext = build_pool(molecules, row_key, mask, None, (), external)
    fps_all = np.vstack([fps, fps_ext]) if len(fps_ext) else fps

    # ---- pool annotations (formulas for cand_n_same_formula, source stats)
    lib_codes, tims_flags, lib_names = load_row_sources(args.train, len(row_key))
    annotate_pool_sources(pool, visible_source_stats(row_key, mask, lib_codes, tims_flags,
                                                     lib_names, len(molecules)))
    f_cache = formulas_for(pool.keys, pool.smiles)
    annotate_pool_formulas(pool, f_cache)

    # ---- test queries
    test = pd.read_parquet(args.test)
    if args.limit:
        keep = test["molecule_id"].drop_duplicates().iloc[: args.limit]
        test = test[test["molecule_id"].isin(keep)]
    queries = []
    for mid, g in test.groupby("molecule_id", sort=False):
        mass = query_mass(g["precursor_mz"], g["adduct"])
        cands, err = pool.window(mass, args.ppm)
        queries.append({"molecule_id": mid, "cands": cands.astype(np.int64), "ppm_err": err,
                        "spectra": query_spectra(g, params), "mass": mass})
    n_cands = np.array([len(q["cands"]) for q in queries])
    common.log(f"[predict] {len(queries)} test molecules; candidates: median "
               f"{np.median(n_cands):.0f}, max {n_cands.max()}, none for "
               f"{int((n_cands == 0).sum())} molecules")

    # ---- features (analogue channel off -> an empty library, like stage2)
    lib = build_library(np.empty(0, dtype=np.int64), row_key, args.train, params,
                        {"tag_extra": {"stage2": "predict"},
                         "cache_hash": common.cache_hash(cache)}, workers, use_cache=False)
    Xs = featurize_queries(queries, pool, {}, lib, args.tol, args.frag_ppm, workers,
                           "[predict] features", args.mtf_tol)

    # ---- channel P (v2, guarded — no eval keys at serve time) and scoring
    fp_model, fp_info = fpchannel.build_fp_model(
        mask, row_key, args.train, fps, params, 2027, workers,
        eval_row_ids=np.empty(0, dtype=np.int64), eval_keys=set(), query_rows=None,
        artifact_dir=args.artifacts)
    common.log(f"[predict] channel P: {fp_info.get('kind')}"
               + (f" ({fp_info.get('encoder')} + {fp_info.get('head')})" if fp_model else ""))
    append_channel_blocks(queries, Xs, pool, fps_all, fp_model, {}, "[predict] fp features")

    rows, diag = [], []
    n_padded = 0
    for q, X in zip(queries, Xs):
        have, scores = [], []
        if len(q["cands"]):
            s = predict(model, X, [len(q["cands"])])
            order = rank_order(s, np.abs(q["ppm_err"]), q["cands"])[: args.top]
            for j in order:
                sm = pool.smiles[q["cands"][j]]
                if sm is not None and sm not in have:
                    have.append(sm)
                    scores.append(round(float(s[j]), 4))
        if len(have) < args.top:
            n_padded += 1
            have = pad_from_pool(q, pool, have, args.top)
        rows.append({"molecule_id": q["molecule_id"], "smiles": ";".join(have[: args.top])})
        diag.append({"molecule_id": q["molecule_id"], "query_mass": q["mass"],
                     "n_spectra": len(q["spectra"]), "n_candidates": len(q["cands"]),
                     "n_model_guesses": len(scores),
                     "p_top1": scores[0] if scores else np.nan,
                     "p_top5": (scores[4] if len(scores) > 4 else np.nan),
                     "fp_channel": fp_info.get("kind")})

    sub = pd.DataFrame(rows)
    assert sub["molecule_id"].is_unique and sub["smiles"].notna().all() \
        and (sub["smiles"] != "").all(), "bad submission rows"
    n_slots = sub["smiles"].str.count(";") + 1
    assert int(n_slots.max()) <= args.top, "too many guesses for a molecule"
    assert set(sub["molecule_id"]) == set(test["molecule_id"].unique()), "missing test molecules"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    if str(args.diag) and str(args.diag) != ".":
        args.diag.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diag).to_parquet(args.diag)
    common.log(f"[predict] wrote {args.out} ({len(sub)} molecules)")
    common.log(f"[predict] slots: min {n_slots.min()} / median {n_slots.median():.0f} / "
               f"max {n_slots.max()}; molecules padded from the pool: {n_padded}")
    common.log(f"[predict] total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
