"""Run the stage-2 re-ranker on the competition test set and build submission.csv.

Per test molecule (predictions are made per `molecule_id`, not per spectrum):
  1. query mass = median neutral mass over the molecule's spectra
     (`enveda_casmi26.pipeline.neutral_mass`), candidates = every train
     molecule within ±--ppm of it (all train rows are visible for the real
     test — no eval split, no hidden rows);
  2. the candidates' library spectra are featurized with the same `featurize`
     as training (R4) after test-like query cleaning (`dataloader.clean_peaks`
     then `clean_spectrum`, exactly like the difficulty-1 eval spectra);
  3. the saved stage-2 model (`<model>`, trained by stage2.py on the
     difficulty-1 simulator) gives P(candidate is the truth); candidates are
     ranked by it (ties: smaller mass error, then key);
  4. the top --top candidate SMILES go into the submission, best first.

Every molecule must appear exactly once with at least one guess, so shortlists
shorter than --top (and molecules with no candidates at all) are padded with
the train structures whose exact mass is closest to the query mass (a wrong
guess costs only its rank; a missing row would be rejected). Molecules without
a usable query mass get the globally most-spectra train structures. This is a
class-1 pipeline: molecules whose structure is not in train (classes 2/3) can
only score through luck in the padding.

Works locally and inside the Kaggle kernel (same code path):
    uv run python max_class1_exp/predict_test.py --test data/test.parquet
    uv run python max_class1_exp/predict_test.py --test /kaggle/input/<comp>/test.parquet

Outputs:
    --out        submission.csv (molecule_id, smiles; up to --top ';'-joined guesses)
    --diag       per-molecule diagnostics parquet (n_candidates, top scores, ...)
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
from library import CandidateIndex, build_library, query_mass  # noqa: E402  (also fixes sys.path)
import common  # noqa: E402  (dataset-creation)
import stage2  # noqa: E402
from spectra import CleanParams  # noqa: E402
from stage1 import rank_order  # noqa: E402


def build_queries(test: pd.DataFrame, cidx: CandidateIndex, ppm: float, params: CleanParams):
    """One query dict per test molecule: cleaned spectra, candidate window."""
    queries = []
    for mid, g in test.groupby("molecule_id", sort=False):
        mass = query_mass(g["precursor_mz"], g["adduct"])
        cands, err = cidx.window(mass, ppm)
        spectra = stage2.load_query_spectra(g, params)
        queries.append({"molecule_id": mid, "cands": cands.astype(np.int64), "ppm_err": err,
                        "spectra": spectra, "mass": mass})
    return queries


def rank_molecules(molecules: pd.DataFrame, by: str) -> np.ndarray:
    """Molecule indices usable as submission guesses, ordered by `by`."""
    ok = molecules["smiles"].notna().to_numpy() & np.isfinite(molecules["exact_mass"].to_numpy())
    return np.flatnonzero(ok)


def pad_smiles(q, molecules: pd.DataFrame, pad_by_mass: np.ndarray, fallback: np.ndarray,
               have: list, top: int) -> list:
    """Fill `have` up to `top` guesses: mass-nearest structures, deduped."""
    if len(have) >= top:
        return have
    used = set(have)
    smiles = molecules["smiles"].to_numpy()
    if np.isfinite(q["mass"]):
        mass = molecules["exact_mass"].to_numpy(dtype=float)[pad_by_mass]
        pos = int(np.searchsorted(mass, q["mass"]))
        order = np.argsort(np.abs(mass - q["mass"]), kind="stable")
        pool = pad_by_mass[order]
    else:
        pool = fallback
    for m in pool:
        if len(have) >= top:
            break
        s = smiles[m]
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
                        help="train parquet (default: data/train_clean.parquet when present, "
                             "else data/train.parquet)")
    parser.add_argument("--model", type=Path,
                        default=HERE / "out" / "difficulty1_stage2_model.joblib",
                        help="stage-2 model saved by stage2.py")
    parser.add_argument("--ppm", type=float, default=10.0, help="candidate mass window (± ppm)")
    parser.add_argument("--tol", type=float, default=0.02, help="m/z tolerance for peaks (Da)")
    parser.add_argument("--noise", type=float, default=0.01,
                        help="drop peaks below this share of the base peak")
    parser.add_argument("--top", type=int, default=25, help="guesses per molecule (competition: 25)")
    parser.add_argument("--limit", type=int, default=0, help="only the first N test molecules")
    parser.add_argument("--out", type=Path, default=common.PROJECT_ROOT / "submissions"
                        / "stage2_submission.csv", help="output submission csv")
    parser.add_argument("--diag", type=Path,
                        default=HERE / "out" / "test_stage2_diag.parquet",
                        help="per-molecule diagnostics parquet ('' to skip)")
    parser.add_argument("--workers", type=int, default=None, help="parallel processes")
    parser.add_argument("--cache-dir", type=Path, default=common.CACHE, help="dataset cache directory")
    parser.add_argument("--no-cache", action="store_true", help="rebuild the library cache")
    args = parser.parse_args()
    if args.test is None:
        args.test = common.prefer_cleaned(common.TRAIN.parent / "test.parquet")
    if args.train is None:
        args.train = common.prefer_cleaned(common.TRAIN)
    common.log(f"[predict] train data: {args.train} | test data: {args.test}")
    t_start = time.time()
    params = CleanParams(tol=args.tol, noise=args.noise)
    workers = args.workers or max(1, (os.cpu_count() or 2) - 2)

    # ---- model first: fail fast on a stale feature set
    import joblib
    blob = joblib.load(args.model)
    assert blob.get("features") == stage2.FEATURES, \
        f"{args.model.name} was trained with a different feature set; retrain first"
    model = blob["model"]
    common.log(f"[predict] {blob['kind']} model from {args.model.name} "
               f"(trained with seed {blob.get('train_seed')})")

    # ---- molecules / rows (no eval split: every train row is visible)
    molecules, row_key = common.ensure_cache(args.train, args.cache_dir)
    mask = np.ones(len(row_key), dtype=bool)
    cidx = CandidateIndex(molecules, row_key, mask)
    common.log(f"[predict] {len(molecules):,} train molecules, all rows visible")

    test = pd.read_parquet(args.test)
    if args.limit:
        keep = test["molecule_id"].drop_duplicates().iloc[: args.limit]
        test = test[test["molecule_id"].isin(keep)]
    queries = build_queries(test, cidx, args.ppm, params)
    n_cands = np.array([len(q["cands"]) for q in queries])
    common.log(f"[predict] {len(queries)} test molecules; candidates: median "
               f"{np.median(n_cands):.0f}, max {n_cands.max()}, none for "
               f"{int((n_cands == 0).sum())} molecules")

    # ---- one library for the union of all candidate rows
    need = np.array(sorted({int(m) for q in queries for m in q["cands"]}), dtype=np.int64)
    lib_rows = np.flatnonzero(np.isin(row_key, need)) if len(need) else np.empty(0, dtype=np.int64)
    lib = build_library(lib_rows, row_key, args.train, params,
                        {"tag_extra": {"stage": "test"}, "cache_hash": common.cache_hash(args.cache_dir)},
                        workers, use_cache=not args.no_cache)

    # ---- features + model scores
    Xs = stage2.featurize_queries(queries, lib, params, workers, "[predict] features")
    smiles = molecules["smiles"].to_numpy()

    # padding pools: mass-sorted usable molecules; fallback = most train spectra
    pad_by_mass = rank_molecules(molecules, "mass")
    fallback = pad_by_mass[np.argsort(-molecules["n_rows"].to_numpy()[pad_by_mass], kind="stable")]

    rows, diag = [], []
    n_padded = 0
    for q, X in zip(queries, Xs):
        have, scores = [], []
        if len(q["cands"]):
            p = stage2.predict(model, X, [len(q["cands"])])
            order = rank_order(p, np.abs(q["ppm_err"]), q["cands"])[: args.top]
            for j in order:
                s = smiles[q["cands"][j]]
                if s is not None and s not in have:
                    have.append(s)
                    scores.append(round(float(p[j]), 4))
        if len(have) < args.top:
            n_padded += 1
            have = pad_smiles(q, molecules, pad_by_mass, fallback, have, args.top)
        rows.append({"molecule_id": q["molecule_id"], "smiles": ";".join(have[: args.top])})
        diag.append({"molecule_id": q["molecule_id"], "query_mass": q["mass"],
                     "n_spectra": len(q["spectra"]), "n_candidates": len(q["cands"]),
                     "n_model_guesses": len(scores),
                     "p_top1": scores[0] if scores else np.nan,
                     "p_top5": (scores[4] if len(scores) > 4 else np.nan)})

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
               f"max {n_slots.max()}; molecules padded from the mass pool: {n_padded}")
    common.log(f"[predict] total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
