"""Channel F2 standalone evaluation: rank candidates by ICEBERG alone.

The class-2 protocol of stage2.py (same pool, same visibility mask, same
+-ppm shortlists, same eval spectra) but with exactly one scoring signal:
the matched cosine between each observed eval spectrum and ICEBERG's
predicted spectrum of the candidate structure (adduct/CE/instrument
conditioned). No fusion, no mass-error feature, no training — just the
model. The mass_error baseline (rank by |ppm|) is printed for reference.

Run inside the ms-pred env (from the repo root):

    extern/ms-pred/.venv/bin/python max_class2_exp/f2_eval.py \
        [--split dataset-creation/out/difficulty2.parquet] [--limit N]
        [--ppm 10] [--workers 22] [--cos-ppm 20]

Writes max_class2_exp/out/f2_eval/<name>.json with per-query rows.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "dataset-creation"))

import common  # noqa: E402
from dataloader import load_split, read_rows  # noqa: E402
from pool import build_pool  # noqa: E402

from enveda_casmi26.pipeline import neutral_mass  # noqa: E402


def ce_of(val) -> float:
    """Single collision energy in eV (mean for merged multi-CE rows)."""
    arr = np.atleast_1d(np.asarray(val, dtype=float))
    return float(np.nanmean(arr[arr == arr])) if (arr == arr).any() else float("nan")


def build_queries(split, truth, pool, ppm: float):
    """Per-molecule query records, mirroring stage2.build_eval_queries."""
    ev = split.eval_spectra()
    ev = ev[ev["molecule_id"].isin(truth["molecule_id"])]
    groups = dict(tuple(ev.groupby("molecule_id", sort=False)))
    queries, missing = [], 0
    for r in truth.itertuples():
        member = pool.member_of_key(r.key)
        g = groups.get(r.molecule_id)
        if member < 0 or g is None:
            missing += 1
            continue
        masses = [neutral_mass(float(p), a) for p, a in zip(g["precursor_mz"], g["adduct"])]
        masses = [m for m in masses if np.isfinite(m)]
        mass = float(np.median(masses)) if masses else float("nan")
        cands, err = pool.window(mass, ppm)
        spectra = []
        for (mz, it, add), ce in zip(zip(g["ms2_mzs"], g["ms2_normalized_intensities"], g["adduct"]),
                                     [ce_of(c) for c in g["collision_energy_ev"]]):
            mz, it = np.asarray(mz, float), np.asarray(it, float)
            o = np.argsort(mz)
            spectra.append((mz[o], it[o], add, ce))
        queries.append({"qid": r.molecule_id, "key": r.key, "member": int(member),
                        "cands": cands, "err": err, "mass": mass,
                        "smiles": pool.smiles[cands].tolist(), "spectra": spectra})
    return queries, missing


_WORKER = {}


def _init():
    import nnspec
    _WORKER["predict"] = nnspec.predict_spectra
    _WORKER["cosine"] = nnspec.matched_cosine
    _WORKER["model"] = nnspec._get_model()   # eager load per worker


def _one(q, cos_ppm):
    pred_fn, cos_fn = _WORKER["predict"], _WORKER["cosine"]
    cand_scores = np.zeros(len(q["cands"]))
    n_spec = len(q["spectra"])
    for mz, it, add, ce in q["spectra"]:
        entries = [(s, add, ce) for s in q["smiles"]]
        preds = pred_fn(entries)
        for j, p in enumerate(preds):
            cand_scores[j] += cos_fn(mz, it, p[0], p[1], cos_ppm) if p is not None else 0.0
    cand_scores /= max(n_spec, 1)
    hit = np.flatnonzero(np.asarray(q["cands"]) == q["member"])
    if len(hit) == 0:                     # truth outside its own window: a miss
        return {"qid": q["qid"], "key": q["key"], "n_cands": len(q["cands"]),
                "n_spec": n_spec, "rank": len(q["cands"]) + 1, "mass_rank": len(q["cands"]) + 1,
                "truth_cos": 0.0, "best_decoy_cos": float(cand_scores.max() or 0.0)}
    mem = int(hit[0])                     # position of the truth in the shortlist
    order = np.argsort(-cand_scores)
    rank = int(np.where(order == mem)[0][0]) + 1
    mass_order = np.argsort(np.abs(q["err"]))
    mass_rank = int(np.where(mass_order == mem)[0][0]) + 1
    tc = float(cand_scores[mem])
    decoy = np.delete(cand_scores, mem) if len(cand_scores) > 1 else np.array([0.0])
    return {"qid": q["qid"], "key": q["key"], "n_cands": len(q["cands"]),
            "n_spec": n_spec, "rank": rank, "mass_rank": mass_rank,
            "truth_cos": tc, "best_decoy_cos": float(decoy.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dataset-creation/out/difficulty2.parquet")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N molecules")
    ap.add_argument("--ppm", type=float, default=10.0)
    ap.add_argument("--cos-ppm", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=min(22, __import__("os").cpu_count()))
    args = ap.parse_args()

    split = load_split(args.split)
    molecules = pd.read_parquet(common.CACHE / "molecules.parquet")
    row_key = np.load(common.CACHE / "row_key.npy")
    pool, _ = build_pool(molecules, row_key, split.train_mask,
                         split.library_additions(), split.excluded_library_keys)
    truth = split.truth
    if args.limit:
        truth = truth.head(args.limit)

    # which source library each truth molecule's eval rows come from
    ev_rows = np.concatenate([np.asarray(r, dtype=np.int64) for r in truth["eval_row_ids"]]) \
        if len(truth) else np.empty(0, dtype=np.int64)
    libs = read_rows(ev_rows, ["ingest_lib"]).to_pandas()["ingest_lib"].tolist()
    row_mol = {int(r): mid for mid, rows in zip(truth["molecule_id"], truth["eval_row_ids"])
               for r in rows}
    q_lib = pd.Series(libs, index=[row_mol[int(r)] for r in ev_rows]).groupby(level=0).first()

    print(f"pool: {len(pool)} structures; queries: {len(truth)} molecules", flush=True)
    queries, missing = build_queries(split, truth, pool, args.ppm)
    print(f"built {len(queries)} queries ({missing} without truth/spectra); "
          f"median shortlist {int(np.median([len(q['cands']) for q in queries]))}", flush=True)

    queries.sort(key=lambda q: q["mass"])   # cache locality: neighbours share candidates
    t0 = time.time()
    import multiprocess as mp
    with mp.Pool(args.workers, initializer=_init) as pool_mp:
        rows = list(pool_mp.starmap(_one, [(q, args.cos_ppm) for q in queries], chunksize=1))

    df = pd.DataFrame(rows).set_index("qid")
    df["ingest_lib"] = q_lib.reindex(df.index).fillna("?")
    df["max_visible_tc"] = truth.set_index("molecule_id")["max_visible_tc"].reindex(df.index)
    for meth, col in [("f2", "rank"), ("mass_error", "mass_rank")]:
        df[f"rr_{meth}"] = np.where(df[col] <= 25, 1.0 / df[col], 0.0)

    def block(d, label):
        if not len(d):
            return
        print(f"{label:<32} n={len(d):<5} MRR@25={d['rr_f2'].mean():.3f} "
              f"top1={(d['rank'] == 1).mean():.2f} top10={(d['rank'] <= 10).mean():.2f} "
              f"(mass_err MRR@25={d['rr_mass_error'].mean():.3f})", flush=True)

    print(f"\n=== F2-only, {df.shape[0]} queries, {time.time() - t0:.0f}s ===", flush=True)
    block(df, "overall")
    for lo, hi in [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]:
        block(df[(df["max_visible_tc"] >= lo) & (df["max_visible_tc"] < hi)],
              f"tc band {lo:.2f}-{hi:.2f}")
    for lib, d in df.groupby("ingest_lib"):
        block(d, f"lib {lib}")

    out = HERE / "out" / "f2_eval"
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.split).stem + (f"_lim{args.limit}" if args.limit else "")
    df.reset_index().to_parquet(out / f"{name}.parquet")
    (out / f"{name}.json").write_text(json.dumps(
        {"split": args.split, "mrr_f2": float(df["rr_f2"].mean()),
         "mrr_mass": float(df["rr_mass_error"].mean()),
         "top1_f2": float((df["rank"] == 1).mean()),
         "top10_f2": float((df["rank"] <= 10).mean()),
         "median_rank": float(df["rank"].median()), "n": int(len(df))}, indent=2))
    print(f"saved {out / name}.parquet", flush=True)


if __name__ == "__main__":
    main()
