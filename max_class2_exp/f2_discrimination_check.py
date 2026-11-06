"""Channel F2 discrimination sanity check (needs data/ + out/iceberg weights).

For N real measured queries (enveda-180, timsTOF), score the true structure
plus ~30 mass-window decoys by predicted-vs-observed matched cosine, and
report where the truth ranks. Random chance with 31 candidates: mean rank
~16, top-1 ~3 %. Run inside the ms-pred env:

    extern/ms-pred/.venv/bin/python max_class2_exp/f2_discrimination_check.py \
        [--n-queries 10] [--n-decoys 30]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nnspec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-queries", type=int, default=10)
    ap.add_argument("--n-decoys", type=int, default=30)
    ap.add_argument("--ppm", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    cols = ["normalized_smiles", "inchikey14", "adduct", "precursor_mz",
            "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities",
            "ingest_lib", "num_peaks"]
    df = pd.read_parquet(HERE.parent / "data" / "train.parquet", columns=cols)
    lib = df[(df["ingest_lib"] == "enveda-180") & (df["adduct"] == "[M+H]+")
             & (df["num_peaks"] >= 10)
             & df["collision_energy_ev"].map(
                 lambda x: hasattr(x, "__len__") and len(x) == 1)]  # single CE only
    queries = lib.groupby("inchikey14", sort=False).head(1).sample(
        args.n_queries, random_state=args.seed)

    # candidate structures: distinct molecules whose [M+H]+ precursor lies in
    # the +-ppm window (relaxed to +-0.05 Da if the window is too sparse)
    mol_rows = df[(df["adduct"] == "[M+H]+")].groupby("inchikey14", sort=False).first()

    ranks, topk = [], []
    t0 = time.time()
    for qi, (_, q) in enumerate(queries.iterrows()):
        window = mol_rows[np.abs(mol_rows["precursor_mz"] - q["precursor_mz"])
                           <= q["precursor_mz"] * args.ppm * 1e-6]
        if len(window) < args.n_decoys:
            window = mol_rows[np.abs(mol_rows["precursor_mz"] - q["precursor_mz"]) <= 0.05]
        window = window[window.index != q["inchikey14"]]
        if len(window) < args.n_decoys:
            continue
        decoys = window.sample(args.n_decoys, random_state=int(rng.integers(1 << 30)))
        entries = [(q["normalized_smiles"], "[M+H]+", float(np.atleast_1d(q["collision_energy_ev"])[0]))]
        entries += [(s, "[M+H]+", float(np.atleast_1d(q["collision_energy_ev"])[0]))
                    for s in decoys["normalized_smiles"]]
        preds = nnspec.predict_spectra(entries)
        sims = []
        for pred in preds:
            sims.append(nnspec.matched_cosine(
                q["ms2_mzs"], q["ms2_normalized_intensities"],
                pred[0], pred[1], args.ppm) if pred is not None else 0.0)
        order = np.argsort(sims)[::-1]
        rank = int(np.where(order == 0)[0][0]) + 1
        ranks.append(rank)
        topk.append((rank, sims[0], float(np.mean(sims[1:])),
                     q["inchikey14"][:6]))
        print(f"[{qi + 1}/{len(queries)}] {q['inchikey14'][:13]} "
              f"truth cosine {sims[0]:.3f} vs decoy mean {np.mean(sims[1:]):.3f} "
              f"-> rank {rank}/{len(sims)}")

    ranks = np.array(ranks)
    print(f"\n{len(ranks)} queries, {args.n_decoys + 1} candidates each "
          f"({time.time() - t0:.0f}s total)")
    print(f"mean rank {ranks.mean():.1f} (chance ~{(args.n_decoys + 2) / 2:.0f})  "
          f"top-1 {(ranks == 1).mean():.0%}  top-5 {(ranks <= 5).mean():.0%}  "
          f"top-10 {(ranks <= 10).mean():.0%}")
    print("truth-vs-decoy cosine margins (truth - decoy mean):")
    for rank, s_true, s_dec, ikey in topk:
        print(f"  {ikey}  margin {s_true - s_dec:+.3f}  rank {rank}")


if __name__ == "__main__":
    main()
