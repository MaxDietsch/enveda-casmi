"""Batch ICEBERG predictions for the stage2 fusion (f2_bridge's subprocess).

Runs inside the ms-pred venv (never the repo venv):

    extern/ms-pred/.venv/bin/python max_class2_exp/f2_batch.py \
        --manifest max_class2_exp/out/cache/f2_manifest.parquet \
        --cache max_class2_exp/out/cache/f2_predictions.parquet [--workers N]

Reads the manifest (smiles, adduct, ce), skips keys already in the cache,
predicts the rest with nnspec.predict_spectra in a multiprocess pool (one
frozen model per worker, ~0.2-0.4 s per structure single-core), and appends
every result to the cache parquet — including ok=False rows for SMILES
ICEBERG cannot parse, so they are never retried. The cache is written once
at the end; a killed run leaves it untouched.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import nnspec  # noqa: E402  (adds the ms-pred src path itself)

_WORKER = {}


def _init():
    _WORKER["predict"] = nnspec.predict_spectra
    nnspec._get_model()                     # eager load per worker (like f2_eval)


def _predict_chunk(entries):
    return _WORKER["predict"](list(entries))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True, help="parquet: smiles, adduct, ce")
    ap.add_argument("--cache", required=True, help="prediction cache parquet (appended)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--chunk", type=int, default=256,
                    help="entries per pool task (one ICEBERG batch per task)")
    args = ap.parse_args()

    man = pd.read_parquet(args.manifest)
    have = set()
    if Path(args.cache).exists():
        c = pd.read_parquet(args.cache)
        have = set(zip(c["smiles"].astype(str), c["adduct"].astype(str), c["ce"].astype(float)))
    todo = list(dict.fromkeys(
        (str(s), str(a), float(ce)) for s, a, ce in zip(man["smiles"], man["adduct"], man["ce"])
        if (str(s), str(a), float(ce)) not in have))
    if not todo:
        print("f2_batch: nothing to predict (all cached)")
        return
    print(f"f2_batch: predicting {len(todo):,} spectra "
          f"({len(have):,} cached) with {args.workers} workers ...", flush=True)
    chunks = [todo[i: i + args.chunk] for i in range(0, len(todo), args.chunk)]
    t0 = time.time()
    rows = []
    import multiprocess as mp
    with mp.Pool(args.workers, initializer=_init) as pool:
        for ci, res in enumerate(pool.imap(_predict_chunk, chunks)):
            for (s, a, ce), r in zip(chunks[ci], res):
                ok = r is not None
                rows.append({"smiles": s, "adduct": a, "ce": ce,
                             "mz": r[0].tolist() if ok else [],
                             "inten": r[1].tolist() if ok else [],
                             "ok": ok})
            print(f"  {min((ci + 1) * args.chunk, len(todo)):,}/{len(todo):,} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    df_new = pd.DataFrame(rows)
    cache = Path(args.cache)
    old = pd.read_parquet(cache) if cache.exists() else None
    out = pd.concat([old, df_new], ignore_index=True) if old is not None else df_new
    out = out.drop_duplicates(subset=["smiles", "adduct", "ce"], keep="first")
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    print(f"f2_batch: cache now {len(out):,} rows ({int(out['ok'].sum()):,} usable) "
          f"-> {cache}", flush=True)


if __name__ == "__main__":
    main()
