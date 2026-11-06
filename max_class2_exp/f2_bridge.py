"""Channel F2 bridge: stage2 (repo venv) <-> ICEBERG (ms-pred venv).

ICEBERG (`nnspec.py`) runs in its own CPU venv (`extern/ms-pred/.venv` —
different rdkit/torch pins), which stage2 must not import. The bridge:

  1. `collect_requests()` — the unique (smiles, adduct, ce-bucket) triples a
     run needs: every candidate of every shortlist (eval, training, blind)
     for every (adduct, collision energy) its spectra actually use;
  2. `ensure_predictions()` — checks the persistent cache
     (`out/cache/f2_predictions.parquet`), ships the missing triples to
     `f2_batch.py` through the ms-pred interpreter (ONE subprocess, batched,
     one frozen model per worker), then loads the cache back;
  3. the feature block (features.channel_block) compares each observed
     spectrum with the candidate's predicted spectrum (matched cosine, ppm
     tolerance — `spectra_utils.matched_cosine_one_vs_many_ppm`).

Zero-shot leakage: the public MassSpecGym checkpoint never saw any Enveda
spectrum, so neither eval truths nor training pseudo-truths can be
memorised (README §F2). A self-trained ICEBERG upgrade must be trained
molecule-disjoint from every split before its predictions may be cached
here — the same guard philosophy as channel P v2.

When the ms-pred env or the weights are absent the channel degrades: all
`f2_*` features stay 0 and `f2_available` is 0 everywhere — the fusion
keeps working on the remaining channels. The cache makes the expensive
first prediction pass a one-time cost per candidate pool; `--no-cache`
(library cache) does not touch it — delete the parquet to force
re-prediction.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "dataset-creation"))

import common  # noqa: E402
from fpchannel import ce_bucket  # noqa: E402

MS_PRED_PY = PROJECT_ROOT / "extern" / "ms-pred" / ".venv" / "bin" / "python"
BATCH_SCRIPT = HERE / "f2_batch.py"
WEIGHTS_DIR = HERE / "out" / "iceberg"
CACHE_PATH = HERE / "out" / "cache" / "f2_predictions.parquet"
MANIFEST_PATH = HERE / "out" / "cache" / "f2_manifest.parquet"


def weights_present() -> bool:
    return MS_PRED_PY.exists() and any(WEIGHTS_DIR.rglob("*.ckpt"))


def available() -> bool:
    return weights_present()


def collect_requests(queries, pool) -> set:
    """Unique (smiles, adduct, ce_bucket) prediction keys a query set needs."""
    reqs = set()
    for q in queries:
        smiles = [str(pool.smiles[c]) for c in q["cands"]]
        for spec in q["spectra"]:
            adduct = str(spec[4])
            ce = ce_bucket(spec[5]) if len(spec) > 5 else ce_bucket(None)
            for s in smiles:
                reqs.add((s, adduct, ce))
    return reqs


def _read_cache() -> pd.DataFrame:
    return pd.read_parquet(CACHE_PATH) if CACHE_PATH.exists() else pd.DataFrame(
        columns=["smiles", "adduct", "ce", "mz", "inten", "ok"])


def _usable(df: pd.DataFrame) -> dict:
    """{(smiles, adduct, ce): (mz, p, ||p||^2)} — usable predictions only."""
    out = {}
    for row in df[df["ok"]].itertuples(index=False):
        p = np.asarray(row.inten, dtype=float)
        out[(str(row.smiles), str(row.adduct), float(row.ce))] = (
            np.asarray(row.mz, dtype=float), p, float((p ** 2).sum()))
    return out


def _all_keys(df: pd.DataFrame) -> set:
    """Every cached key, INCLUDING ok=False (unparseable SMILES / unsupported
    adduct): they must never be re-requested, just never scored."""
    return set(zip(df["smiles"].astype(str), df["adduct"].astype(str),
                   df["ce"].astype(float)))


def ensure_predictions(requests: set, workers: int, log=common.log):
    """Make sure every requested key is in the cache; return (preds, info).

    preds maps (smiles, adduct, ce_bucket) -> (mz, p, ||p||^2); keys that
    ICEBERG cannot produce (unparseable SMILES, unsupported adduct) are
    cached as ok=False and simply never appear in the dict."""
    if not requests:
        return {}, {"available": available(), "n_requests": 0}
    if not weights_present():
        log("[f2] ICEBERG env/weights not found -> channel off (f2_* features stay 0)")
        return {}, {"available": False, "n_requests": len(requests)}
    cached = _read_cache()
    have = _all_keys(cached)
    missing = sorted(requests - have)
    info = {"available": True, "n_requests": len(requests),
            "n_cached": len(requests & have), "n_missing": len(missing),
            "cache": str(CACHE_PATH)}
    if missing:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(missing, columns=["smiles", "adduct", "ce"]).to_parquet(MANIFEST_PATH)
        log(f"[f2] predicting {len(missing):,} candidate spectra with ICEBERG "
            f"({len(requests) - len(missing):,} already cached; ~0.2-0.4 s per "
            f"structure per worker, so the FIRST run over a fresh pool takes "
            f"a while — later runs reuse the cache)")
        t0 = time.time()
        r = subprocess.run([str(MS_PRED_PY), str(BATCH_SCRIPT),
                            "--manifest", str(MANIFEST_PATH), "--cache", str(CACHE_PATH),
                            "--workers", str(max(1, workers))],
                           cwd=str(PROJECT_ROOT))
        info["predict_seconds"] = round(time.time() - t0, 1)
        info["subprocess_returncode"] = int(r.returncode)
        if r.returncode != 0:
            log(f"[f2] WARNING: f2_batch.py exited with {r.returncode}; the "
                f"missing predictions stay 0 (f2_available marks them)")
    preds = _usable(_read_cache())
    info["n_ready"] = int(sum(1 for k in requests if k in preds))
    return preds, info
