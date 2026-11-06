"""Combine difficulty splits into one eval set (default: difficulty1 + difficulty2).

A combined split is the union of the source truth rows — no re-simulation.
Everything downstream is per-row and recomputed at load time, so the union
is safe by construction:

- compute_train_mask hides every hidden/eval row and every removed
  neighbour of BOTH sources;
- the channel-P v2 artifact guards cover the union automatically
  (eval_split_keys reads every dataset-creation/out/*.parquet, and the
  union's keys are exactly the sources' keys — the encoder was trained
  with both sources excluded);
- load_split verifies the combined cache_hash against this machine's cache,
  exactly as for any other split.

All sources must have been built against the same molecule-key cache (same
cache_hash) — the split would be meaningless otherwise, so this script
refuses to mix.

Usage:
    uv run python dataset-creation/make_difficulty12.py
    uv run python dataset-creation/make_difficulty12.py \
        --source dataset-creation/out/difficulty1.parquet \
        --source dataset-creation/out/difficulty2.parquet \
        --out dataset-creation/out/difficulty12.parquet

Then:
    uv run python dataset-creation/check_split.py \
        dataset-creation/out/difficulty12.parquet --no-rerun
    uv run python max_class2_exp/stage2.py \
        --split dataset-creation/out/difficulty12.parquet --blind-control --model-seeds 3
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import cache_hash, log, read_split_info, write_output  # noqa: E402

DEFAULT_SOURCES = ("difficulty1.parquet", "difficulty2.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, action="append", default=None,
                        help=f"source split parquet (repeatable; default: "
                             f"{' + '.join(DEFAULT_SOURCES)} in dataset-creation/out/)")
    parser.add_argument("--out", type=Path, default=common.OUT / "difficulty12.parquet",
                        help="combined split path")
    parser.add_argument("--dedupe", action="store_true",
                        help="if a molecule appears in several sources, keep its FIRST row "
                             "instead of refusing")
    args = parser.parse_args()

    names = args.source or [common.OUT / n for n in DEFAULT_SOURCES]
    frames, infos = [], []
    for p in names:
        if not p.exists():
            raise SystemExit(f"source split not found: {p}")
        frames.append(pd.read_parquet(p))
        infos.append(read_split_info(p))
        log(f"[in] {p.name}: {len(frames[-1])} molecules, "
            f"cache {str(infos[-1].get('cache_hash', '?'))[:12]}")
    if len(frames) < 2:
        raise SystemExit("need at least two --source splits to combine")

    # all sources against the same molecule-key cache ...
    hashes = sorted({str(i.get("cache_hash")) for i in infos})
    if len(hashes) != 1:
        raise SystemExit("the sources were built against different caches "
                         f"({', '.join(h[:12] for h in hashes)}) — do not combine")
    # ... and against THIS machine's cache
    here = cache_hash(common.CACHE)
    if hashes[0] != here:
        raise SystemExit(f"this machine's cache is {here[:12]}, but the sources were "
                         f"built against {hashes[0][:12]} — re-sync the cache first")
    stamps = {(i.get("train_size"), i.get("train_mtime")) for i in infos}
    if len(stamps) != 1:                     # paranoia: same train file, too
        raise SystemExit("the sources were built against different train files")

    combined = pd.concat(frames, ignore_index=True)
    dup = combined["key"].duplicated(keep="first").to_numpy()
    if dup.any():
        keys = combined.loc[dup, "key"].tolist()
        if not args.dedupe:
            raise SystemExit(f"{len(keys)} molecule(s) appear in more than one source "
                             f"(e.g. {keys[:3]}); pass --dedupe to keep the first "
                             "occurrence of each")
        log(f"[combine] dropping {len(keys)} duplicate molecule(s): {sorted(set(keys))[:5]} ...")
        combined = combined[~dup].reset_index(drop=True)
    id_dup = combined["molecule_id"].duplicated(keep="first").to_numpy()
    if id_dup.any():
        # molecule_id embeds the difficulty (d1_/d2_...), so this can only happen
        # when combining two splits of the same difficulty level; molecule_id is
        # a label (not a row pointer), so re-suffix to keep ids unique.
        combined.loc[id_dup, "molecule_id"] = [
            f"{mid}_c{i}" for i, mid in enumerate(combined.loc[id_dup, "molecule_id"])]
        log(f"[combine] re-suffixed {int(id_dup.sum())} colliding molecule_id(s)")

    info = {"script": "make_difficulty12.py",
            "sources": [{"name": p.name, "n_molecules": len(f)}
                        for p, f in zip(names, frames)],
            "params": {p.name: i.get("params") for p, i in zip(names, infos)},
            "seed": {p.name: i.get("seed") for p, i in zip(names, infos)},
            "cache_hash": hashes[0],
            "train_size": infos[0].get("train_size"),
            "train_mtime": infos[0].get("train_mtime")}
    write_output(combined, args.out, info)

    log(f"[combine] {args.out.name}: {len(frames[0])} + {len(frames[1])}"
        + (f" (+{len(frames) - 2} more)" if len(frames) > 2 else "")
        + f" -> {len(combined)} molecules")
    log("[combine] next: uv run python dataset-creation/check_split.py "
        f"{args.out} --no-rerun ; then max_class2_exp/stage2.py --split {args.out}")


if __name__ == "__main__":
    main()
