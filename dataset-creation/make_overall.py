"""Overall eval set: a mix of difficulty 1, 2 and 3 (default 20/60/20).

Difficulty 2 and 3 are drawn first, because they decide which neighbours
are hidden; difficulty 1 is drawn last and never from a hidden neighbour, so
its visible rows survive. The three molecule sets are disjoint and
--diversity-tc applies across all of them.

Usage:
    uv run python dataset-creation/make_overall.py [--n-molecules 1000]
        [--share 0.2,0.6,0.2] [--neighbour-removed-frac 0.10] [--neighbour-tc 0.5]
        [--diversity-tc 0.6] [--seed 2026]
        [--out dataset-creation/out/overall.parquet]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log  # noqa: E402
from make_difficulty1 import (CHANGES, DEFAULT_CHANGE_MIX, DEFAULT_MIXED_DIMS,  # noqa: E402
                              parse_mixed_dims, select_difficulty1)
from make_difficulty2 import add_removal_args, check_frac, select_difficulty2  # noqa: E402


def add_args(parser: argparse.ArgumentParser) -> None:
    common.add_common_args(parser)
    parser.add_argument("--share", default="0.2,0.6,0.2", help="difficulty 1,2,3 shares")
    parser.add_argument("--change-mix", default=DEFAULT_CHANGE_MIX,
                        help="difficulty 1: share of molecules per change type (instrument, adduct, ce, mixed)")
    parser.add_argument("--mixed-dims", default=DEFAULT_MIXED_DIMS,
                        help="difficulty 1 mixed: weight per combo of held-out dims")
    add_removal_args(parser)
    parser.add_argument("--out", type=Path, default=common.OUT / "overall.parquet")


def build(args: argparse.Namespace, prepared=None) -> tuple[pd.DataFrame, dict]:
    check_frac(args.neighbour_removed_frac)
    meta, molecules, row_key = prepared or common.prepare(args)
    shares = [float(x) for x in args.share.split(",")]
    n1, n2, n3 = common.split_counts(args.n_molecules, shares)
    log(f"[overall] targets d1 {n1}, d2 {n2}, d3 {n3}")
    rng = np.random.default_rng(args.seed)
    rows_index = common.mol_rows(meta)
    sim, diversity, base = common.selection_context(args, molecules)
    used_keys, used_ids = set(), set()

    records, shortfalls = [], {}
    for n, mock in ((n2, False), (n3, True)):
        rec, sf = select_difficulty2(meta, molecules, n, rng, used_keys, used_ids, rows_index, sim,
                                     diversity, base, neighbour_removed_frac=args.neighbour_removed_frac,
                                     neighbour_tc=args.neighbour_tc,
                                     max_removed_neighbours=args.max_removed_neighbours,
                                     mock_library=mock, ion_pos_share=args.ion_pos_share)
        records += rec
        shortfalls.update(sf)
    rec, sf = select_difficulty1(meta, molecules, n1, rng, used_keys, used_ids, rows_index,
                                 common.parse_mix(args.change_mix, CHANGES), diversity, base,
                                 ion_pos_share=args.ion_pos_share, mixed_dims=parse_mixed_dims(args))
    records = rec + records
    shortfalls.update(sf)
    df = common.annotate_similarity(common.records_to_frame(records), molecules, row_key, sim)
    log(f"[overall] realised shares {df['difficulty'].value_counts(normalize=True).sort_index().round(3).to_dict()}")
    return df, shortfalls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(parser)
    args = parser.parse_args()
    meta, molecules, row_key = common.prepare(args)
    df, shortfalls = build(args, (meta, molecules, row_key))
    common.finish(df, args, "make_overall.py", meta, molecules, row_key, shortfalls)


if __name__ == "__main__":
    main()
