"""Difficulty 2 (and 3): eval molecules with no spectra left in train.

Every eval molecule has all its train rows hidden. For a fraction of them
(--neighbour-removed-frac) every molecule with Morgan Tanimoto >=
--neighbour-tc to it is hidden as well, so no similar molecule is left in
train. --mock-library turns the output into difficulty 3: the molecules are
also flagged exclude_from_library, which the dataloader exposes so the
candidate pool can drop them.

Molecules are drawn by weighted sampling without replacement. No two eval
molecules may have Tanimoto >= --diversity-tc to each other, and each
molecule's base weight is 1 / its number of neighbours at that threshold, so
large families of similar molecules are not over-represented. Mass-bin and
ion-mode weights apply on top.

Usage:
    uv run python dataset-creation/make_difficulty2.py [--n-molecules 1000]
        [--diversity-tc 0.6] [--neighbour-removed-frac 0.10] [--neighbour-tc 0.5]
        [--max-removed-neighbours 200] [--mock-library] [--seed 2026]
        [--out dataset-creation/out/difficulty2.parquet]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log  # noqa: E402


def select_difficulty2(meta: pd.DataFrame, molecules: pd.DataFrame, n: int, rng: np.random.Generator,
                       exclude_keys: set, used_ids: set, rows_index, sim: "common.Similarity",
                       diversity: "common.Diversity", base_weights: np.ndarray | None = None,
                       neighbour_removed_frac: float = 0.10, neighbour_tc: float = 0.5,
                       max_removed_neighbours: int = 200, mock_library: bool = False,
                       ion_pos_share: float = 0.7) -> tuple[list[dict], dict]:
    """Pick n unseen molecules. Returns (output records, shortfalls).

    exclude_keys (molecule indices) and diversity are updated in place;
    hidden neighbours are added to exclude_keys too, so later levels cannot
    use them.
    """
    difficulty = 3 if mock_library else 2
    order, start = rows_index
    eligible = meta["eligible"].to_numpy()
    cand = common.eval_candidates(molecules)
    cand = cand[~np.isin(cand, list(exclude_keys))]
    picked = common.pick_diverse(rng, molecules, cand, n, diversity, ion_pos_share, base_weights)
    exclude_keys.update(picked)

    # Neighbourhood removal: only molecules with 1..max_removed_neighbours
    # neighbours qualify (with none there is nothing to remove; with very many
    # one choice would wipe out a large part of train).
    n_remove = int(round(neighbour_removed_frac * len(picked)))
    removed = {}
    if n_remove:
        order_pick = [picked[i] for i in rng.permutation(len(picked))]
        for m, nb in zip(order_pick, sim.neighbours(order_pick, neighbour_tc)):
            nb = nb[nb != m]
            if 1 <= len(nb) <= max_removed_neighbours:
                removed[m] = nb
                if len(removed) == n_remove:
                    break
    keys = molecules["key"].to_numpy()

    records = []
    for m in picked:
        rows = order[start[m]:start[m + 1]]
        elig_rows = np.sort(rows[eligible[rows]])
        k = common.draw_n_spectra(rng, len(elig_rows))
        ev = rng.choice(elig_rows, size=k, replace=False)
        nb = removed.get(m, np.empty(0, np.int64))
        records.append(common.make_record(
            molecules, m, common.new_molecule_id(rng, difficulty, used_ids), difficulty, None,
            ev, rows, removed_keys=keys[nb].tolist(), exclude_from_library=mock_library))

    shortfalls = {}
    if len(picked) < n:
        shortfalls[f"difficulty{difficulty}_molecules"] = n - len(picked)
    if len(removed) < n_remove:
        shortfalls[f"difficulty{difficulty}_neighbours_removed"] = n_remove - len(removed)
    if removed:
        hidden = np.unique(np.concatenate(list(removed.values())))
        exclude_keys.update(int(x) for x in hidden)
        n_rows = int(np.isin(meta["mol"].to_numpy(), hidden).sum())
        sizes = np.array([len(v) for v in removed.values()])
        log(f"[d{difficulty}] neighbourhood removal (Tc >= {neighbour_tc}): {len(removed)} molecules, "
            f"neighbours per molecule median {np.median(sizes):.0f} max {sizes.max()}, "
            f"{len(hidden):,} molecules / {n_rows:,} extra rows hidden")
    log(f"[d{difficulty}] picked {len(picked)}/{n} molecules, "
        f"{sum(r['n_eval_spectra'] for r in records)} eval spectra")
    return records, shortfalls


def add_removal_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--neighbour-removed-frac", type=float, default=0.10,
                        help="fraction of difficulty 2/3 molecules whose similar molecules are hidden too")
    parser.add_argument("--neighbour-tc", type=float, default=0.5,
                        help="Morgan r2/2048 Tanimoto at or above which a molecule counts as a neighbour")
    parser.add_argument("--max-removed-neighbours", type=int, default=200,
                        help="molecules with more neighbours than this are not used for removal")


def add_args(parser: argparse.ArgumentParser) -> None:
    common.add_common_args(parser)
    add_removal_args(parser)
    parser.add_argument("--mock-library", action="store_true",
                        help="difficulty 3: also exclude the structures from the candidate library")
    parser.add_argument("--out", type=Path, default=None)


def check_frac(frac: float) -> None:
    if not 0.05 <= frac <= 0.20:
        raise SystemExit(f"--neighbour-removed-frac must lie in [0.05, 0.20], got {frac}")


def build(args: argparse.Namespace, prepared=None) -> tuple[pd.DataFrame, dict]:
    check_frac(args.neighbour_removed_frac)
    meta, molecules, row_key = prepared or common.prepare(args)
    sim, diversity, base = common.selection_context(args, molecules)
    rng = np.random.default_rng(args.seed)
    records, shortfalls = select_difficulty2(
        meta, molecules, args.n_molecules, rng, set(), set(), common.mol_rows(meta), sim, diversity,
        base, neighbour_removed_frac=args.neighbour_removed_frac, neighbour_tc=args.neighbour_tc,
        max_removed_neighbours=args.max_removed_neighbours, mock_library=args.mock_library,
        ion_pos_share=args.ion_pos_share)
    df = common.records_to_frame(records)
    return common.annotate_similarity(df, molecules, row_key, sim), shortfalls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(parser)
    args = parser.parse_args()
    if args.out is None:
        args.out = common.OUT / ("difficulty3.parquet" if args.mock_library else "difficulty2.parquet")
    meta, molecules, row_key = common.prepare(args)
    df, shortfalls = build(args, (meta, molecules, row_key))
    common.finish(df, args, "make_difficulty2.py", meta, molecules, row_key, shortfalls)


if __name__ == "__main__":
    main()
