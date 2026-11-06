"""Difficulty 2 (and 3) with adduct-stratified sampling.

Same split format and semantics as make_difficulty2.py (every eval molecule
has all its train rows hidden; --neighbour-removed-frac of them also lose
every similar molecule; --mock-library flags difficulty 3), but the eval
molecules are drawn in per-adduct buckets so that all ten test adducts are
exercised as far as the eligible pool allows:

  - for each adduct a with quota q, q molecules with at least one *eligible
    row in adduct a* are drawn with the usual mass-bin / ion / diversity
    weighting, and their eval spectra come from those adduct-a rows only
    (the hidden test measures a molecule in one adduct, not a mix);
  - buckets are filled rarest adduct first ([M-H2O-H]-, [M-2H2O+H]+, ...,
    [M+H]+), so rare-adduct molecules are not squeezed out by the common
    ones; a molecule is evaluated at most once;
  - quotas the pool cannot meet are logged as shortfalls, never padded.
    The three dehydrated adducts have almost no eligible timsTOF rows
    ([M-H2O-H]-: 1 molecule, [M-2H2O+H]+: 7, [M-H2O+H]+: 22), so their
    quotas are upper bounds;
  - any remaining slots up to --n-molecules are filled from [M+H]+ and
    [M-H]- in a 5:3 ratio, keeping the two dominant adducts represented.

Usage:
    uv run python dataset-creation/make_difficulty2_adducts.py [--n-molecules 1000]
        [--adduct-quotas "[M+H]+=250,[M-H]-=150,..."] [--neighbour-removed-frac 0.10]
        [--neighbour-tc 0.5] [--max-removed-neighbours 200] [--mock-library]
        [--seed 2026] [--out dataset-creation/out/difficulty2_adducts.parquet]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import TEST_ADDUCTS, log  # noqa: E402
from make_difficulty2 import add_removal_args, check_frac  # noqa: E402

# molecule-count quotas per adduct (pragmatic blend: the common pairs stay
# largest, every rare adduct is exercised up to what the pool allows)
DEFAULT_QUOTAS = {
    "[M+H]+": 250, "[M-H]-": 150, "[M+CH2O2-H]-": 120, "[M+Na]+": 120,
    "[M+NH4]+": 100, "[M+Cl]-": 80, "[M+K]+": 60,
    "[M-H2O+H]+": 30, "[M-2H2O+H]+": 10, "[M-H2O-H]-": 5,
}
# fill rare buckets first so they are not crowded out by the common ones
BUCKET_ORDER = ("[M-H2O-H]-", "[M-2H2O+H]+", "[M-H2O+H]+", "[M+K]+", "[M+Cl]-",
                "[M+NH4]+", "[M+Na]+", "[M+CH2O2-H]-", "[M-H]-", "[M+H]+")
FILL_ADDUCTS = ("[M+H]+", "[M-H]-")     # remainder slots, 5:3
FILL_SHARES = (5.0, 3.0)


def parse_quotas(text: str) -> dict[str, int]:
    """"[M+H]+=250,[M-H]-=150,..." -> dict of molecule counts per adduct."""
    out: dict[str, int] = {}
    for part in text.split(","):
        k, v = part.split("=")
        k = k.strip()
        if k not in TEST_ADDUCTS:
            raise ValueError(f"unknown adduct {k!r}; expected one of {TEST_ADDUCTS}")
        n = int(v)
        if n < 0:
            raise ValueError(f"quota for {k} must be >= 0, got {n}")
        out[k] = n
    return out


def select_adduct_stratified(meta: pd.DataFrame, molecules: pd.DataFrame, n: int,
                             quotas: dict[str, int], rng: np.random.Generator,
                             sim: "common.Similarity", diversity: "common.Diversity",
                             base_weights: np.ndarray | None,
                             neighbour_removed_frac: float = 0.10,
                             neighbour_tc: float = 0.5,
                             max_removed_neighbours: int = 200,
                             mock_library: bool = False,
                             ion_pos_share: float = 0.7) -> tuple[list[dict], dict]:
    """Pick n unseen molecules, stratified over adducts. Returns (records, shortfalls)."""
    difficulty = 3 if mock_library else 2
    order, start = common.mol_rows(meta)
    eligible = meta["eligible"].to_numpy()
    adducts = meta["adduct"].to_numpy()
    mol_of_row = meta["mol"].to_numpy()

    # molecules with at least one eligible row per test adduct
    elig_rows = np.flatnonzero(eligible)
    base_cand = set(common.eval_candidates(molecules).tolist())
    bucket_mols = {}
    for a in quotas:
        m = np.unique(mol_of_row[elig_rows][adducts[elig_rows] == a])
        bucket_mols[a] = np.array(sorted(base_cand & set(m.tolist())), dtype=np.int64)

    used: set[int] = set()
    picked_by: dict[str, list[int]] = {}

    def draw_bucket(a: str, quota: int) -> list[int]:
        cand = np.array(sorted(set(bucket_mols[a].tolist()) - used), dtype=np.int64)
        picked = common.pick_diverse(rng, molecules, cand, quota, diversity,
                                     ion_pos_share, base_weights) if len(cand) else []
        used.update(picked)
        log(f"[d{difficulty}] {a}: {len(picked)}/{quota} molecules "
            f"(pool {len(bucket_mols[a])})")
        return picked

    for a in BUCKET_ORDER:
        if a in quotas and quotas[a] > 0:
            picked_by[a] = draw_bucket(a, quotas[a])
    total = len(used)
    remaining = n - total
    if remaining > 0:
        for a, k in zip(FILL_ADDUCTS, common.split_counts(remaining, list(FILL_SHARES))):
            picked_by[a] = picked_by.get(a, []) + draw_bucket(a, k)
    all_picked = [m for a in BUCKET_ORDER for m in picked_by.get(a, [])]

    # Neighbourhood removal, as in make_difficulty2.py
    n_remove = int(round(neighbour_removed_frac * len(all_picked)))
    removed = {}
    if n_remove:
        order_pick = [all_picked[i] for i in rng.permutation(len(all_picked))]
        for m, nb in zip(order_pick, sim.neighbours(order_pick, neighbour_tc)):
            nb = nb[nb != m]
            if 1 <= len(nb) <= max_removed_neighbours:
                removed[m] = nb
                if len(removed) == n_remove:
                    break
    keys = molecules["key"].to_numpy()
    used_ids: set[str] = set()

    records = []
    for a in BUCKET_ORDER:
        for m in picked_by.get(a, []):
            rows = order[start[m]:start[m + 1]]
            elig_a = np.sort(rows[eligible[rows] & (adducts[rows] == a)])
            k = common.draw_n_spectra(rng, len(elig_a))
            ev = rng.choice(elig_a, size=k, replace=False)
            nb = removed.get(m, np.empty(0, np.int64))
            records.append(common.make_record(
                molecules, m, common.new_molecule_id(rng, difficulty, used_ids), difficulty,
                None, ev, rows, removed_keys=keys[nb].tolist(),
                exclude_from_library=mock_library))

    shortfalls = {}
    for a in BUCKET_ORDER:
        if a in quotas and quotas[a] > 0 and len(picked_by.get(a, [])) < quotas[a]:
            shortfalls[f"{a}_molecules"] = quotas[a] - len(picked_by.get(a, []))
    if len(all_picked) < n:
        shortfalls[f"difficulty{difficulty}_molecules"] = n - len(all_picked)
    if len(removed) < n_remove:
        shortfalls[f"difficulty{difficulty}_neighbours_removed"] = n_remove - len(removed)
    if removed:
        hidden = np.unique(np.concatenate(list(removed.values())))
        n_rows = int(np.isin(mol_of_row, hidden).sum())
        sizes = np.array([len(v) for v in removed.values()])
        log(f"[d{difficulty}] neighbourhood removal (Tc >= {neighbour_tc}): "
            f"{len(removed)} molecules, neighbours per molecule median "
            f"{np.median(sizes):.0f} max {sizes.max()}, {len(hidden):,} molecules / "
            f"{n_rows:,} extra rows hidden")
    log(f"[d{difficulty}] picked {len(all_picked)}/{n} molecules, "
        f"{sum(r['n_eval_spectra'] for r in records)} eval spectra")
    return records, shortfalls


def build(args: argparse.Namespace, prepared=None) -> tuple[pd.DataFrame, dict]:
    """Build the split table (also used by check_split.py for the rerun check)."""
    check_frac(args.neighbour_removed_frac)
    quotas = parse_quotas(args.adduct_quotas) if getattr(args, "adduct_quotas", None) \
        else dict(DEFAULT_QUOTAS)
    if sum(quotas.values()) > args.n_molecules:
        raise SystemExit(f"adduct quotas sum to {sum(quotas.values())} but --n-molecules is "
                         f"{args.n_molecules}; quotas are the primary draw, the remainder is "
                         "filled from [M+H]+ and [M-H]-")
    meta, molecules, row_key = prepared or common.prepare(args)
    sim, diversity, base = common.selection_context(args, molecules)
    rng = np.random.default_rng(args.seed)
    records, shortfalls = select_adduct_stratified(
        meta, molecules, args.n_molecules, quotas, rng, sim, diversity, base,
        neighbour_removed_frac=args.neighbour_removed_frac, neighbour_tc=args.neighbour_tc,
        max_removed_neighbours=args.max_removed_neighbours, mock_library=args.mock_library,
        ion_pos_share=args.ion_pos_share)
    df = common.records_to_frame(records)
    return common.annotate_similarity(df, molecules, row_key, sim), shortfalls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(parser)
    add_removal_args(parser)
    parser.add_argument("--adduct-quotas", default=None,
                        help="molecule quotas per adduct, e.g. \"[M+H]+=250,[M-H]-=150,...\" "
                             f"(default: {DEFAULT_QUOTAS})")
    parser.add_argument("--mock-library", action="store_true",
                         help="difficulty 3: also exclude the structures from the candidate library")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    quotas = parse_quotas(args.adduct_quotas) if args.adduct_quotas else dict(DEFAULT_QUOTAS)
    if sum(quotas.values()) > args.n_molecules:
        raise SystemExit(f"adduct quotas sum to {sum(quotas.values())} but --n-molecules is "
                         f"{args.n_molecules}; quotas are the primary draw, the remainder is "
                         "filled from [M+H]+ and [M-H]-")
    if args.out is None:
        stem = "difficulty3_adducts.parquet" if args.mock_library else "difficulty2_adducts.parquet"
        args.out = common.OUT / stem

    meta, molecules, row_key = common.prepare(args)
    df, shortfalls = build(args, (meta, molecules, row_key))
    common.finish(df, args, "make_difficulty2_adducts.py", meta, molecules, row_key,
                 shortfalls, extra={"adduct_quotas": quotas})


if __name__ == "__main__":
    main()
