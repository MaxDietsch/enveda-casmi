"""Difficulty 1: the molecule stays in train, but in a different setting.

For each eval molecule one dimension separates the visible train rows from
the eval spectra:

  instrument  visible = non-timsTOF rows in the eval adduct; all timsTOF hidden
  adduct      visible = timsTOF rows in the other adducts; the rest hidden
  ce          visible = timsTOF rows in the eval adduct with a held-back CE

--change-mix gives the share of molecules per change type (normalised to 1).
The default favours `instrument`, the most realistic class-1 case: public
library spectra of a test molecule mostly come from other instruments.
Quotas are filled in the order instrument (rarest), adduct, ce; a molecule
gets one change type at most. Shortfalls are logged, not padded.

Usage:
    uv run python dataset-creation/make_difficulty1.py [--n-molecules 1000]
        [--change-mix instrument=0.6,adduct=0.2,ce=0.2] [--diversity-tc 0.6]
        [--seed 2026] [--out dataset-creation/out/difficulty1.parquet]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log  # noqa: E402

CHANGES = ("instrument", "adduct", "ce")


def qualifying(meta: pd.DataFrame) -> dict:
    """change -> {molecule: sorted list of qualifying eval adducts}."""
    e = meta.loc[meta["eligible"], ["mol", "adduct", "ce"]]
    e_pairs = e[["mol", "adduct"]].drop_duplicates()
    other = meta.loc[~meta["is_tims"], ["mol", "adduct"]].dropna().drop_duplicates()
    inst = e_pairs.merge(other, on=["mol", "adduct"])
    n_add = e_pairs.groupby("mol")["adduct"].nunique()
    add = e_pairs[e_pairs["mol"].isin(n_add.index[n_add >= 2])]
    n_ce = e.groupby(["mol", "adduct"])["ce"].nunique().reset_index()
    ce = n_ce.loc[n_ce["ce"] >= 2, ["mol", "adduct"]]

    def to_dict(pairs):
        return {int(m): sorted(g.tolist()) for m, g in pairs.groupby("mol")["adduct"]}
    return {"instrument": to_dict(inst), "adduct": to_dict(add), "ce": to_dict(ce)}


def choose_rows(rng: np.random.Generator, meta_cols: dict, rows: np.ndarray, change: str,
                adducts: list[str]):
    """(eval rows, visible rows) of one molecule for one change type."""
    elig = meta_cols["eligible"][rows]
    tims = meta_cols["is_tims"][rows]
    add = meta_cols["adduct"][rows]
    a = adducts[int(rng.integers(len(adducts)))]
    in_a = add == a
    if change == "instrument":
        pool = rows[elig & in_a]
        visible = rows[~tims & in_a]
    elif change == "adduct":
        pool = rows[elig & in_a]
        visible = rows[tims & ~in_a & pd.notna(add)]
    else:
        ce = meta_cols["ce"][rows]
        tuples = sorted({ce[i] for i in np.nonzero(elig & in_a)[0]})
        perm = [tuples[i] for i in rng.permutation(len(tuples))]
        n_eval = int(rng.integers(1, len(tuples)))
        eval_ce, vis_ce = set(perm[:n_eval]), set(perm[n_eval:])
        pool = rows[elig & in_a & np.array([c in eval_ce for c in ce], dtype=bool)]
        visible = rows[tims & in_a & np.array([c in vis_ce for c in ce], dtype=bool)]
    pool = np.sort(pool)
    k = common.draw_n_spectra(rng, len(pool))
    return rng.choice(pool, size=k, replace=False), visible


def select_difficulty1(meta: pd.DataFrame, molecules: pd.DataFrame, n: int, rng: np.random.Generator,
                       exclude_keys: set, used_ids: set, rows_index, change_mix: dict,
                       diversity: "common.Diversity", base_weights: np.ndarray | None = None,
                       ion_pos_share: float = 0.7) -> tuple[list[dict], dict]:
    """Pick up to n molecules split over change types. Returns (records, shortfalls).

    exclude_keys (molecule indices, including neighbours hidden by difficulty
    2/3) and diversity are updated in place.
    """
    order, start = rows_index
    qual = qualifying(meta)
    base = common.eval_candidates(molecules)
    names = [c for c in CHANGES if change_mix.get(c, 0) > 0]
    quotas = dict(zip(names, common.split_counts(n, [change_mix[c] for c in names])))
    log(f"[d1] qualifying molecules: " + ", ".join(f"{c} {len(qual[c]):,}" for c in CHANGES)
        + f"; quotas {quotas}")
    cols = {c: meta[c].to_numpy() for c in ("eligible", "is_tims", "adduct", "ce")}

    records, shortfalls = [], {}
    for change in names:
        cand = base[np.isin(base, list(qual[change]))]
        cand = cand[~np.isin(cand, list(exclude_keys))]
        picked = common.pick_diverse(rng, molecules, cand, quotas[change], diversity,
                                     ion_pos_share, base_weights)
        exclude_keys.update(picked)
        for m in picked:
            rows = order[start[m]:start[m + 1]]
            ev, visible = choose_rows(rng, cols, rows, change, qual[change][m])
            hidden = np.setdiff1d(rows, visible)
            records.append(common.make_record(
                molecules, m, common.new_molecule_id(rng, 1, used_ids), 1, change, ev, hidden))
        if len(picked) < quotas[change]:
            shortfalls[f"difficulty1_{change}"] = quotas[change] - len(picked)
        log(f"[d1] {change}: picked {len(picked)}/{quotas[change]} from {len(cand):,} candidates")
    return records, shortfalls


def add_args(parser: argparse.ArgumentParser) -> None:
    common.add_common_args(parser)
    parser.add_argument("--change-mix", default="instrument=0.6,adduct=0.2,ce=0.2",
                        help="share of molecules per change type: instrument, adduct, ce")
    parser.add_argument("--out", type=Path, default=common.OUT / "difficulty1.parquet")


def build(args: argparse.Namespace, prepared=None) -> tuple[pd.DataFrame, dict]:
    meta, molecules, row_key = prepared or common.prepare(args)
    sim, diversity, base = common.selection_context(args, molecules)
    rng = np.random.default_rng(args.seed)
    records, shortfalls = select_difficulty1(
        meta, molecules, args.n_molecules, rng, set(), set(), common.mol_rows(meta),
        common.parse_mix(args.change_mix, CHANGES), diversity, base, ion_pos_share=args.ion_pos_share)
    df = common.records_to_frame(records)
    return common.annotate_similarity(df, molecules, row_key, sim), shortfalls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(parser)
    args = parser.parse_args()
    meta, molecules, row_key = common.prepare(args)
    df, shortfalls = build(args, (meta, molecules, row_key))
    common.finish(df, args, "make_difficulty1.py", meta, molecules, row_key, shortfalls)


if __name__ == "__main__":
    main()
