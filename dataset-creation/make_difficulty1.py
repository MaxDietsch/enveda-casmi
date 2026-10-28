"""Difficulty 1: the molecule stays in train, but in a different setting.

For each eval molecule one dimension separates the visible train rows from
the eval spectra:

  instrument  visible = non-timsTOF rows in the eval adduct; all timsTOF hidden
  adduct      visible = timsTOF rows in the other adducts; the rest hidden
  ce          visible = timsTOF rows in the eval adduct with a held-back CE
  mixed       (advanced) two or three of the above at once: every visible row
              differs from every eval row in each held-out dimension of a
              combo; dims not held out keep the pins above (adduct = eval
              adduct, instrument = timsTOF, CE free). Feasibility is computed
              by construction per (molecule, adduct, combo); the mixed quota
              is split over combos by --mixed-dims, rarest combo first

--change-mix gives the share of molecules per change type (normalised to 1).
The default favours `instrument`, the most realistic class-1 case: public
library spectra of a test molecule mostly come from other instruments.
Quotas are filled in the order mixed (rarest), instrument, adduct, ce; a
molecule gets one change type at most. Shortfalls are logged, not padded.

Usage:
    uv run python dataset-creation/make_difficulty1.py [--n-molecules 1000]
        [--change-mix instrument=0.3,adduct=0.2,ce=0.2,mixed=0.3]
        [--mixed-dims instrument+ce=0.4,adduct+ce=0.3,instrument+adduct=0.2,instrument+adduct+ce=0.1]
        [--diversity-tc 0.6] [--seed 2026] [--out dataset-creation/out/difficulty1.parquet]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log  # noqa: E402

CHANGES = ("mixed", "instrument", "adduct", "ce")     # quota fill order (rarest first)
DIMS = ("instrument", "adduct", "ce")
MIXED_COMBOS = ("instrument+ce", "adduct+ce", "instrument+adduct", "instrument+adduct+ce")
MIXED_FILL_ORDER = ("instrument+adduct+ce", "instrument+adduct", "instrument+ce", "adduct+ce")  # rarest first
DEFAULT_CHANGE_MIX = "instrument=0.3,adduct=0.2,ce=0.2,mixed=0.3"
DEFAULT_MIXED_DIMS = "instrument+ce=0.4,adduct+ce=0.3,instrument+adduct=0.2,instrument+adduct+ce=0.1"


def _ce_feasible(e: set, c: set) -> bool:
    """A non-empty eval CE subset of e leaves a visible CE in c outside it."""
    return bool(e) and bool(c) and len(e | c) >= 2


def mixed_options(meta: pd.DataFrame) -> dict:
    """molecule -> sorted feasible (combo, eval adduct) pairs for change `mixed`.

    Feasible by construction: the adduct has eligible rows and some row of
    the molecule can stay visible while differing from every eval row in
    each held-out dim (instrument: non-timsTOF; adduct: another known adduct;
    ce: a known CE outside the eval CE set). One pass over meta.
    """
    ce_id = pd.factorize(meta["ce"])[0]                    # None -> -1
    d = pd.DataFrame({"mol": meta["mol"].to_numpy(), "adduct": meta["adduct"].to_numpy(),
                      "ce": ce_id, "tims": meta["is_tims"].to_numpy(),
                      "elig": meta["eligible"].to_numpy()})
    d = d[d["mol"].isin(d.loc[d["elig"], "mol"].unique()) & d["adduct"].notna()]
    d = d[["mol", "adduct", "ce", "tims", "elig"]].drop_duplicates()
    e_sets = d[d["elig"]].groupby(["mol", "adduct"])["ce"].agg(set)
    v_sets = d[d["ce"] >= 0].groupby(["mol", "tims", "adduct"])["ce"].agg(set)
    n_other = d[~d["tims"]].groupby("mol")["adduct"].agg(set)  # non-timsTOF adducts (any CE)
    vis = {}
    for (m, t, a), cs in v_sets.items():
        vis.setdefault((int(m), bool(t)), {})[a] = cs
    out = {}
    for (m, a), e in e_sets.items():
        m = int(m)
        tims_v, other_v = vis.get((m, True), {}), vis.get((m, False), {})
        opts = []
        if _ce_feasible(e, other_v.get(a, set())):
            opts.append(("instrument+ce", a))
        if _ce_feasible(e, set().union(*(c for b, c in tims_v.items() if b != a))):
            opts.append(("adduct+ce", a))
        if n_other.get(m, set()) - {a}:
            opts.append(("instrument+adduct", a))
        if _ce_feasible(e, set().union(*(c for b, c in other_v.items() if b != a))):
            opts.append(("instrument+adduct+ce", a))
        if opts:
            out.setdefault(m, []).extend(opts)
    return {m: sorted(o) for m, o in out.items()}


def qualifying(meta: pd.DataFrame) -> dict:
    """change -> {molecule: sorted list of qualifying eval adducts}; for `mixed`
    the union over combos, and "mixed_pairs" -> {molecule: feasible (combo, adduct)}."""
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
    mixed = mixed_options(meta)
    return {"instrument": to_dict(inst), "adduct": to_dict(add), "ce": to_dict(ce),
            "mixed": {m: sorted({a for _, a in ps}) for m, ps in mixed.items()}, "mixed_pairs": mixed}


def _choose_mixed(rng, meta_cols, rows, pairs, weights):
    """(eval rows, visible rows) for a (combo, adduct) drawn from the feasible pairs."""
    combos = sorted({c for c, _ in pairs})
    w = np.array([weights[c] for c in combos], dtype=float)
    combo = combos[int(rng.choice(len(combos), p=w / w.sum()))]
    adducts = sorted(a for c, a in pairs if c == combo)
    a = adducts[int(rng.integers(len(adducts)))]
    dims = combo.split("+")
    elig, tims = meta_cols["eligible"][rows], meta_cols["is_tims"][rows]
    add = meta_cols["adduct"][rows]
    in_a = add == a
    pool_m = elig & in_a
    vis_m = (~tims if "instrument" in dims else tims) & ((pd.notna(add) & ~in_a) if "adduct" in dims else in_a)
    if "ce" in dims:
        ce = meta_cols["ce"][rows]
        e_t = sorted({ce[i] for i in np.nonzero(pool_m)[0]})
        c_t = sorted({ce[i] for i in np.nonzero(vis_m)[0] if ce[i] is not None})
        anchors = [t for t in c_t if any(x != t for x in e_t)]  # a visible CE that some eval CE differs from
        anchor = anchors[int(rng.integers(len(anchors)))]
        rest = [e_t[i] for i in rng.permutation(len(e_t)) if e_t[i] != anchor]
        eval_ce = set(rest[: int(rng.integers(1, len(rest) + 1))])
        pool_m &= np.array([c in eval_ce for c in ce], dtype=bool)
        vis_m &= np.array([c is not None and c not in eval_ce for c in ce], dtype=bool)
    return rows[pool_m], rows[vis_m]


def choose_rows(rng: np.random.Generator, meta_cols: dict, rows: np.ndarray, change: str,
                adducts: list, mixed_weights: dict | None = None):
    """(eval rows, visible rows) of one molecule for one change type. For
    `mixed`, adducts is the molecule's feasible (combo, adduct) pairs and
    mixed_weights the combo weights (default --mixed-dims)."""
    if change == "mixed":
        weights = mixed_weights or common.parse_mix(DEFAULT_MIXED_DIMS, MIXED_COMBOS)
        pool, visible = _choose_mixed(rng, meta_cols, rows, adducts, weights)
        assert len(pool) and len(visible), "infeasible mixed pair"
        pool = np.sort(pool)
        k = common.draw_n_spectra(rng, len(pool))
        return rng.choice(pool, size=k, replace=False), visible
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


def changed_dims(meta_cols: dict, ev: np.ndarray, vis: np.ndarray) -> list[str]:
    """Dims in which every (eval, visible) pair differs. A visible row with an
    unknown adduct or CE never counts as differing in that dim."""
    out = []
    for d, col in (("instrument", "is_tims"), ("adduct", "adduct"), ("ce", "ce")):
        e, v = meta_cols[col][ev], meta_cols[col][vis]
        if len(vis) and all(x is not None and x == x for x in v) and not set(e) & set(v):
            out.append(d)
    return out


def change_slots(qual: dict, quotas: dict, mixed_dims: dict) -> list[tuple[str, str, int, dict]]:
    """(change, label, quota, {molecule: options}) in fill order. `mixed` is
    split into one slot per combo by mixed_dims, rarest combo first, so the
    realised combo shares follow mixed_dims (almost every mixed-feasible
    molecule is feasible only for adduct+ce)."""
    slots = []
    for change, n in quotas.items():
        if change != "mixed":
            slots.append((change, change, n, qual[change]))
            continue
        combos = [c for c in MIXED_FILL_ORDER if mixed_dims.get(c, 0) > 0]
        for c, k in zip(combos, common.split_counts(n, [mixed_dims[c] for c in combos])):
            opts = {m: [p for p in ps if p[0] == c] for m, ps in qual["mixed_pairs"].items()}
            slots.append(("mixed", f"mixed_{c}", k, {m: ps for m, ps in opts.items() if ps}))
    return slots


def select_difficulty1(meta: pd.DataFrame, molecules: pd.DataFrame, n: int, rng: np.random.Generator,
                       exclude_keys: set, used_ids: set, rows_index, change_mix: dict,
                       diversity: "common.Diversity", base_weights: np.ndarray | None = None,
                       ion_pos_share: float = 0.7, mixed_dims: dict | None = None) -> tuple[list[dict], dict]:
    """Pick up to n molecules split over change types. Returns (records, shortfalls).

    exclude_keys (molecule indices, including neighbours hidden by difficulty
    2/3) and diversity are updated in place.
    """
    order, start = rows_index
    qual = qualifying(meta)
    mixed_dims = mixed_dims or common.parse_mix(DEFAULT_MIXED_DIMS, MIXED_COMBOS)
    base = common.eval_candidates(molecules)
    names = [c for c in CHANGES if change_mix.get(c, 0) > 0]
    quotas = dict(zip(names, common.split_counts(n, [change_mix[c] for c in names])))
    log(f"[d1] qualifying molecules: " + ", ".join(f"{c} {len(qual[c]):,}" for c in CHANGES)
        + f"; quotas {quotas}")
    log("[d1] mixed-feasible molecules per combo: " + ", ".join(
        f"{c} {sum(any(p[0] == c for p in ps) for ps in qual['mixed_pairs'].values()):,}"
        for c in MIXED_COMBOS))
    cols = {c: meta[c].to_numpy() for c in ("eligible", "is_tims", "adduct", "ce")}

    records, shortfalls = [], {}
    for change, label, quota, options in change_slots(qual, quotas, mixed_dims):
        cand = base[np.isin(base, list(options))]
        cand = cand[~np.isin(cand, list(exclude_keys))]
        picked = common.pick_diverse(rng, molecules, cand, quota, diversity,
                                     ion_pos_share, base_weights)
        exclude_keys.update(picked)
        for m in picked:
            rows = order[start[m]:start[m + 1]]
            ev, visible = choose_rows(rng, cols, rows, change, options[m], mixed_dims)
            hidden = np.setdiff1d(rows, visible)
            records.append(common.make_record(
                molecules, m, common.new_molecule_id(rng, 1, used_ids), 1, change, ev, hidden))
        if len(picked) < quota:
            shortfalls[f"difficulty1_{label}"] = quota - len(picked)
        log(f"[d1] {label}: picked {len(picked)}/{quota} from {len(cand):,} candidates")
    return records, shortfalls


def parse_mixed_dims(args: argparse.Namespace) -> dict:
    return common.parse_mix(getattr(args, "mixed_dims", None) or DEFAULT_MIXED_DIMS, MIXED_COMBOS)


def add_args(parser: argparse.ArgumentParser) -> None:
    common.add_common_args(parser)
    parser.add_argument("--change-mix", default=DEFAULT_CHANGE_MIX,
                        help="share of molecules per change type: instrument, adduct, ce, mixed")
    parser.add_argument("--mixed-dims", default=DEFAULT_MIXED_DIMS,
                        help="mixed: weight per combo of held-out dims (drawn per molecule among its feasible combos)")
    parser.add_argument("--out", type=Path, default=common.OUT / "difficulty1.parquet")


def build(args: argparse.Namespace, prepared=None) -> tuple[pd.DataFrame, dict]:
    meta, molecules, row_key = prepared or common.prepare(args)
    sim, diversity, base = common.selection_context(args, molecules)
    rng = np.random.default_rng(args.seed)
    records, shortfalls = select_difficulty1(
        meta, molecules, args.n_molecules, rng, set(), set(), common.mol_rows(meta),
        common.parse_mix(args.change_mix, CHANGES), diversity, base, ion_pos_share=args.ion_pos_share,
        mixed_dims=parse_mixed_dims(args))
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
