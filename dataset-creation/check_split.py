"""Validate an eval-set file produced by the make_*.py scripts.

Checks that no eval spectrum or hidden molecule leaks into the visible train
rows, that neighbour-removed molecules have no similar visible molecule, that difficulty-1 molecules differ in the declared dimension, that the
eval rows are eligible, that molecule sets are disjoint, that no two eval
molecules are similar (Tanimoto >= --diversity-tc), and (unless --no-rerun) that rebuilding with the same seed gives the
same table. Prints the sanity report. Exits 1 on any failure.

Usage:
    uv run python dataset-creation/check_split.py dataset-creation/out/overall.parquet [--no-rerun]
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import log  # noqa: E402
from dataloader import load_split  # noqa: E402

PATH_PARAMS = ("train", "cache_dir", "out")


def params_namespace(params: dict) -> argparse.Namespace:
    ns = argparse.Namespace(**params)
    for k in PATH_PARAMS:
        if getattr(ns, k, None) is not None:
            setattr(ns, k, Path(getattr(ns, k)))
    return ns


def run_checks(path: Path, rerun: bool = True) -> bool:
    info = common.read_split_info(path)
    args = params_namespace(info["params"])
    split = load_split(path, args.train, args.cache_dir)
    df, mask = split.truth, split.train_mask
    meta = common.load_meta(args.train)
    molecules, row_key = common.ensure_cache(args.train, args.cache_dir, meta=meta)
    common.attach_keys(meta, molecules, row_key, args.ppm_tol)
    key_index = {k: i for i, k in enumerate(molecules["key"])}
    failures = []

    def check(name, ok, detail=""):
        log(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(name)

    ev_all = split.eval_row_ids
    # 1
    check("1 no eval row visible", not mask[ev_all].any(), f"{int(mask[ev_all].sum())} visible")
    # eval rows belong to their molecule
    bad_owner = [mid for mid, k, rows in zip(df["molecule_id"], df["key"], df["eval_row_ids"])
                 if (row_key[np.asarray(rows, dtype=np.int64)] != key_index[k]).any()]
    check("eval rows carry the molecule's key", not bad_owner, str(bad_owner[:5]))

    vis_mol = row_key[mask]
    # 2
    d23 = df[df["difficulty"] >= 2]
    leak = np.isin([key_index[k] for k in d23["key"]], vis_mol)
    check("2 difficulty 2/3 keys absent from visible train", not leak.any(),
          str(d23.loc[leak, "molecule_id"].tolist()[:5]))
    # 3
    cols = {c: meta[c].to_numpy() for c in ("is_tims", "adduct", "ce")}
    order, start = common.mol_rows(meta)
    bad = []
    for r in df[df["difficulty"] == 1].itertuples():
        m = key_index[r.key]
        rows = order[start[m]:start[m + 1]]
        vis = rows[mask[rows]]
        ev = np.asarray(r.eval_row_ids, dtype=np.int64)
        if len(vis) == 0:
            bad.append((r.molecule_id, "no visible row"))
            continue
        ev_add = set(cols["adduct"][ev])
        if r.change == "instrument":
            ok = not cols["is_tims"][vis].any()
        elif r.change == "adduct":
            ok = not np.isin(cols["adduct"][vis], list(ev_add)).any()
        else:
            ev_ce = set(cols["ce"][ev])
            ok = not any(a in ev_add and c in ev_ce for a, c in zip(cols["adduct"][vis], cols["ce"][vis]))
        if not ok:
            bad.append((r.molecule_id, r.change))
    check("3 difficulty 1 visible rows differ in the declared dimension", not bad, str(bad[:5]))
    # 4
    visible = np.zeros(len(molecules), dtype=bool)
    visible[vis_mol] = True
    sim = common.similarity(molecules, args.cache_dir)
    res = sim.max_visible([key_index[k] for k in df["key"]], visible)
    max_tc = np.array([t for t, _ in res])
    nr = df["neighbours_removed"].to_numpy(dtype=bool)
    tc = getattr(args, "neighbour_tc", None)
    if nr.any():
        leak = nr & (max_tc >= tc)
        check(f"4a neighbour-removed molecules have no visible molecule with Tc >= {tc}", not leak.any(),
              str(df.loc[leak, "molecule_id"].tolist()[:5]))
    removed = {k for keys in df["removed_keys"] for k in keys}
    check("4b removed neighbours have no visible row",
          not np.isin([key_index[k] for k in removed], vis_mol).any() if removed else True)
    check("4c stored max_visible_tc matches recomputation",
          np.allclose(df["max_visible_tc"].to_numpy(dtype=float), max_tc, atol=1e-3))
    # 5
    elig = meta["eligible"].to_numpy()[ev_all]
    tims = meta["is_tims"].to_numpy()[ev_all]
    add_ok = meta["adduct"].isin(common.TEST_ADDUCTS).to_numpy()[ev_all]
    check("5 eval rows eligible, timsTOF, test adducts", elig.all() and tims.all() and add_ok.all(),
          f"{int((~elig).sum())} not eligible")
    # 6
    check("6a molecule_id unique", df["molecule_id"].is_unique)
    check("6b molecule sets disjoint", df["key"].is_unique,
          str(df.loc[df["key"].duplicated(), "key"].tolist()[:5]))
    check("6c eval rows unique", len(np.unique(ev_all)) == len(ev_all))
    # 7
    dtc = args.diversity_tc
    ev_idx = np.array([key_index[k] for k in df["key"]])
    close = [(df["molecule_id"].iat[i], df["molecule_id"].iat[j])
             for i in range(len(ev_idx)) for j in np.nonzero(sim.tc_to(ev_idx[i], ev_idx[i + 1:]) >= dtc)[0] + i + 1]
    check(f"7 no two eval molecules with Tanimoto >= {dtc}", not close, str(close[:5]))
    # 8
    if rerun:
        module = importlib.import_module(Path(info["script"]).stem)
        df2, _ = module.build(args, (meta, molecules, row_key))
        h1, h2 = info.get("content_hash"), common.content_hash(df2)
        check("8 rerun with the same seed gives an identical table", h1 == h2 == common.content_hash(df),
              f"{(h1 or '?')[:12]} vs {h2[:12]}")
    # 9
    log(f"shortfalls: {info.get('shortfalls')}")
    common.sanity_report(df, meta, mask, row_key, molecules)
    log(f"{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'} for {path}")
    return not failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split", type=Path)
    parser.add_argument("--no-rerun", action="store_true", help="skip the reproducibility check")
    args = parser.parse_args()
    sys.exit(0 if run_checks(args.split, rerun=not args.no_rerun) else 1)


if __name__ == "__main__":
    main()
