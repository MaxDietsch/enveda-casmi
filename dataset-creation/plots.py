"""Diagnostic plots for an eval-set file, next to the hidden-test targets.

Written automatically by every make_*.py script as <out>.png next to the
output parquet. Can also be run on its own:
    uv run python dataset-creation/plots.py dataset-creation/out/overall.parquet

Panels: neutral mass (histogram with medians, and bin shares versus target),
spectra per molecule, adducts, collision energy (single values and number of
merged energies), peaks per spectrum, base-peak intensity, the difficulty /
change / neighbour-removal composition, similarity to the nearest visible
train molecule per difficulty, and neighbourhood-removal sizes.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

REAL = "#2a78d6"      # realised eval set
TARGET = "#eb6834"    # hidden-test target
POOL = "#b9b8b2"      # eligible timsTOF pool (context)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
TEST_MASS_MEDIAN = 348.0


def _style(ax, title):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def target_spectra_pmf(kmax: int = common.MAX_SPECTRA) -> np.ndarray:
    k = np.arange(1, kmax + 1)
    p = 0.25 * 0.75 ** (k - 1)
    return p / p.sum()


def plot_split(df: pd.DataFrame, meta: pd.DataFrame, molecules: pd.DataFrame, out_png: Path,
               title: str = "", neighbour_tc: float | None = None) -> Path:
    rows = np.concatenate([np.asarray(r, dtype=np.int64) for r in df["eval_row_ids"]]) \
        if len(df) else np.empty(0, np.int64)
    ev = meta.iloc[rows]
    fig, axes = plt.subplots(4, 3, figsize=(15, 16.5))
    fig.patch.set_facecolor("white")
    ax = axes.ravel()

    # 1 mass histogram + medians, eligible pool as context
    mass = df["exact_mass"].to_numpy(dtype=float)
    pool = molecules["exact_mass"].to_numpy()[common.eval_candidates(molecules)]
    bins = np.arange(150, 1175, 25)
    ax[0].hist(pool, bins=bins, density=True, color=POOL, label=f"eligible pool (n={len(pool):,})")
    ax[0].hist(mass, bins=bins, density=True, histtype="step", linewidth=2, color=REAL,
               label=f"eval (n={len(mass)})")
    med = float(np.median(mass)) if len(mass) else np.nan
    ax[0].axvline(med, color=REAL, linewidth=2, linestyle="-")
    ax[0].axvline(TEST_MASS_MEDIAN, color=TARGET, linewidth=2, linestyle="--")
    ax[0].text(0.98, 0.95, f"median eval {med:.0f} Da\nmedian test {TEST_MASS_MEDIAN:.0f} Da",
               transform=ax[0].transAxes, ha="right", va="top", fontsize=8, color=INK)
    ax[0].set_xlabel("neutral monoisotopic mass (Da)", fontsize=8)
    ax[0].legend(fontsize=7, frameon=False, loc="center right")
    _style(ax[0], "Mass distribution")

    # 2 mass-bin shares vs target
    names = list(common.TEST_TARGETS["mass_bin_shares"])
    mbin = np.clip(np.searchsorted(common.MASS_BINS, mass, side="right") - 1, 0, len(names) - 1)
    real = np.array([(mbin == i).mean() if len(mass) else 0 for i in range(len(names))])
    x = np.arange(len(names))
    ax[1].bar(x - 0.2, real, 0.38, color=REAL, label="eval")
    ax[1].bar(x + 0.2, common.MASS_TARGET, 0.38, color=TARGET, label="target")
    for i, v in enumerate(real):
        ax[1].text(i - 0.2, v + 0.005, f"{v:.2f}", ha="center", fontsize=7, color=INK2)
    ax[1].set_xticks(x, names, fontsize=8)
    ax[1].set_xlabel("mass bin (Da)", fontsize=8)
    ax[1].set_ylabel("share of molecules", fontsize=8)
    ax[1].legend(fontsize=7, frameon=False)
    _style(ax[1], "Mass-bin shares vs target")

    # 3 spectra per molecule vs truncated geometric
    n = df["n_eval_spectra"].to_numpy()
    k = np.arange(1, common.MAX_SPECTRA + 1)
    cnt = np.array([(n == i).mean() if len(n) else 0 for i in k])
    ax[2].bar(k, cnt, 0.8, color=REAL, label="eval")
    ax[2].plot(k, target_spectra_pmf(), "o-", color=TARGET, markersize=4, linewidth=1.5,
               label="draw distribution (before capping)")
    ax[2].text(0.98, 0.7, f"median eval {np.median(n) if len(n) else 0:.0f}\nmedian test 3\n"
               f"spectra {n.sum()} (test rate: ~{round(3.75 * len(n))})", transform=ax[2].transAxes, ha="right",
               fontsize=8, color=INK)
    ax[2].set_xlabel("eval spectra per molecule", fontsize=8)
    ax[2].legend(fontsize=7, frameon=False)
    _style(ax[2], "Spectra per molecule")

    # 4 adducts (all ten, zeros shown)
    ac = ev["adduct"].value_counts().reindex(common.TEST_ADDUCTS, fill_value=0)
    y = np.arange(len(ac))[::-1]
    ax[3].barh(y, ac.to_numpy(), color=REAL)
    for yi, v in zip(y, ac.to_numpy()):
        ax[3].text(v, yi, f" {v}", va="center", fontsize=7, color=INK2)
    ax[3].set_yticks(y, ac.index, fontsize=8)
    ax[3].set_xlabel("eval spectra", fontsize=8)
    pos = (ev["ionization_mode"] == "positive").mean() if len(ev) else 0
    ax[3].text(0.98, 0.05, f"positive {pos:.0%} of spectra", transform=ax[3].transAxes,
               ha="right", fontsize=8, color=INK)
    ax[3].grid(axis="x", color=GRID, linewidth=0.8)
    _style(ax[3], "Adducts (spectra)")
    ax[3].grid(axis="y", visible=False)

    # 5 CE values of single-energy spectra
    ce = ev["ce"].tolist()
    single = np.array([t[0] for t in ce if t is not None and len(t) == 1], dtype=float)
    allv = np.concatenate([np.asarray(t, float) for t in ce if t is not None]) if ce else np.empty(0)
    if len(single):
        vals, c = np.unique(np.round(single), return_counts=True)
        ax[4].bar(vals, c, width=3, color=REAL)
    ax[4].set_xlabel("collision energy (eV), single-energy spectra", fontsize=8)
    ax[4].set_ylabel("eval spectra", fontsize=8)
    if len(allv):
        ax[4].text(0.98, 0.95, f"all CE values: median {np.median(allv):.0f} eV\n"
                   f"5–95%: {np.quantile(allv, .05):.0f}–{np.quantile(allv, .95):.0f} eV",
                   transform=ax[4].transAxes, ha="right", va="top", fontsize=8, color=INK)
    _style(ax[4], "Collision energy (single)")

    # 6 number of energies per spectrum (single vs merged)
    nce = pd.Series([len(t) if t is not None else 0 for t in ce]).value_counts().sort_index()
    ax[5].bar(nce.index.astype(str), nce.to_numpy(), color=REAL)
    for i, v in enumerate(nce.to_numpy()):
        ax[5].text(i, v, f"{v / max(len(ce), 1):.0%}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax[5].set_xlabel("energies per spectrum (1 = single, >1 = merged)", fontsize=8)
    ax[5].set_ylabel("eval spectra", fontsize=8)
    _style(ax[5], "Single vs merged CE")

    # 7 peaks per spectrum
    npk = ev["num_peaks"].to_numpy(dtype=float)
    if len(npk):
        ax[6].hist(npk, bins=np.logspace(0, np.log10(max(npk.max(), 10)) + 0.05, 30), color=REAL)
        ax[6].set_xscale("log")
        ax[6].text(0.98, 0.95, f"median {np.median(npk):.0f} peaks", transform=ax[6].transAxes,
                   ha="right", va="top", fontsize=8, color=INK)
    ax[6].set_xlabel("peaks per spectrum (before >precursor+2 Da removal)", fontsize=8)
    _style(ax[6], "Peaks per spectrum")

    # 8 base peak intensity
    bpi = ev["base_peak_intensity"].to_numpy(dtype=float)
    bpi = bpi[np.isfinite(bpi) & (bpi > 0)]
    if len(bpi):
        ax[7].hist(np.log10(bpi), bins=30, color=REAL)
        ax[7].axvline(3, color=TARGET, linestyle="--", linewidth=2)
        ax[7].text(3.05, ax[7].get_ylim()[1] * 0.95, "test cut-off 1,000", fontsize=7,
                   color=INK2, va="top")
    ax[7].set_xlabel("log10 base-peak intensity (raw counts)", fontsize=8)
    _style(ax[7], "Base-peak intensity")

    # 9 composition
    comp = df.assign(group=np.where(df["difficulty"] == 1, "d1 · " + df["change"].fillna(""),
                                    "d" + df["difficulty"].astype(int).astype(str)))
    comp["group"] = comp["group"] + np.where(comp["neighbours_removed"], " · neighbours removed", "")
    gc = comp["group"].value_counts().sort_index()
    y = np.arange(len(gc))[::-1]
    ax[8].barh(y, gc.to_numpy(), color=REAL)
    for yi, v in zip(y, gc.to_numpy()):
        ax[8].text(v, yi, f" {v} ({v / max(len(df), 1):.0%})", va="center", fontsize=7, color=INK2)
    ax[8].set_yticks(y, gc.index, fontsize=8)
    ax[8].set_xlabel("eval molecules", fontsize=8)
    ax[8].grid(axis="x", color=GRID, linewidth=0.8)
    _style(ax[8], "Composition")
    ax[8].grid(axis="y", visible=False)

    # 10 similarity to the nearest visible train molecule, per difficulty
    tcbins = np.linspace(0, 1, 41)
    level_colors = {1: REAL, 2: TARGET, 3: "#1baf7a"}
    for d, col in level_colors.items():
        v = df.loc[df["difficulty"] == d, "max_visible_tc"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            ax[9].hist(v, bins=tcbins, histtype="step", linewidth=2, color=col,
                       label=f"difficulty {d} (n={len(v)}, median {np.median(v):.2f})")
    if neighbour_tc is not None:
        ax[9].axvline(neighbour_tc, color=INK2, linestyle=":", linewidth=1.5)
        ax[9].text(neighbour_tc, 0.55, f" removal\n threshold {neighbour_tc}", fontsize=7, color=INK2,
                   va="center", transform=ax[9].get_xaxis_transform())
    ax[9].set_xlabel("max Tanimoto (Morgan r2) to any visible train molecule", fontsize=8)
    ax[9].set_ylabel("eval molecules", fontsize=8)
    ax[9].legend(fontsize=7, frameon=False, loc="upper left")
    _style(ax[9], "Similarity to nearest visible molecule")

    # 11 neighbour-removal sizes
    nr = df.loc[df["neighbours_removed"].astype(bool), "removed_keys"].map(len).to_numpy()
    if len(nr):
        ax[10].hist(nr, bins=min(30, max(5, len(np.unique(nr)))), color=REAL)
        ax[10].text(0.98, 0.95, f"{len(nr)} molecules ({len(nr) / max(len(df), 1):.0%})\n"
                    f"median {np.median(nr):.0f}, max {nr.max()} neighbours hidden",
                    transform=ax[10].transAxes, ha="right", va="top", fontsize=8, color=INK)
    else:
        ax[10].text(0.5, 0.5, "no neighbourhood removal", transform=ax[10].transAxes,
                    ha="center", fontsize=9, color=INK2)
    ax[10].set_xlabel("neighbours hidden per molecule", fontsize=8)
    _style(ax[10], "Neighbourhood removal")
    ax[11].axis("off")

    fig.suptitle(title or out_png.stem, fontsize=13, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    common.log(f"[plots] wrote {out_png}")
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("split", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="default: <split>.png")
    args = parser.parse_args()
    info = common.read_split_info(args.split)
    p = info.get("params", {})
    train = Path(p.get("train", common.TRAIN))
    cache_dir = Path(p.get("cache_dir", common.CACHE))
    meta = common.load_meta(train)
    molecules = pd.read_parquet(cache_dir / "molecules.parquet")
    row_key = np.load(cache_dir / "row_key.npy")
    common.attach_keys(meta, molecules, row_key, p.get("ppm_tol", 10.0))
    df = pd.read_parquet(args.split)
    plot_split(df, meta, molecules, args.out or args.split.with_suffix(".png"),
               f"{args.split.name} · {info.get('script', '')}", p.get("neighbour_tc"))


if __name__ == "__main__":
    main()
