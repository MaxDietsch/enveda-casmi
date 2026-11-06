"""Apply spectrum cleaning to train and test data and save cleaned copies.

This is a ONE-TIME preprocessing step: run it once after downloading the
competition data (scripts/download_data.py). It writes data/train_clean.parquet
and data/test_clean.parquet, which every other script (dataset-creation's
dataloader, scripts/mine_losses.py, scripts/train_scorer.py, ...) reads
instead of the raw files.

The cleaned copies are ROW-ALIGNED with the raw files: cleaning only modifies
the peak arrays (ms2_mzs / ms2_normalized_intensities) and adds the columns
num_peaks_clean, n_removed and drop_reason. No row is ever removed, so row
ids (spectrum ids) stay valid for both files. Spectra that end up with fewer
than --min-peaks peaks are flagged (drop_reason="too_few_peaks") but kept.

The cleaning itself lives in src/enveda_casmi26/cleaning.py (SpectrumCleaner,
same 7-step pipeline the Kaggle kernel applies on the fly): precursor cutoff,
relative intensity floor, deisotoping, mass-defect junk filter, top-K per m/z
window, global peak cap, sqrt intensity transform + renormalization.

Usage:
    uv run python dataset-cleaning/clean_data.py                     # full run, saves cleaned parquet
    uv run python dataset-cleaning/clean_data.py --sample 10000      # quick trial run
    uv run python dataset-cleaning/clean_data.py --floor 0.01 --top-k-per-window 0 --max-peaks 128
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from enveda_casmi26.cleaning import CleanConfig, SpectrumCleaner  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"


def build_config(args: argparse.Namespace) -> CleanConfig:
    return CleanConfig(
        relative_intensity_floor=args.floor,
        precursor_margin_da=args.precursor_margin,
        deisotope=not args.no_deisotope,
        isotope_tolerance_da=args.isotope_tol,
        top_k_per_window=args.top_k_per_window or None,
        window_da=args.window_da,
        max_peaks=args.max_peaks or None,
        min_peaks=args.min_peaks,
        intensity_transform=args.transform,
    )


def clean_file(cleaner: SpectrumCleaner, name: str, sample: int | None, save: bool) -> None:
    path = DATA_DIR / f"{name}.parquet"
    out_path = DATA_DIR / f"{name}_clean.parquet"
    print(f"\nLoading {name} ...")
    df = pd.read_parquet(path)
    n_raw = len(df)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=42).reset_index(drop=True)

    t0 = time.time()
    cleaned = cleaner.clean_dataframe(df)
    print(f"  cleaning took {time.time() - t0:.1f}s")
    cleaner.summarize(cleaned, name)
    assert len(cleaned) == len(df), "cleaning must not drop rows"

    if save:
        if sample and n_raw > sample:
            print("  NOTE: --sample given, NOT saving a partial train_clean.parquet "
                  "(it must stay row-aligned with train.parquet); use --skip-train "
                  "or a full run to save")
        else:
            cleaned.to_parquet(out_path, index=False)
            size_mb = out_path.stat().st_size / 1e6
            print(f"  saved {out_path.name} ({size_mb:.0f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--floor", type=float, default=0.001, help="relative intensity floor (0 disables)")
    parser.add_argument("--precursor-margin", type=float, default=1.5, help="Da above precursor to keep")
    parser.add_argument("--no-deisotope", action="store_true")
    parser.add_argument("--isotope-tol", type=float, default=0.01)
    parser.add_argument("--top-k-per-window", type=int, default=6, help="0 disables window selection")
    parser.add_argument("--window-da", type=float, default=50.0)
    parser.add_argument("--max-peaks", type=int, default=128, help="global cap, 0 disables")
    parser.add_argument("--min-peaks", type=int, default=5, help="flag threshold (rows are kept)")
    parser.add_argument("--transform", choices=["sqrt", "log", "none"], default="sqrt")
    parser.add_argument("--sample", type=int, default=None, help="only clean a sample of train (never saved)")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    config = build_config(args)
    print(f"CleanConfig: {config}")

    cleaner = SpectrumCleaner(config)
    if not args.skip_test:
        clean_file(cleaner, "test", None, not args.no_save)
    if not args.skip_train:
        clean_file(cleaner, "train", args.sample, not args.no_save)


if __name__ == "__main__":
    main()
