# spectra-viz

A small native GUI for browsing the Enveda CASMI26 training data: it indexes
`data/train.parquet` (2.5M MS/MS spectra, ~278k unique molecules), lets you
search and filter molecules, select one or more of their spectra, and shows the
peak plots (overlay or mirror view) next to 2D structure depictions rendered
from the SMILES.

Built with [egui](https://github.com/emilk/egui)/eframe,
[parquet-arrow](https://crates.io/crates/parquet), and
[cosmolkit](https://crates.io/crates/cosmolkit) for the structure depictions.

## Requirements

- Rust 1.82 or newer (edition 2021; the code uses `Option::is_none_or`)
- macOS, Linux, or Windows with a GPU/OpenGL-capable desktop session
- The competition data at `data/train.parquet` (see the repository README for
  how to download it)

## Building

```bash
cd spectra-viz
cargo build --release
```

Always use `--release`. The first launch builds a compact cache by scanning the
full parquet file, which takes about a minute in release mode and is painfully
slow in debug builds. Normal startup after that is well under a second.

## Running

The binary auto-detects the data file by checking `data/train.parquet`,
`../data/train.parquet`, and `../../data/train.parquet` relative to the working
directory, so you can start it from the repository root or from `spectra-viz/`:

```bash
# from the repository root
./spectra-viz/target/release/spectra-viz

# or from spectra-viz/
cargo run --release
```

The window opens at 1440x920 (minimum 960x620).

### First run: the cache

Random access into a 2.8 GB parquet file costs seconds per row, so the app
builds a compact cache on first launch and shows build progress in the window:

- Path: `<data-dir>/.spectra-viz/train-k32.svz` (~718 MB for the full training
  set)
- Content: the top 32 peaks per spectrum by intensity (sorted by m/z), plus
  per-spectrum metadata (precursor m/z, adduct, collision energy, library,
  ionization, peak count) and per-molecule strings (SMILES, formula, InChIKey)
- Rebuild with `--rebuild` or `--cache-only`; tune the peak count with
  `--max-peaks N` (the cache filename reflects it)

## Command-line reference

```
--data <PATH>        train.parquet (default: auto-detect data/train.parquet)
--cache <PATH>       cache file (default: <data-dir>/.spectra-viz/<stem>-k<K>.svz)
--max-peaks <N>      peaks stored per spectrum (default: 32)
--rebuild            rebuild the cache even if it exists
--cache-only         build the cache and exit (no GUI)
--check-smiles       parse every cached SMILES and report failures
--render-smiles <S>  render a SMILES depiction to --out (SVG or PNG)
--render-molecule <N> render cached molecule N to --out (SVG or PNG)
--out <PATH>         output path for depiction rendering (default: molecule.svg)
-h, --help           show this help
```

Examples:

```bash
# rebuild the cache from the terminal and watch progress
./spectra-viz/target/release/spectra-viz --cache-only --rebuild

# headless depiction of a single molecule (900x650 SVG or PNG)
./spectra-viz/target/release/spectra-viz --render-smiles "Cn1cnc2c1c(=O)n(C)c(=O)n2C" --out caffeine.svg

# render the 1000th molecule of the cached catalog
./spectra-viz/target/release/spectra-viz --render-molecule 1000 --out mol1000.png

# sanity-check SMILES parsing/depiction across the cached molecules
./spectra-viz/target/release/spectra-viz --check-smiles
```

`--check-smiles` samples every `N/20000`-th molecule (about 20k for the full
training set), parses each SMILES, renders up to 2000 depictions, and prints
the failure count. On the current dataset it reports zero parse and zero
render failures.

## Using the GUI

**Left panel - filters and molecule list**

- *Search*: case-insensitive, split on whitespace; every token must match the
  SMILES, molecular formula, or InChIKey
- *Library / Adduct / Ionization* dropdowns filter by source library, adduct,
  and positive/negative mode
- *m/z range* filters molecules by precursor m/z (drag or type; 0-5000)
- *Sort*: precursor mass (default), formula, spectra count, or dataset order
- *Reset filters* restores the full list; the molecule count is shown next to it
- Click a molecule to activate it. The list is virtualized and scrolls to the
  active molecule when it changes

**Middle panel - spectrum table**

Lists every spectrum of the active molecule with adduct, collision energy
(both CE bounds if a range), precursor m/z, peak count, source library, and
dataset row. Click rows to select/deselect spectra for plotting (up to 12),
or use *Select all* / *Clear selection*. The dataset row number is the row
index into `train.parquet`.

**Bottom panel - peak plot**

- Overlays all selected spectra as stick plots; toggle *Mirror* to draw
  even-indexed spectra upwards and odd-indexed downwards for comparison
- *Peak labels* annotates the most intense peaks with their m/z
- Hover near a peak for its exact m/z and intensity
- The legend shows each spectrum's color, adduct/CE detail, and an `x` to
  remove it; *Clear* empties the selection
- Note: plots show the stored top-K peaks by intensity (`K` = `--max-peaks`,
  default 32), not the complete peak list; the table's *peaks* column shows
  the true peak count from the dataset

**Right panel - structures**

- One card per selected spectrum's molecule with the cosmolkit-rendered 2D
  depiction (RDKit-quality layout, 640x400) and the SMILES with a *copy*
  button
- Depictions are rendered on demand and kept in an LRU cache of 64 textures;
  unparseable SMILES show an error placeholder instead

The header shows the total molecule/spectrum counts of the current filter and
a dark/light theme toggle.

## Troubleshooting

- **"Training parquet not found" screen**: the working directory has no
  `data/train.parquet` nearby; start the app from the repository root or pass
  `--data <path>`.
- **Stale or corrupt cache**: delete `data/.spectra-viz/` or run with
  `--rebuild`.
- **Crashes on launch (bus error / illegal instruction)**: a shell profile
  that exports `DYLD_LIBRARY_PATH` (for example pointing at
  `/opt/homebrew/lib`) makes macOS substitute Homebrew builds of libraries
  (libpng, libjpeg, libtiff, ...) for the ones Apple's frameworks expect
  during startup, which corrupts the process. The app detects any `DYLD_*`
  variables and re-executes itself with them stripped, printing a one-line
  notice; you should not see crashes even with a polluted environment.
  Removing `DYLD_LIBRARY_PATH` from your shell profile is still recommended —
  nothing in a normal Homebrew setup needs it, and it can break other GUI apps
  the same way. lldb/debugger launches are unaffected because debugserver
  strips these variables itself.
- **Diagnostics**: on every run the app writes startup breadcrumbs, an
  environment summary, and resource limits to
  `spectra-viz-diagnostics.log` next to the binary (the same lines are echoed
  to the terminal). If the process dies without a clean shutdown, a watchdog
  child copies macOS's own crash report to `spectra-viz-crash-report.txt`, and
  Rust panics are written to `spectra-viz-panic.log`. Include these files in
  bug reports.

## Implementation notes

- `src/main.rs` - CLI, app state, filters, panels, spectrum table
- `src/cache.rs` - parquet ingestion, cache writer/reader, `PeakStore`
  (per-spectrum peak loads with an in-memory LRU)
- `src/plot.rs` - stick plot with hover and labels
- `src/structure.rs` - SMILES to PNG (cosmolkit) to egui texture cache

The cache file starts with the magic `ENVSVZ01` and a version word, then
length-prefixed strings (adducts, libraries, per-molecule SMILES/formula/
InChIKey), one 32-byte record per spectrum, and a fixed-stride block of peak
pairs (8 bytes each) after it. Peak loads are plain pread-style reads, so the
GUI never holds the full 2.5M peak lists in memory. Neither the cache nor
`target/` is committed; see the repository `.gitignore`.
