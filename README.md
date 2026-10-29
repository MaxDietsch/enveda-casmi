# enveda-casmi26

Solution code for the Enveda CASMI26 molecule identification from mass spectra competition.

## Setup

```bash
uv sync
```

## Downloading the data

The competition data is not included in this repository (it is too large).
Download it with the helper script, which uses [kagglehub](https://pypi.org/project/kagglehub/)
to fetch the Kaggle competition dataset `enveda-CASMI26-molecule-id-mass-spectra`
into the `data/` folder:

```bash
python scripts/download_data.py
```

You will need Kaggle API credentials (`~/.kaggle/kaggle.json`) for the download to work.
Other derived datasets (`coconut_pool.parquet`, FP caches, etc.) are produced by the
scripts in `scripts/` (e.g. `build_coconut_pool.py`).

## Spectra visualizer

`spectra-viz/` contains a small Rust GUI for browsing the training data:
filter molecules by SMILES/formula/InChIKey and metadata, select spectra, and
view overlaid or mirrored peak plots next to 2D structure depictions. See
[spectra-viz/README.md](spectra-viz/README.md) for build and usage
instructions.

## Experiment tracking

Training and evaluation scripts log parameters, metrics, and model artifacts to
[MLflow](https://mlflow.org/). Runs are stored locally in `mlflow.db` by default:

```bash
uv run python scripts/train_denovo.py
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

For a shared/remote server, copy the template to the gitignored `mlflow.env`
and fill in your server URI and token:

```bash
cp mlflow.env.example mlflow.env
```

`MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` are supported for basic
auth. Environment variables override `mlflow.env`; explicit
`--mlflow-tracking-uri`/`--mlflow-tracking-token` flags override both. Use
`--mlflow-config PATH` for a file elsewhere, `--mlflow-experiment` and
`--mlflow-run-name` to organize runs, or `--no-mlflow` to disable tracking.
