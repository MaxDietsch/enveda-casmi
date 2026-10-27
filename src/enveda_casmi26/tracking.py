"""MLflow experiment tracking helpers shared by the training and evaluation scripts.

Runs are logged to a local SQLite store by default, so no server is needed:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

For a remote server, create a gitignored ``mlflow.env`` in the repo root (or
pass ``--mlflow-config``) with MLflow's standard variables::

    MLFLOW_TRACKING_URI=https://mlflow.example.com
    MLFLOW_TRACKING_TOKEN=...

Environment variables take precedence over the file and explicit CLI flags take
precedence over both. ``MLFLOW_TRACKING_USERNAME``/``MLFLOW_TRACKING_PASSWORD``
are supported for basic auth; S3 artifact stores use boto3 and the standard
``AWS_*`` variables. Pass ``--no-mlflow`` to disable tracking for a run. If
MLflow is not installed or cannot be initialized, the helpers degrade to a
no-op so training still runs.
"""

from __future__ import annotations

import json
import math
import numbers
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = "enveda-casmi26"
DEFAULT_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "mlflow.env"
CONFIG_PREFIXES = ("MLFLOW_", "AWS_")


def add_arguments(parser) -> None:
    group = parser.add_argument_group("mlflow tracking")
    group.add_argument("--mlflow-tracking-uri", default=None,
                       help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or local mlflow.db)")
    group.add_argument("--mlflow-tracking-token", default=None,
                       help="tracking server API token (prefer MLFLOW_TRACKING_TOKEN or mlflow.env)")
    group.add_argument("--mlflow-tracking-username", default=None,
                       help="tracking server username for basic auth")
    group.add_argument("--mlflow-tracking-password", default=None,
                       help="tracking server password for basic auth")
    group.add_argument("--mlflow-config", type=Path, default=None,
                       help=f"key=value file with MLFLOW_* settings (default: {DEFAULT_CONFIG_PATH.name})")
    group.add_argument("--mlflow-experiment", default=None,
                       help="MLflow experiment name (default: $MLFLOW_EXPERIMENT_NAME "
                            f"or {DEFAULT_EXPERIMENT})")
    group.add_argument("--mlflow-run-name", default=None, help="MLflow run name")
    group.add_argument("--no-mlflow", action="store_true", help="disable MLflow tracking")


def _read_config(path: Path) -> dict[str, str]:
    """Read ``KEY=value`` lines, ignoring comments, blanks, and unrelated keys."""
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key.startswith(CONFIG_PREFIXES):
            values[key] = value
    return values


def _apply_config(args) -> None:
    path = getattr(args, "mlflow_config", None)
    explicit = path is not None
    path = Path(path) if explicit else DEFAULT_CONFIG_PATH
    if not path.exists():
        if explicit:
            print(f"[mlflow] config file not found: {path}")
        return
    try:
        values = _read_config(path)
    except OSError as exc:
        print(f"[mlflow] cannot read config file {path}: {exc}")
        return
    for key, value in values.items():
        os.environ.setdefault(key, value)
    print(f"[mlflow] loaded config from {path}")


def _apply_overrides(args) -> None:
    for attr, key in (("mlflow_tracking_token", "MLFLOW_TRACKING_TOKEN"),
                      ("mlflow_tracking_username", "MLFLOW_TRACKING_USERNAME"),
                      ("mlflow_tracking_password", "MLFLOW_TRACKING_PASSWORD")):
        value = getattr(args, attr, None)
        if value:
            os.environ[key] = value
    uri = getattr(args, "mlflow_tracking_uri", None)
    if uri:
        os.environ["MLFLOW_TRACKING_URI"] = uri


def _public_uri(uri: str) -> str:
    """Hide credentials embedded in the URI before printing it."""
    try:
        parts = urlsplit(uri)
    except ValueError:
        return uri
    if parts.username is None and parts.password is None:
        return uri
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))


def public_args(args) -> dict[str, Any]:
    """Script arguments safe to log or persist (no MLflow secrets)."""
    return {k: v for k, v in vars(args).items()
            if k != "no_mlflow" and not k.startswith("mlflow_")}


def _clean(name: str, allowed: str) -> str:
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-") | set(allowed)
    cleaned = "".join(c if c in keep else "_" for c in name)
    return cleaned.strip("_/ ") or "_"


def _param_name(name: str) -> str:
    return _clean(name, "")


def _metric_name(name: str) -> str:
    return _clean(name, "/")


def _param_value(value: Any) -> str | int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else str(number)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def _flatten_params(params: Mapping[str, Any], prefix: str = "") -> dict[str, str | int | float | bool]:
    flat: dict[str, str | int | float | bool] = {}
    for key, value in params.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_params(value, name))
        elif isinstance(value, (list, tuple, set)):
            flat[_param_name(name)] = json.dumps(list(value), default=str)
        else:
            flat[_param_name(name)] = _param_value(value)
    return flat


def _flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten_metrics(value, name))
        elif isinstance(value, bool):
            flat[_metric_name(name)] = float(value)
        elif isinstance(value, numbers.Real):
            number = float(value)
            if math.isfinite(number):
                flat[_metric_name(name)] = number
    return flat


class Tracker:
    """Thin wrapper around the active MLflow run; a no-op when disabled."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        import mlflow

        mlflow.log_params(_flatten_params(params))

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if not self.enabled:
            return
        flat = _flatten_metrics(metrics)
        if not flat:
            return
        import mlflow

        mlflow.log_metrics(flat, step=step)

    def log_tags(self, tags: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        import mlflow

        mlflow.set_tags({_param_name(str(k)): str(v) for k, v in tags.items()})

    def log_artifact(self, path: str | os.PathLike, artifact_path: str | None = None) -> None:
        if not self.enabled:
            return
        path = Path(path)
        if not path.exists():
            return
        import mlflow

        mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_dict(self, payload: Mapping[str, Any], artifact_file: str) -> None:
        if not self.enabled:
            return
        import mlflow

        mlflow.log_dict(dict(payload), artifact_file)

    def finish(self, status: str = "FINISHED") -> None:
        if not self.enabled:
            return
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.end_run(status=status)
        self.enabled = False


def _start(args, run_name: str | None, tags: Mapping[str, Any] | None) -> Tracker:
    if getattr(args, "no_mlflow", False):
        print("[mlflow] tracking disabled (--no-mlflow)")
        return Tracker()
    _apply_config(args)
    _apply_overrides(args)
    try:
        import mlflow

        uri = os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
        os.environ["MLFLOW_TRACKING_URI"] = uri
        experiment = (getattr(args, "mlflow_experiment", None)
                      or os.environ.get("MLFLOW_EXPERIMENT_NAME")
                      or DEFAULT_EXPERIMENT)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        run = mlflow.start_run(
            run_name=getattr(args, "mlflow_run_name", None) or run_name,
            tags={_param_name(str(k)): str(v) for k, v in (tags or {}).items()},
        )
    except Exception as exc:
        print(f"[mlflow] tracking disabled: {exc}")
        return Tracker()

    print(f"[mlflow] run {run.info.run_id} -> {_public_uri(uri)} (experiment: {experiment})")
    tracker = Tracker(enabled=True)
    try:
        tracker.log_params(public_args(args))
    except Exception as exc:
        print(f"[mlflow] failed to log run parameters: {exc}")
    return tracker


@contextmanager
def session(args, run_name: str | None = None, tags: Mapping[str, Any] | None = None):
    tracker = _start(args, run_name, tags)
    try:
        yield tracker
    except BaseException:
        tracker.finish(status="FAILED")
        raise
    else:
        tracker.finish()


def log_run(args, params: Mapping[str, Any] | None = None,
            metrics: Mapping[str, Any] | None = None,
            artifacts: Iterable[str | os.PathLike] = (),
            run_name: str | None = None,
            tags: Mapping[str, Any] | None = None) -> None:
    """Log a completed experiment in one call (for scripts that run once)."""
    with session(args, run_name=run_name, tags=tags) as tracker:
        if params:
            tracker.log_params(params)
        if metrics:
            tracker.log_metrics(metrics)
        for artifact in artifacts:
            tracker.log_artifact(artifact)
