"""Deprecated alias: the cleaning entry point moved to dataset-cleaning/clean_data.py.

All arguments are forwarded unchanged; see that script's docstring.
"""

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "dataset-cleaning" / "clean_data.py"),
    run_name="__main__",
)
