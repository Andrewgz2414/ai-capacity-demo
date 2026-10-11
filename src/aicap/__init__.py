"""AI load, parameter uncertainty and capacity sizing (APMA 4903 code demo)."""

from __future__ import annotations

import datetime as _dt
import platform as _platform
import subprocess as _sp
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"


def run_info() -> dict:
    """Timestamp, git commit (and whether the tree had uncommitted changes), versions."""
    import numpy
    import scipy

    def git(*args):
        try:
            return _sp.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # git missing or not a repo
            return ""

    return {
        "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "--short", "HEAD") or "unknown",
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": _platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }
