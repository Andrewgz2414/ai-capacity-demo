"""End-to-end checks (spec section 9): run_all.py and the notebook complete, outputs exist.

Both run against a temporary output directory (AICAP_OUTPUTS) seeded with the cached regime map
and Cox simulation, so the real outputs/ folder is not touched. Marked slow (about 30 s).
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
CACHES = ["regime_grid.npz", "cox_sim.npz"]
FIGS = ["fig0_newsvendor_foc", "fig1_reproduction", "fig2_regime_map", "fig3_cox_floor", "fig4_handoff"]

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tmp_outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp("outputs")
    (out / "cache").mkdir()
    for c in CACHES:
        src = ROOT / "outputs" / "cache" / c
        if not src.exists():
            pytest.skip(f"cache {c} missing: run scripts/run_all.py once first")
        shutil.copy(src, out / "cache" / c)
    return out


def _env(out):
    env = dict(os.environ, AICAP_OUTPUTS=str(out), MPLBACKEND="Agg")
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_run_all_completes(tmp_outputs):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_all.py")], env=_env(tmp_outputs),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((tmp_outputs / "summary.json").read_text())
    for part in ("partA", "partB", "partC", "partD", "sobol"):
        assert part in s
    for f in FIGS:
        assert (tmp_outputs / "figures" / f"{f}.png").stat().st_size > 50_000
        assert (tmp_outputs / "figures" / f"{f}.svg").exists()
    for f in ["dQ_samples_DY2028.csv", "dQ_samples_DY2030.csv", "energy_samples_2028.csv",
              "energy_samples_2030.csv", "handoff_summary.json", "README.md"]:
        assert (tmp_outputs / "handoff" / f).exists()
    assert "energy" in s["partD"] and "energy" in json.loads((tmp_outputs / "handoff" / "handoff_summary.json").read_text())
    import pandas as pd
    df = pd.read_csv(tmp_outputs / "handoff" / "dQ_samples_DY2030.csv", comment="#")
    assert list(df.columns) == ["draw_id", "dQ_GW", "P0_GW", "PT_GW", "efficiency_structure"]
    assert len(df) == 2 * s["partD"]["n_draws"]
    assert (abs(df["PT_GW"] - df["P0_GW"] - df["dQ_GW"]) < 1e-4 * (1 + df["PT_GW"].abs())).all()
    en = pd.read_csv(tmp_outputs / "handoff" / "energy_samples_2030.csv", comment="#")
    assert list(en.columns) == ["draw_id", "efficiency_structure", "E2025_TWh_global", "E2030_TWh_global",
                                "E2025_TWh_US", "E2030_TWh_US", "E2025_TWh_PJM", "E2030_TWh_PJM"]
    assert len(en) == len(df) and (en["draw_id"].values == df["draw_id"].values).all()


def test_notebook_executes_without_errors(tmp_outputs):
    out_dir = tmp_outputs / "nb"
    r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                        str(ROOT / "notebooks" / "demo.ipynb"), "--output-dir", str(out_dir),
                        "--ExecutePreprocessor.timeout=300"],
                       env=_env(tmp_outputs), capture_output=True, text=True, timeout=600, cwd=ROOT / "notebooks")
    assert r.returncode == 0, r.stderr[-3000:]
    nb = nbformat.read(out_dir / "demo.ipynb", as_version=4)
    # errors raised inside display formatters do not fail the cell, so scan the outputs too
    errors = [(i, o.get("ename")) for i, c in enumerate(nb.cells) if c.cell_type == "code"
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, errors
    n_images = sum("image/png" in o.get("data", {}) for c in nb.cells if c.cell_type == "code"
                   for o in c.get("outputs", []))
    assert n_images == 5
    assert any("Bridge:" in o.get("text", "") for c in nb.cells if c.cell_type == "code"
               for o in c.get("outputs", []))
