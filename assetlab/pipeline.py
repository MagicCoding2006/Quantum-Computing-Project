"""Run snapshots and provenance, with recorded revisions to derived reports."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
import uuid

import numpy as np
import pandas as pd

from . import __version__
from .analysis import analyze
from .data import load_prices
from .report import write_report


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def code_provenance():
    root = Path(__file__).resolve().parent.parent
    files = [root / "asset_lab.py", root / "trade_lab.py", root / "ibm_quantum.py", root / "benchmark_asset_lab.py", root / "requirements.txt"]
    files += sorted((root / "assetlab").glob("*.py"))
    files += sorted((root / "assetlab").glob("*.html"))
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files if p.exists()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    packages = {}
    for name in ("numpy", "pandas", "scipy", "statsmodels", "yfinance", "qiskit", "qiskit-ibm-runtime"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {"version": __version__, "python": platform.python_version(),
            "packages": packages, "git_commit": commit, "git_dirty": dirty,
            "source_sha256": hashes}


def run_study(config, *, csv_path=None, cache_dir="data_cache/asset_lab", out="results_asset_lab", refresh=False):
    prices, provenance = load_prices(
        config["tickers"], config["start"], config["end"], demo=config["demo"],
        csv_path=csv_path, cache_dir=cache_dir, refresh=refresh, seed=config["seed"],
    )
    payload, returns = analyze(prices, provenance, config)
    return save_results(payload, prices, returns, out)


def save_results(payload, prices, returns, out="results_asset_lab"):
    """Export one study, including benchmark cases fitted to a frozen panel."""
    config, provenance = payload["config"], payload["provenance"]
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = Path(out).resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    payload.update(run_id=run_id, created_at_utc=now.isoformat())
    prices.to_csv(output / "prices.csv", index_label="date")
    returns.to_csv(output / "returns.csv", index_label="date")
    for period, matrices in payload["correlations"].items():
        for method, matrix in matrices.items():
            pd.DataFrame(matrix, index=payload["tickers"], columns=payload["tickers"]).to_csv(
                output / f"{period}_{method}.csv", index_label="asset")
    pd.DataFrame(payload["pairs"]).to_csv(output / "pairs.csv", index=False)
    pd.DataFrame(payload["rolling_latest"]).to_csv(output / "rolling_latest.csv", index=False)
    pd.DataFrame(payload["exact"]).to_csv(output / "exact_selections.csv", index=False)
    write_json(output / "problem.json", payload["problem"])
    write_json(output / "analysis.json", payload)
    write_report(output, json_safe(payload))
    manifest = {"schema_version": 1, "run_id": run_id, "created_at_utc": now.isoformat(),
                "config": config, "data": provenance, "code": code_provenance(),
                "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    return output, payload
