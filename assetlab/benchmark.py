"""Local scaling experiments with exact, greedy, and feasible-random baselines."""

from copy import deepcopy
from datetime import datetime, timezone
from math import comb
from pathlib import Path
import json
import time
import uuid

import numpy as np
import pandas as pd

from .analysis import portfolio_metrics
from .pipeline import save_results, write_json
from .quantum import simulate
from .report import refresh_report
from .selection import MAX_CANDIDATES, build_problem, solve_exact


def variance_of(indices, covariance):
    return float(covariance[np.ix_(indices, indices)].sum() / len(indices) ** 2)


def classical_baselines(problem, exact, *, shots, seed):
    covariance = np.asarray(problem["annual_covariance"])
    n, k = len(problem["tickers"]), problem["choose"]
    optimum = exact[0]["train_annual_variance"]
    optimal_masks = {r["mask"] for r in exact if np.isclose(
        r["train_annual_variance"], optimum, atol=1e-12, rtol=1e-8)}
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    # Equal measurement/sample count, with the cardinality constraint enforced.
    # This is a stronger baseline than random unrestricted bitstrings.
    indices = np.argsort(rng.random((shots, n)), axis=1)[:, :k]
    masks = (1 << indices).sum(axis=1)
    values = covariance[indices[:, :, None], indices[:, None, :]].sum(axis=(1, 2)) / k ** 2
    random_seconds = time.perf_counter() - started
    started = time.perf_counter()
    chosen = []
    evaluations = 0
    for _ in range(k):
        choices = [(variance_of(chosen + [i], covariance), i) for i in range(n) if i not in chosen]
        evaluations += len(choices)
        chosen.append(min(choices)[1])
    # Strictly improving one-for-one swaps; no exhaustive subset enumeration.
    while True:
        current = variance_of(chosen, covariance)
        best_value, best_set = current, None
        for old in chosen:
            for new in range(n):
                if new in chosen:
                    continue
                candidate = [i for i in chosen if i != old] + [new]
                value = variance_of(candidate, covariance)
                evaluations += 1
                if value < best_value - 1e-14:
                    best_value, best_set = value, candidate
        if best_set is None:
            break
        chosen = best_set
    greedy_seconds = time.perf_counter() - started
    greedy_variance = variance_of(chosen, covariance)

    def gap(value):
        return float(100 * max(0.0, value - optimum) / optimum) if optimum > 1e-15 else None

    return {
        "random_legal": {"shots": shots, "best_gap_percent": gap(float(values.min())),
                         "mean_gap_percent": gap(float(values.mean())),
                         "optimal_fraction": float(np.isin(masks, list(optimal_masks)).mean()),
                         "expected_optimal_probability": len(optimal_masks) / comb(n, k),
                         "elapsed_seconds": random_seconds},
        "greedy_swap": {"assets": [problem["tickers"][i] for i in sorted(chosen)],
                        "train_variance": greedy_variance, "best_gap_percent": gap(greedy_variance),
                        "objective_evaluations": evaluations, "elapsed_seconds": greedy_seconds},
    }


def run_benchmark(parent_path, *, sizes=(6, 10, 14), layers=(1, 2), seeds=(42, 43, 44), shots=4096):
    parent_path = Path(parent_path).resolve()
    original = json.loads((parent_path / "analysis.json").read_text(encoding="utf-8"))
    n_assets = len(original["tickers"])
    if not sizes or any(not 2 <= n <= min(MAX_CANDIDATES, n_assets) for n in sizes):
        raise ValueError(f"Sizes must be between 2 and {min(MAX_CANDIDATES, n_assets)}.")
    if not layers or any(p not in (1, 2, 3) for p in layers):
        raise ValueError("Use QAOA layer counts 1, 2, or 3.")
    if not seeds or any(not 0 <= s < 2 ** 32 for s in seeds) or not 1 <= shots <= 100000:
        raise ValueError("Use nonnegative 32-bit seeds and 1-100000 shots.")
    returns = pd.DataFrame(original["returns"], index=pd.to_datetime(original["dates"]), columns=original["tickers"])
    train, holdout = returns.iloc[:original["split"]["train_rows"]], returns.iloc[original["split"]["train_rows"]:]
    prices = pd.read_csv(parent_path / "prices.csv", index_col=0, parse_dates=True)
    now = datetime.now(timezone.utc)
    benchmark_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = parent_path / "benchmarks" / benchmark_id
    folder.mkdir(parents=True, exist_ok=False)
    rows = []
    for n in sizes:
        candidates = original["tickers"][:n]
        choose = n // 2
        problem = build_problem(train[candidates].cov().to_numpy() * 252, candidates, choose)
        started = time.perf_counter()
        exact = solve_exact(problem)
        exact_seconds = time.perf_counter() - started
        for p in layers:
            for seed in seeds:
                print(f"Local benchmark: {n} assets, choose {choose}, p={p}, seed={seed}; {comb(n, choose):,} legal portfolios", flush=True)
                quantum = simulate(problem, layers=p, shots=shots, seed=seed)
                baselines = classical_baselines(problem, exact, shots=shots, seed=seed)
                payload = deepcopy(original)
                payload.pop("run_id", None)
                payload.pop("created_at_utc", None)
                payload.update(problem=problem, exact=exact, quantum=quantum,
                               parent_run_id=original["run_id"], benchmark_id=benchmark_id)
                payload["config"].update(candidates=candidates, choose=choose, layers=p, shots=shots, seed=seed)
                payload["portfolios"] = {
                    "Exact selection": portfolio_metrics(holdout, exact[0]["assets"]),
                    "All candidates": portfolio_metrics(holdout, candidates),
                }
                if quantum["selected_assets"]:
                    payload["portfolios"]["QAOA sampled selection"] = portfolio_metrics(holdout, quantum["selected_assets"])
                payload["classical_baselines"] = baselines
                payload["limitations"].append("This benchmark reuses an existing holdout; further tuning needs a new final evaluation period.")
                case_path, _ = save_results(payload, prices, returns, folder / "cases")
                optimum = exact[0]["train_annual_variance"]
                lookup = {r["mask"]: r["train_annual_variance"] for r in exact}
                best_variance = lookup.get(quantum["best_feasible_mask"])
                row = {"assets": n, "choose": choose, "legal_portfolios": len(exact),
                       "basis_states": 1 << n, "layers": p, "seed": seed,
                       "shots": shots, "feasible_fraction": quantum["feasible_fraction"],
                       "optimal_fraction": quantum["optimal_fraction"],
                       "qaoa_best_gap_percent": float(100 * max(0.0, best_variance - optimum) / optimum)
                       if best_variance is not None and optimum > 1e-15 else None,
                       "simulation_seconds": quantum["elapsed_seconds"],
                       "exact_seconds": exact_seconds, "optimizer_evaluations": quantum["optimizer_evaluations"],
                       "logical_pair_gates": p * sum(abs(v) > 1e-14 for line in problem["ising_J_upper"] for v in line),
                       "random_legal": baselines["random_legal"], "greedy_swap": baselines["greedy_swap"],
                       "case_run": case_path.relative_to(parent_path).as_posix(),
                       "hardware_executed": False}
                rows.append(row)
                write_json(folder / "results.json", {"benchmark_id": benchmark_id, "parent_run_id": original["run_id"],
                           "created_at_utc": now.isoformat(), "rows": rows, "status": "running"})
    result = {"benchmark_id": benchmark_id, "parent_run_id": original["run_id"],
              "created_at_utc": now.isoformat(), "rows": rows, "status": "complete",
              "scope": "Local ideal simulation; exact enumeration; feasible-uniform random samples; greedy plus swaps",
              "selection_policy": "First n assets in the original fixed universe; k=floor(n/2); training covariance only",
              "hardware_executed": False, "results_path": (folder / "results.json").relative_to(parent_path).as_posix()}
    write_json(folder / "results.json", result)
    pd.json_normalize(rows, sep="_").to_csv(folder / "results.csv", index=False)
    write_json(parent_path / "benchmarks" / "latest.json", result)
    refresh_report(parent_path)
    return folder, result
