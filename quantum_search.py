#!/usr/bin/env python3
"""Offline Grover feasibility audit: market-condition rules, real oracle costs, no QPU job."""

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import time
import uuid

import numpy as np
import pandas as pd

from assetlab.pipeline import code_provenance, write_json
from assetlab.search import (classical_reference, compiled_resources, coverage_reference, grover_circuit,
                            ideal_search, phase_oracle, scaling_table, search_problem, with_encoding)


def price_episodes(path, *, training_end, target="SPY", market="QQQ", episodes=8,
                   horizon=10, cost_bps=10., unit_bps=25., feature_assets=(), conditions=None):
    """First chronological, nonoverlapping episodes; no threshold/return selection."""
    if not 1 <= episodes <= 256 or not 1 <= horizon <= 252:
        raise ValueError("Use 1-256 episodes and a 1-252 session holding period.")
    if not np.isfinite([cost_bps, unit_bps]).all() or not 0 <= cost_bps < 10000 or unit_bps <= 0:
        raise ValueError("Use finite nonnegative costs below 10,000 bps and a positive score unit.")
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    prices.index = pd.DatetimeIndex(prices.index)
    if prices.index.hasnans or prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be valid, unique, and increasing.")
    cutoff = pd.Timestamp(training_end)
    if pd.isna(cutoff):
        raise ValueError("Training cutoff must be a valid date.")
    feature_assets = list(dict.fromkeys(feature_assets))
    if conditions is not None:
        remaining = conditions - 6
        needed = []
        for asset in feature_assets:
            if remaining <= 0:
                break
            if asset != target:
                needed.append(asset)
                remaining -= 4 if asset == market else 5
        feature_assets = needed
    prices = prices.loc[:cutoff, list(dict.fromkeys([target, market, *feature_assets]))]
    if len(prices) < 22 + horizon or not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
        raise ValueError("Need enough complete, finite, positive training prices for target and market.")
    returns = prices[target].pct_change(fill_method=None)
    observed = pd.DataFrame({
        "return_1_positive": prices[target].pct_change(1, fill_method=None) > 0,
        "return_5_positive": prices[target].pct_change(5, fill_method=None) > 0,
        "return_20_positive": prices[target].pct_change(20, fill_method=None) > 0,
        "above_mean_20": prices[target] > prices[target].rolling(20).mean(),
        "vol_5_above_20": returns.rolling(5).std() > returns.rolling(20).std(),
        "market_return_5_positive": prices[market].pct_change(5, fill_method=None) > 0,
    })
    for asset in feature_assets:
        if asset == target:
            continue
        asset_returns = prices[asset].pct_change(fill_method=None)
        for window in (1, 5, 20):
            if asset == market and window == 5:
                continue  # Already present as market_return_5_positive.
            observed[f"{asset}_return_{window}_positive"] = prices[asset].pct_change(window, fill_method=None) > 0
        observed[f"{asset}_above_mean_20"] = prices[asset] > prices[asset].rolling(20).mean()
        observed[f"{asset}_vol_5_above_20"] = asset_returns.rolling(5).std() > asset_returns.rolling(20).std()
    if conditions is not None:
        if not 1 <= conditions <= len(observed.columns):
            raise ValueError(f"Requested condition count exceeds the {len(observed.columns)} declared indicators.")
        observed = observed.iloc[:, :conditions]
    rows = []
    fee = cost_bps / 10000
    for signal in range(20, len(prices) - horizon - 1, horizon + 1):
        entry, exit_ = signal + 1, signal + 1 + horizon
        net_bps = (prices[target].iloc[exit_] / prices[target].iloc[entry] * (1 - fee) / (1 + fee) - 1) * 10000
        rows.append({"signal_date": prices.index[signal].date().isoformat(),
                     "entry_date": prices.index[entry].date().isoformat(),
                     "exit_date": prices.index[exit_].date().isoformat(),
                     **{name: int(value) for name, value in observed.iloc[signal].items()},
                     "net_return_bps": float(net_bps), "score_units": int(np.rint(net_bps / unit_bps))})
        if len(rows) == episodes:
            break
    if len(rows) < episodes:
        raise ValueError("Not enough training data for the requested nonoverlapping episodes.")
    frame = pd.DataFrame(rows)
    return frame, list(observed.columns)


def resource_preflight(problem):
    return (problem["search_qubits"] <= 20 and len(problem["episode_masks"]) <= 64
            and problem["score_bits"] <= 16)


def save_manifest(folder):
    write_json(folder / "manifest.json", {"code": code_provenance(), "files_sha256": {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "manifest.json"}})


def run_benchmark(args):
    if not args.prices or not args.training_end:
        raise ValueError("Scaling benchmarks require --prices and an explicit --training-end.")
    condition_sizes = sorted(set(args.condition_sizes))
    episode_sizes = sorted(set(args.episode_sizes))
    if not condition_sizes or not episode_sizes or not 1 <= min(condition_sizes) <= max(condition_sizes) <= 64:
        raise ValueError("Use condition sizes from 1 to 64.")
    if not 1 <= min(episode_sizes) <= max(episode_sizes) <= 256:
        raise ValueError("Use episode sizes from 1 to 256.")
    if not 1 <= args.min_support <= min(episode_sizes):
        raise ValueError("Minimum support must fit within the smallest episode case.")
    assets = args.feature_assets or [args.market, "IWM", "TLT", "IEF", "GLD"]
    frame, names = price_episodes(args.prices, training_end=args.training_end, target=args.target,
        market=args.market, episodes=max(episode_sizes), horizon=args.horizon,
        cost_bps=args.cost_bps, unit_bps=args.unit_bps, feature_assets=assets, conditions=max(condition_sizes))
    now = datetime.now(timezone.utc)
    folder = args.out.resolve() / "benchmarks" / (now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    rows, cases = [], []
    for episodes in episode_sizes:
        for conditions in condition_sizes:
            case_names = names[:conditions]
            data = frame.iloc[:episodes].copy()
            data = data[[c for c in data if c not in names or c in case_names]]
            problem = search_problem(data[case_names].to_numpy(), data.score_units.to_numpy(),
                        names=case_names, min_support=args.min_support, min_score=args.min_score)
            reference = classical_reference(problem) if conditions <= 20 else None
            coverage = coverage_reference(problem)
            results = {}
            print(f"Benchmark: {conditions} conditions, {episodes} historical episodes", flush=True)
            for implementation in ("baseline", "optimized"):
                encoded = with_encoding(problem, implementation)
                start = time.perf_counter()
                resources = compiled_resources(phase_oracle(encoded)) if resource_preflight(encoded) else None
                results[implementation] = {"encoding": encoded["encoding"], "oracle_resources": resources,
                                           "construction_and_compilation_seconds": time.perf_counter() - start,
                                           "predicate_sha256": encoded["predicate_sha256"]}
            baseline, optimized = (results[k]["oracle_resources"] for k in ("baseline", "optimized"))
            row = {"conditions": conditions, "episodes": episodes, "candidate_rules": str(1 << conditions),
                   "predicate_sha256": problem["predicate_sha256"],
                   "passing_rules": reference["marked_count"] if reference is not None else None,
                   "classical_scoring_seconds": reference["elapsed_seconds"] if reference is not None else None,
                   "distinct_firing_sets": coverage["distinct_firing_sets"],
                   "coverage_complete": coverage["complete"], "coverage_search_seconds": coverage["elapsed_seconds"],
                   "baseline_qubits": results["baseline"]["encoding"]["total_qubits"],
                   "optimized_qubits": results["optimized"]["encoding"]["total_qubits"],
                   "baseline_cx": baseline["cx_gates"] if baseline else None,
                   "optimized_cx": optimized["cx_gates"] if optimized else None,
                   "baseline_depth": baseline["depth"] if baseline else None,
                   "optimized_depth": optimized["depth"] if optimized else None,
                   "cx_reduction_fraction": 1 - optimized["cx_gates"] / baseline["cx_gates"] if baseline and optimized and baseline["cx_gates"] else None,
                   "hardware_executed": False}
            case = folder / f"conditions_{conditions}_episodes_{episodes}"
            case.mkdir()
            data.to_csv(case / "episodes.csv", index=False)
            write_json(case / "problem.json", problem)
            write_json(case / "audit.json", {"comparison": row, "implementations": results,
                "classical_coverage_search": coverage,
                "ideal_success_diagnostic": [ideal_search(conditions, reference["marked_count"], k) for k in args.iterations] if reference is not None else None})
            rows.append(row)
            cases.append({"path": case.name, **row})
            if baseline and optimized:
                print(f"  qubits {baseline['qubits']} -> {optimized['qubits']}; CX {baseline['cx_gates']:,} -> {optimized['cx_gates']:,}; depth {baseline['depth']:,} -> {optimized['depth']:,}", flush=True)
            else:
                print("  Register plans saved; reference compilation skipped by preflight bounds.", flush=True)
    pd.DataFrame(rows).to_csv(folder / "comparison.csv", index=False)
    write_json(folder / "summary.json", {"schema_version": 1, "kind": "grover_encoding_scaling_benchmark",
        "created_at_utc": now.isoformat(), "training_end": args.training_end, "target": args.target,
        "market": args.market, "feature_assets": assets, "condition_order": names,
        "source_sha256": hashlib.sha256(args.prices.read_bytes()).hexdigest(),
        "horizon_sessions": args.horizon, "one_way_cost_bps": args.cost_bps, "score_unit_bps": args.unit_bps,
        "min_support": args.min_support, "min_score": args.min_score, "cases": cases,
        "hardware_executed": False,
        "interpretation": "Same predicate within each baseline/optimized pair. All-to-all reference compilation, not routed QPU performance or demonstrated quantum advantage. Classical timings are single exploratory observations, not calibrated speed comparisons. Conditions follow a fixed definition order, not profit ranking; episodes are chronological training prefixes."})
    save_manifest(folder)
    print(f"Saved scaling benchmark: {folder}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="synthetic observations, not market results")
    source.add_argument("--episodes-csv", type=Path, help="binary condition columns plus integer score_units")
    source.add_argument("--prices", type=Path, help="saved adjusted-close CSV; no downloads")
    parser.add_argument("--training-end", help="inclusive cutoff required with --prices")
    parser.add_argument("--target", default="SPY")
    parser.add_argument("--market", default="QQQ")
    parser.add_argument("--feature-assets", nargs="*", default=[], help="extra assets supplying observed conditions, in declared order")
    parser.add_argument("--conditions", type=int, help="use the first N indicators in the declared feature order")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--cost-bps", type=float, default=10.)
    parser.add_argument("--unit-bps", type=float, default=25., help="round each price-derived net return to this score unit")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-score", type=int, default=1, help="minimum summed integer score, fixed before search")
    parser.add_argument("--iterations", type=int, nargs="+", default=[0, 1, 2, 3], help="ideal success diagnostics; circuit export uses --export-iterations")
    parser.add_argument("--export-iterations", type=int, default=1)
    parser.add_argument("--implementation", choices=["baseline", "optimized"], default="optimized")
    parser.add_argument("--compare-baseline", action="store_true", help="also compile the original carry-based oracle for comparison")
    parser.add_argument("--benchmark", action="store_true", help="compare both encodings on a saved-market-data scaling grid")
    parser.add_argument("--condition-sizes", nargs="+", type=int, default=[6, 10, 14])
    parser.add_argument("--episode-sizes", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--out", type=Path, default=Path("results_asset_lab/search"))
    args = parser.parse_args(argv)
    try:
        if any(k < 0 or k > 1000000 for k in args.iterations) or not 0 <= args.export_iterations <= 32:
            raise ValueError("Use diagnostic iterations 0-1,000,000 and exported iterations 0-32.")
        if args.benchmark:
            return run_benchmark(args)
        metadata = {"mode": "synthetic" if args.demo else "training_only" if args.prices else "uploaded_unverified_episodes",
                    "price_settings_apply_only_to": "--prices", "score_units": "arbitrary integers"}
        if args.demo:
            rng = np.random.default_rng(42)
            frame = pd.DataFrame(rng.integers(0, 2, (8, 4)), columns=[f"condition_{i}" for i in range(4)])
            frame["score_units"] = [2, -1, 1, -2, 3, -1, 2, -1]
            names = [c for c in frame if c != "score_units"]
        elif args.prices:
            if not args.training_end:
                raise ValueError("Supply --training-end to keep this audit within declared training data.")
            frame, names = price_episodes(args.prices, training_end=args.training_end, target=args.target,
                       market=args.market, episodes=args.episodes, horizon=args.horizon,
                       cost_bps=args.cost_bps, unit_bps=args.unit_bps,
                       feature_assets=args.feature_assets, conditions=args.conditions)
            metadata.update(training_end=args.training_end, target=args.target, market=args.market,
                            horizon_sessions=args.horizon, one_way_cost_bps=args.cost_bps,
                            score_unit_bps=args.unit_bps,
                            score_units=f"{args.unit_bps:g} basis points per unit",
                            feature_assets=args.feature_assets, condition_order=names,
                            rounding_error_bps_per_episode_at_most=args.unit_bps / 2)
        else:
            frame = pd.read_csv(args.episodes_csv)
            names = [c for c in frame if c not in {"score_units", "signal_date", "entry_date", "exit_date", "net_return_bps"}]
            if "score_units" not in frame:
                raise ValueError("Episode CSV needs binary condition columns and score_units.")
            metadata["warning"] = "Caller must supply training-only episodes, valid costs, and nonoverlapping execution intervals."
        input_path = args.prices or args.episodes_csv
        if input_path:
            metadata.update(input_file=str(input_path.resolve()),
                            input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest())
        problem = search_problem(frame[names].to_numpy(), frame["score_units"].to_numpy(),
                                 min_support=args.min_support, min_score=args.min_score, names=names,
                                 implementation=args.implementation)
        # Resource preflight stays bounded; larger input spaces still get a plan.
        compile_allowed = resource_preflight(problem)
        reference = classical_reference(problem) if problem["search_qubits"] <= 20 else None
        oracle = phase_oracle(problem) if compile_allowed else None
        oracle_resources = compiled_resources(oracle) if oracle is not None else None
        export_allowed = compile_allowed and args.export_iterations * oracle_resources["cx_gates"] <= 200000
        circuit = grover_circuit(problem, args.export_iterations) if export_allowed else None
        baseline = with_encoding(problem, "baseline")
        baseline_resources = compiled_resources(phase_oracle(baseline)) if args.compare_baseline and resource_preflight(baseline) else None
        now = datetime.now(timezone.utc)
        report = {
            "schema_version": 1, "kind": "offline_grover_feasibility", "created_at_utc": now.isoformat(),
            "source": metadata, "problem": problem,
            "classical_reference": {k: v for k, v in reference.items() if k not in {"support", "score", "marked"}} if reference is not None else None,
            "classical_coverage_search": coverage_reference(problem),
            "ideal_success_diagnostic": [ideal_search(problem["search_qubits"], reference["marked_count"], k) for k in args.iterations] if reference is not None else None,
            "single_solution_scaling_not_this_market_instance": scaling_table(),
            "oracle_resources": oracle_resources,
            "baseline_oracle_resources": baseline_resources,
            "exported_circuit_resources": compiled_resources(circuit) if circuit is not None else None,
            "exported_iterations": args.export_iterations,
            "circuit_export_status": "exported_offline" if export_allowed else "skipped_preflight_limit: 20 conditions, 64 episodes, 16 score bits; exported oracle repetitions limited to 200,000 estimated CX gates",
            "hardware_executed": False,
            "interpretation": [
                "Grover gives a quadratic black-box query improvement, not a guaranteed market or wall-clock advantage.",
                "Marked counts and success curves come from classical exhaustive diagnostics, not QPU measurements.",
                "The same data admits a structured classical superset-sum baseline; brute force is not the only comparison.",
                "The oracle includes training data checks, signed return arithmetic, comparisons, and uncomputation.",
                "No QRAM assumption, hidden precomputed solution table, backend fidelity prediction, or broker orders.",
                "A threshold chosen using these results becomes another research trial; it is not an untouched test.",
            ],
        }
        folder = args.out.resolve() / (now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
        folder.mkdir(parents=True, exist_ok=False)
        frame.to_csv(folder / "episodes.csv", index=False)
        write_json(folder / "problem.json", problem)
        write_json(folder / "audit.json", report)
        if reference is not None:
            pd.DataFrame({"mask": np.arange(1 << problem["search_qubits"]), "support": reference["support"],
                          "score_units": reference["score"], "marked": reference["marked"]}).to_csv(folder / "classical_rules.csv", index=False)
        if circuit is not None:
            from qiskit import qasm3, qpy
            with (folder / "grover.qpy").open("wb") as stream:
                qpy.dump(circuit, stream)
            (folder / "grover.qasm").write_text(qasm3.dumps(circuit), encoding="utf-8")
        save_manifest(folder)
        print(f"Offline audit: {folder}")
        print(f"Oracle implementation: {args.implementation}")
        print(f"{problem['search_qubits']} search qubits + {problem['workspace_qubits']} workspace = {problem['total_qubits']} total qubits.")
        if reference is not None:
            print(f"{reference['marked_count']} / {1 << problem['search_qubits']} rules pass; structured classical scoring: {reference['elapsed_seconds']:.6f}s.")
        coverage = report['classical_coverage_search']
        print(f"Classical coverage search: {coverage['distinct_firing_sets']} distinct firing sets, complete={coverage['complete']}.")
        if oracle is not None:
            r = report["oracle_resources"]
            print(f"One oracle: {r['cx_gates']:,} CX gates, depth {r['depth']:,} in all-to-all reference compilation.")
        if baseline_resources is not None:
            print(f"Baseline oracle: {baseline_resources['qubits']} qubits, {baseline_resources['cx_gates']:,} CX gates, depth {baseline_resources['depth']:,}.")
        if circuit is not None:
            r = report["exported_circuit_resources"]
            print(f"Exported {args.export_iterations}-iteration circuit: {r['cx_gates']:,} CX gates, depth {r['depth']:,}.")
        else:
            print(report["circuit_export_status"])
        print("Success curves are ideal mathematical diagnostics. No IBM job was submitted.")
        return 0
    except (ValueError, KeyError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
