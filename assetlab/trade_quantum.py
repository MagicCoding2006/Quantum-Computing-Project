"""Quantum trade-selection experiments from frozen, timestamped signal inputs."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from .pipeline import code_provenance, write_json
from .quantum import simulate
from .selection import MAX_CANDIDATES, binary_states, encode_qubo, solve_exact
from .trading import TradingConfig, check_panel, fit_relationships, market_allowed, model_z
from .trading_io import render_trading_report, strategy_digest


def build_trade_problem(covariance, names, net_returns, *, capacity, fraction, risk_aversion, horizon):
    n = len(names)
    capacity = min(capacity, n)
    if not n or capacity < 1 or len(set(names)) != n:
        raise ValueError("A trade experiment needs distinct eligible assets and positive capacity.")
    slack_bits = capacity.bit_length()
    total = n + slack_bits
    if total > MAX_CANDIDATES:
        raise ValueError(f"Trade decisions plus cash/slack encoding must fit within {MAX_CANDIDATES} qubits.")
    covariance = np.asarray(covariance, dtype=float)
    means = np.asarray(net_returns, dtype=float)
    if covariance.shape != (n, n) or means.shape != (n,) or not np.isfinite(covariance).all() or not np.isfinite(means).all():
        raise ValueError("Expected scores and covariance must be finite and match the candidate list.")
    if not np.allclose(covariance, covariance.T) or np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError("Risk covariance must be symmetric and positive semidefinite.")
    if not np.isfinite(fraction + risk_aversion) or not 0 < fraction <= 1 or capacity * fraction > 1 + 1e-12 or risk_aversion < 0 or horizon < 1:
        raise ValueError("Use nonnegative risk aversion and unlevered positive position fractions and horizon.")
    risk = risk_aversion * covariance * fraction ** 2
    reward = means * fraction
    scale = max(float(np.abs(risk).max()), float(np.abs(reward).max()), 1e-8)
    risk, reward = risk / scale, reward / scale
    bound = float(np.abs(risk).sum() + np.abs(reward).sum())
    penalty = 2 * bound + 1
    # Binary slack represents unused position slots uniquely: sum(x)+s=K.
    # Every subset with <=K real assets has one feasible encoding, including cash.
    budget = np.r_[np.ones(n), [1 << i for i in range(slack_bits)]]
    qubo = np.diag(penalty * (budget ** 2 - 2 * capacity * budget))
    qubo += np.triu(2 * penalty * np.outer(budget, budget), 1)
    qubo[:n, :n] += np.diag(np.diag(risk) - reward) + np.triu(2 * risk, 1)
    return {
        "schema_version": 1, "kind": "trade_selection", "tickers": [*names, *[f"unused_slots_bit_{i}" for i in range(slack_bits)]],
        "decision_tickers": list(names), "choose": capacity, "slack_bits": slack_bits,
        "position_fraction": fraction, "risk_aversion": risk_aversion, "horizon_sessions": horizon,
        "horizon_covariance": covariance.tolist(), "expected_net_returns": means.tolist(),
        "objective_scale": scale, "penalty": penalty,
        "objective": "Minimize risk_aversion * horizon_variance - model_implied_net_return, with at most K new trades; cash allowed",
        "feasibility": "sum(real_asset_bits) + binary_unused_slots = capacity; no shorting or leverage",
        "bit_order": "Real assets occupy low-index bits; unused-slot bits follow. Display strings show highest bit on the left.",
        **encode_qubo(qubo, penalty * capacity ** 2),
    }


def candidate_input(prices, universe, config, as_of, *, max_candidates=8, horizon=10, model_cache=None):
    """Use only observed prices and the fold's earlier formation window."""
    known = prices.loc[:pd.Timestamp(as_of)]
    if len(known) < config.train_sessions:
        raise ValueError("The chosen snapshot lacks formation history.")
    next_index = len(known)
    end = config.train_sessions + ((next_index - config.train_sessions) // config.test_sessions) * config.test_sessions
    history = known.iloc[end - config.train_sessions:end]
    if model_cache is not None and end in model_cache:
        models = model_cache[end]
    else:
        models, _ = fit_relationships(history, universe, config)
        if model_cache is not None:
            model_cache[end] = models
    allowed = market_allowed(known, len(known) - 1, config)
    rows = []
    for name, model in models.items():
        z = model_z(model, known.iloc[-1])
        if not allowed or not -config.stop_z < z <= -config.entry_z:
            continue
        # Convergence score assumes a stationary peer basket. It is not a
        # calibrated forecast of absolute returns or established alpha.
        persistence = float(0.5 ** (horizon / model["half_life"]))
        implied_log = -z * model["residual_std"] * (1 - persistence)
        gross = float(np.expm1(implied_log))
        round_trip = 2 * config.cost_bps / 10000
        rows.append({"ticker": name, "z": float(z), "peers": model["peers"],
                     "half_life": model["half_life"], "model_gross_return": gross,
                     "assumed_round_trip_cost": round_trip, "model_net_return": gross - round_trip})
    # Same predetermined signal ordering as the strategy, not a realized-P&L ranking.
    rows.sort(key=lambda row: (row["z"], row["ticker"]))
    rows = rows[:max_candidates]
    names = [row["ticker"] for row in rows]
    covariance = history[names].pct_change(fill_method=None).iloc[1:].cov().to_numpy() * horizon if names else np.empty((0, 0))
    return rows, covariance, {"as_of": known.index[-1].date().isoformat(),
                              "training_end": history.index[-1].date().isoformat(),
                              "market_filter_allows_entries": allowed,
                              "available_history_rows": len(known)}


def render_trade_quantum_report(folder):
    folder = Path(folder)
    data = json.loads((folder / "analysis.json").read_text())
    data["circuit_prepared"] = (folder / "ibm" / "circuit.qasm").exists()
    hardware_path = folder / "ibm" / "hardware_results.json"
    if hardware_path.exists():
        receipt = json.loads((folder / "ibm" / "submission.json").read_text())
        hardware = json.loads(hardware_path.read_text())
        digest = hashlib.sha256((folder / "problem.json").read_bytes()).hexdigest()
        if receipt["source_run_id"] != data["run_id"] or receipt["problem_sha256"] != digest or hardware["job_id"] != receipt["job_id"]:
            raise ValueError("Hardware results do not match the recorded trade-selection experiment.")
        data["hardware"] = hardware
        data["submission"] = receipt
    encoded = json.dumps(data, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = Path(__file__).with_name("trade_quantum_report.html").read_text()
    (folder / "report.html").write_text(template.replace("__QUANTUM_TRADE_DATA__", encoded), encoding="utf-8")
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        previous = manifest["files_sha256"].get("report.html")
        current = hashlib.sha256((folder / "report.html").read_bytes()).hexdigest()
        manifest["files_sha256"]["report.html"] = current
        if current != previous:
            manifest.setdefault("report_updates", []).append({"updated_at_utc": datetime.now(timezone.utc).isoformat(),
                       "previous_report_sha256": previous, "report_sha256": current})
            for path in (hardware_path, folder / "ibm" / "submission.json"):
                if path.exists():
                    manifest["files_sha256"][path.relative_to(folder).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(manifest_path, manifest)
    parent = folder.parent.parent
    if (parent / "trading.json").exists():
        parent_data = json.loads((parent / "trading.json").read_text())
        if parent_data["run_id"] == data["source_trading_run_id"]:
            summary = {"run_id": data["run_id"], "mode": data["snapshot"]["mode"], "as_of": data["snapshot"]["as_of"],
                       "report_href": f"quantum/{folder.name}/report.html", "num_qubits": len(data["problem"]["tickers"]),
                       "candidate_assets": data["problem"]["decision_tickers"],
                       "ideal_selected_assets": data["quantum"]["selected_assets"],
                       "hardware": data.get("hardware"), "status": "ready"}
            write_json(parent / "quantum" / "latest.json", summary)
            render_trading_report(parent)


def create_trade_quantum(run, *, example=False, as_of=None, max_candidates=8, capacity=3,
                         horizon=10, risk_aversion=5.0, layers=1, shots=4096, seed=42):
    if example and as_of is not None:
        raise ValueError("Choose --example or --as-of, not both.")
    run = Path(run).resolve()
    rules = json.loads((run / "rules.json").read_text())
    expected = hashlib.sha256(json.dumps({k: v for k, v in rules.items() if k != "rule_id"}, sort_keys=True).encode()).hexdigest()
    if expected != rules["rule_id"] or rules["strategy_sha256"] != strategy_digest():
        raise ValueError("Trading rules or strategy code changed after freezing; create a new trading run.")
    if not 1 <= max_candidates <= 12 or not 1 <= capacity <= rules["config"]["max_positions"] or not 1 <= horizon <= rules["config"]["max_hold"]:
        raise ValueError("Use 1-12 candidates, capacity within frozen position limits, and horizon within the frozen holding limit.")
    if not np.isfinite(risk_aversion) or risk_aversion < 0:
        raise ValueError("Risk aversion must be finite and nonnegative.")
    if layers not in (1, 2, 3) or not 1 <= shots <= 100000 or not 0 <= seed < 2 ** 32:
        raise ValueError("Use 1-3 layers, 1-100000 shots, and a nonnegative 32-bit seed.")
    config = TradingConfig(**rules["config"])
    prices = pd.read_csv(run / "prices.csv", index_col=0, parse_dates=True)
    mode = "historical_replay" if as_of else "historical_example" if example else "latest_snapshot"
    if not example and as_of is None and (run / "latest_signal.json").exists():
        recorded = json.loads((run / "latest_signal.json").read_text())
        prices = pd.read_csv(run / "forward_snapshots" / recorded["observation_id"] / "prices.csv", index_col=0, parse_dates=True)
    elif not example and as_of is None:
        mode = "historical_preview"
    prices = check_panel(prices, rules["universe"], config)
    if example:
        rows = []
        cache = {}
        # Pick the first eligible formation snapshot chronologically. Realized
        # later returns never enter example selection or the optimization model.
        for index in range(config.train_sessions - 1, len(prices)):
            rows, covariance, snapshot = candidate_input(prices, rules["universe"], config, prices.index[index],
                                                         max_candidates=max_candidates, horizon=horizon, model_cache=cache)
            if len(rows) >= 2:
                break
        if len(rows) < 2:
            raise ValueError("No historical snapshot has two eligible signals. No circuit was created; the frozen entry rules were not relaxed.")
    else:
        date = pd.Timestamp(as_of) if as_of else prices.index[-1]
        if date not in prices.index:
            raise ValueError("--as-of must be an observed session in the saved price snapshot.")
        rows, covariance, snapshot = candidate_input(prices, rules["universe"], config, date,
                                                     max_candidates=max_candidates, horizon=horizon)
    if not rows:
        target = run / "quantum"
        target.mkdir(exist_ok=True)
        write_json(target / "latest.json", {"status": "no_candidates", "mode": mode, "as_of": snapshot["as_of"],
                   "message": "No entry candidates meet the frozen rules; no simulation or hardware job was created."})
        render_trading_report(run)
        return None, None
    problem = build_trade_problem(covariance, [r["ticker"] for r in rows], [r["model_net_return"] for r in rows],
                                   capacity=capacity, fraction=1 / config.max_positions,
                                   risk_aversion=risk_aversion, horizon=horizon)
    exact = solve_exact(problem)
    quantum = simulate(problem, layers=layers, shots=shots, seed=seed)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = run / "quantum" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    snapshot["mode"] = mode
    payload = {"schema_version": 1, "kind": "trade_selection", "run_id": run_id,
               "source_trading_run_id": json.loads((run / "trading.json").read_text())["run_id"],
               "source_rule_id": rules["rule_id"], "created_at_utc": now.isoformat(),
               "config": {"seed": seed, "layers": layers, "shots": shots, "horizon": horizon, "risk_aversion": risk_aversion},
               "snapshot": snapshot, "candidates": rows, "problem": problem, "exact": exact, "quantum": quantum,
               "assumptions": [
                   "Convergence scores assume a stationary peer basket and are not calibrated predictions or established alpha.",
                   "Covariance uses only the frozen formation window; no later returns enter the objective.",
                   "Round-trip cost estimates use twice the frozen one-way cost; actual execution costs can differ.",
                   "Binary unused-slot variables allow staying in cash without duplicate encodings of a real portfolio.",
                   "This selects new hypothetical trades without existing account holdings; it places no broker orders.",
                   "The original backtest is unchanged. This experiment does not establish improved strategy P&L.",
                   "Hardware submission is separate; selecting --example creates an explicitly historical test.",
               ]}
    write_json(folder / "analysis.json", payload)
    write_json(folder / "problem.json", problem)
    prices.loc[:snapshot["as_of"]].to_csv(folder / "prices_as_of.csv", index_label="date")
    render_trade_quantum_report(folder)
    write_json(folder / "manifest.json", {"run_id": run_id, "source_rule_id": rules["rule_id"],
               "code": code_provenance(), "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                           for p in folder.iterdir() if p.is_file()}})
    return folder, payload
