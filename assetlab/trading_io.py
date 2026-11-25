"""Frozen trading runs, timestamped forward observations, and local reports."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

import pandas as pd

from .data import load_prices
from .pipeline import code_provenance, json_safe, write_json
from .report import refresh_report
from .trading import TradingConfig, backtest, fit_relationships, market_allowed, model_z, check_panel


def strategy_digest():
    return hashlib.sha256(Path(__file__).with_name("trading.py").read_bytes()).hexdigest()


def signal_preview(prices, universe, config):
    # Match the backtest's anchored recalibration cadence, rather than silently
    # refitting a new strategy every time the snapshot command is invoked.
    next_index = len(prices)
    formation_end = config.train_sessions + ((next_index - config.train_sessions) // config.test_sessions) * config.test_sessions
    history = prices.iloc[formation_end - config.train_sessions:formation_end]
    models, audit = fit_relationships(history, universe, config)
    allowed = market_allowed(prices, len(prices) - 1, config)
    values = []
    for ticker, model in models.items():
        z = model_z(model, prices.iloc[-1])
        candidate = allowed and -config.stop_z < z <= -config.entry_z
        values.append({"ticker": ticker, "peers": model["peers"], "z": z,
                       "eligible_entry": candidate, "action": "ENTRY_CANDIDATE" if candidate else "WATCH",
                       "half_life": model["half_life"], "pvalue": model["pvalue"]})
    values.sort(key=lambda row: (not row["eligible_entry"], row["z"], row["ticker"]))
    return {"as_of": prices.index[-1].date().isoformat(), "model_train_end": history.index[-1].date().isoformat(),
            "market_filter_allows_entries": allowed,
            "relationships": values, "accepted_models": len(models), "models": models, "audit": audit,
            "entry_candidates": [row["ticker"] for row in values if row["eligible_entry"]][:config.max_positions],
            "execution_instruction": "Research candidates only. A later session's fill must be recorded separately; no fill is inferred from this snapshot."}


def render_trading_report(folder):
    folder = Path(folder)
    data = json.loads((folder / "trading.json").read_text(encoding="utf-8"))
    if (folder / "latest_signal.json").exists():
        data["signal_preview"] = json.loads((folder / "latest_signal.json").read_text(encoding="utf-8"))
    else:
        data["signal_preview"] = json.loads((folder / "signal_preview.json").read_text(encoding="utf-8"))
    journal = folder / "signals.jsonl"
    data["forward_observations"] = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()] if journal.exists() else []
    if (folder / "quantum" / "latest.json").exists():
        data["trade_quantum"] = json.loads((folder / "quantum" / "latest.json").read_text())
    template = Path(__file__).with_name("trading_report.html").read_text(encoding="utf-8")
    encoded = json.dumps(data, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    (folder / "report.html").write_text(template.replace("__TRADING_DATA__", encoded), encoding="utf-8")
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        digest = hashlib.sha256((folder / "report.html").read_bytes()).hexdigest()
        previous = manifest["files_sha256"].get("report.html")
        manifest["report_sha256"] = digest
        manifest["report_template_sha256"] = hashlib.sha256(Path(__file__).with_name("trading_report.html").read_bytes()).hexdigest()
        manifest["files_sha256"]["report.html"] = digest
        if previous != digest:
            revision = {"updated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "previous_report_sha256": previous, "report_sha256": digest}
            for name in ("latest_signal.json", "signals.jsonl", "forward_index.json", "quantum/latest.json"):
                path = folder / name
                if path.exists():
                    manifest["files_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
                    revision[name] = manifest["files_sha256"][name]
            manifest.setdefault("report_updates", []).append(revision)
        write_json(manifest_path, manifest)


def attach_report(folder, parent):
    folder, parent = Path(folder).resolve(), Path(parent).resolve()
    if not (parent / "analysis.json").exists():
        raise ValueError("--attach-to must identify an existing Asset Lab correlation run.")
    data = json.loads((folder / "trading.json").read_text())
    target = parent / "trading"
    target.mkdir(exist_ok=True)
    write_json(target / "latest.json", {
        "run_id": data["run_id"], "report_href": Path(os.path.relpath(folder / "report.html", parent)).as_posix(),
        "metrics": data["metrics"], "cost_bps": data["config"]["cost_bps"],
        "start": data["dates"][0], "end": data["dates"][-1],
        "source": data["provenance"]["source"], "mode": "historical_walk_forward_research",
    })
    refresh_report(parent)


def run_trading(universe, config, *, start, end, demo=False, csv_path=None,
                cache="data_cache/asset_lab", out="results_asset_lab/trading", refresh=False, attach_to=None):
    required = list(dict.fromkeys([*universe, config.market]))
    prices, provenance = load_prices(required, start, end, demo=demo, csv_path=csv_path,
                                     cache_dir=cache, refresh=refresh)
    prices = check_panel(prices, universe, config)
    print(f"Fitting trailing {config.train_sessions}-session relationships; evaluating {config.test_sessions}-session folds...", flush=True)
    result = backtest(prices, universe, config)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = Path(out).resolve() / run_id
    folder.mkdir(parents=True, exist_ok=False)
    source_settings = {"start": start, "end_exclusive": end, "demo": demo, "csv_name": Path(csv_path).name if csv_path else None}
    rules = {"schema_version": 1, "strategy_version": "relative_value_long_only_v1",
             "config": asdict(config), "universe": list(universe), "source_settings": source_settings,
             "strategy_sha256": strategy_digest()}
    rule_id = hashlib.sha256(json.dumps(rules, sort_keys=True).encode()).hexdigest()
    result.update(run_id=run_id, created_at_utc=now.isoformat(), rule_id=rule_id, provenance=provenance,
                  source_settings=source_settings)
    write_json(folder / "trading.json", result)
    write_json(folder / "rules.json", {**rules, "rule_id": rule_id})
    preview = signal_preview(prices, universe, config)
    preview.update(observation_type="historical_preview", recorded_at_utc=None,
                   note="A preview from the last supplied historical close; not a forward observation.")
    write_json(folder / "signal_preview.json", preview)
    prices.to_csv(folder / "prices.csv", index_label="date")
    for name, values in [("daily", result["strategy"]["daily"]), ("trades", result["strategy"]["trades"]),
                         ("fills", result["strategy"]["fills"]), ("folds", result["strategy"]["folds"]),
                         ("models", result["model_audit"])]:
        pd.DataFrame(values).to_csv(folder / f"{name}.csv", index=False)
    pd.json_normalize([{k: v for k, v in row.items() if k != "equity"} for row in result["cost_scenarios"]], sep="_").to_csv(folder / "cost_sensitivity.csv", index=False)
    render_trading_report(folder)
    manifest = {"schema_version": 1, "run_id": run_id, "created_at_utc": now.isoformat(),
                "rule_id": rule_id, "code": code_provenance(), "data": provenance,
                "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}}
    write_json(folder / "manifest.json", manifest)
    if attach_to:
        attach_report(folder, attach_to)
    return folder, result


def append_observation(folder, prices, provenance, *, observation_type="forward_snapshot", now=None):
    """Append an actual-time signal observation, never historical paper fills."""
    folder = Path(folder)
    rules_path = folder / "rules.json"
    rules = json.loads(rules_path.read_text())
    expected_id = hashlib.sha256(json.dumps({k: v for k, v in rules.items() if k != "rule_id"}, sort_keys=True).encode()).hexdigest()
    if expected_id != rules["rule_id"]:
        raise ValueError("Frozen rules were modified. Create a new research run instead of changing a forward journal.")
    if rules["strategy_sha256"] != strategy_digest():
        raise ValueError("Strategy code changed after these rules were frozen. Create a new research run.")
    config = TradingConfig(**rules["config"])
    prices = check_panel(prices, rules["universe"], config)
    now = now or datetime.now(timezone.utc)
    if prices.index[-1].date() > now.date():
        raise ValueError("A forward snapshot cannot contain future-dated prices.")
    snapshot = signal_preview(prices, rules["universe"], config)
    snapshot_hash = hashlib.sha256(prices.to_csv().encode()).hexdigest()
    observation_id = hashlib.sha256(f"{rules['rule_id']}:{observation_type}:{snapshot_hash}".encode()).hexdigest()[:24]
    journal = folder / "signals.jsonl"
    existing = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()] if journal.exists() else []
    if any(row["observation_id"] == observation_id for row in existing):
        return None
    if existing and snapshot["as_of"] < max(row["as_of"] for row in existing):
        raise ValueError("The supplied prices are older than the last journal observation; no backdated snapshot was appended.")
    recorded_at = now.isoformat()
    snapshot.update(observation_id=observation_id, observation_type=observation_type,
                    recorded_at_utc=recorded_at, rule_id=rules["rule_id"], snapshot_sha256=snapshot_hash,
                    quote_age_calendar_days=(now.date() - prices.index[-1].date()).days,
                    provenance=provenance)
    archive = folder / "forward_snapshots" / observation_id
    archive.mkdir(parents=True, exist_ok=False)
    prices.to_csv(archive / "prices.csv", index_label="date")
    write_json(archive / "signal.json", snapshot)
    entry = {"observation_id": observation_id, "observation_type": observation_type,
             "recorded_at_utc": recorded_at, "as_of": snapshot["as_of"], "rule_id": rules["rule_id"],
             "entry_candidates": snapshot["entry_candidates"], "accepted_models": snapshot["accepted_models"],
             "market_filter_allows_entries": snapshot["market_filter_allows_entries"],
             "snapshot_sha256": snapshot_hash, "archive": archive.relative_to(folder).as_posix()}
    # One append-only write. Earlier observations and quote snapshots are retained.
    line = json.dumps(json_safe(entry), allow_nan=False) + "\n"
    with journal.open("a", encoding="utf-8") as stream:
        stream.write(line)
    write_json(folder / "latest_signal.json", snapshot)
    write_json(folder / "forward_index.json", {"journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
               "observations": len(existing) + 1, "latest_observation_id": observation_id})
    render_trading_report(folder)
    return snapshot


def collect_signals(folder, *, refresh=False, csv_path=None):
    folder = Path(folder)
    rules = json.loads((folder / "rules.json").read_text())
    config = TradingConfig(**rules["config"])
    settings = rules["source_settings"]
    if settings["csv_name"] and csv_path is None:
        raise ValueError("This run used uploaded prices. Supply --prices with an updated snapshot; it will be labeled uploaded_snapshot.")
    end = datetime.now(timezone.utc).date().isoformat()
    required = list(dict.fromkeys([*rules["universe"], config.market]))
    prices, provenance = load_prices(required, settings["start"], end, demo=settings["demo"],
                                     csv_path=csv_path, refresh=refresh)
    kind = "synthetic_snapshot" if settings["demo"] else "uploaded_snapshot" if csv_path else "forward_snapshot"
    return append_observation(folder, prices, provenance, observation_type=kind)
