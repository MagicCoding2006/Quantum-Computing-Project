#!/usr/bin/env python3
"""Backtest a frozen long-only relative-value strategy and record forward signal snapshots."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import shlex
import webbrowser
from zoneinfo import ZoneInfo

import pandas as pd

from assetlab.data import DEFAULT_TICKERS, ticker_list
from assetlab.trading import TradingConfig
from assetlab.trading_io import collect_signals, run_trading


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="historical walk-forward backtest; no orders placed")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true")
    source.add_argument("--prices", type=Path)
    run.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    run.add_argument("--start", default="2010-01-01")
    run.add_argument("--end", default=datetime.now(ZoneInfo("America/Indiana/Indianapolis")).date().isoformat(), help="exclusive end date")
    run.add_argument("--train-sessions", type=int, default=504)
    run.add_argument("--test-sessions", type=int, default=63)
    run.add_argument("--cost-bps", type=float, default=10)
    run.add_argument("--entry-z", type=float, default=2)
    run.add_argument("--exit-z", type=float, default=0.5)
    run.add_argument("--stop-z", type=float, default=4)
    run.add_argument("--max-hold", type=int, default=10)
    run.add_argument("--max-positions", type=int, default=5)
    run.add_argument("--initial-capital", type=float, default=100000)
    run.add_argument("--market", default="SPY")
    run.add_argument("--no-trend-filter", action="store_true")
    run.add_argument("--refresh", action="store_true")
    run.add_argument("--out", default="results_asset_lab/trading")
    run.add_argument("--attach-to", type=Path, help="add a trading summary to an existing correlation/quantum report")
    run.add_argument("--open", action="store_true")
    signal = sub.add_parser("signals", help="record a timestamped candidate snapshot; no inferred paper fills")
    signal.add_argument("--run", required=True, type=Path)
    signal.add_argument("--prices", type=Path)
    signal.add_argument("--refresh", action="store_true")
    quantum = sub.add_parser("quantum", help="fit and export a QAOA trade selector; submits no hardware job")
    quantum.add_argument("--run", required=True, type=Path)
    mode = quantum.add_mutually_exclusive_group()
    mode.add_argument("--example", action="store_true", help="first chronological historical snapshot with at least two eligible signals")
    mode.add_argument("--as-of", help="an explicit historical session in the saved price snapshot")
    quantum.add_argument("--max-candidates", type=int, default=8)
    quantum.add_argument("--capacity", type=int, default=3, help="maximum new positions; unused slots stay in cash")
    quantum.add_argument("--horizon", type=int, default=10)
    quantum.add_argument("--risk-aversion", type=float, default=5)
    quantum.add_argument("--layers", type=int, default=1)
    quantum.add_argument("--shots", type=int, default=4096)
    quantum.add_argument("--seed", type=int, default=42)
    quantum.add_argument("--open", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "quantum":
            from assetlab.trade_quantum import create_trade_quantum
            folder, data = create_trade_quantum(args.run, example=args.example, as_of=args.as_of,
                       max_candidates=args.max_candidates, capacity=args.capacity, horizon=args.horizon,
                       risk_aversion=args.risk_aversion, layers=args.layers, shots=args.shots, seed=args.seed)
            if folder is None:
                print("No eligible entry candidates in this snapshot; no circuit or hardware job created.")
                print("To test the workflow on the first eligible historical example, add --example.")
                return 0
            print(f"Prepared {data['snapshot']['mode']} as of {data['snapshot']['as_of']}: "
                  f"{len(data['problem']['decision_tickers'])} trade decisions, {data['problem']['slack_bits']} unused-slot bits.")
            selection = ", ".join(data["quantum"]["selected_assets"]) or ("CASH" if data["quantum"]["best_feasible_mask"] is not None else "no feasible measured selection")
            print("Best feasible simulated selection: " + selection)
            print(f"Report: {folder / 'report.html'}")
            print("Hardware command (uses QPU allocation):")
            print("python ibm_quantum.py submit --run " + shlex.quote(str(folder)) + " --backend ibm_kingston --shots " + str(args.shots) + " --max-execution-seconds 60")
            if args.open:
                webbrowser.open((folder / "report.html").as_uri())
            return 0
        if args.action == "signals":
            snapshot = collect_signals(args.run, refresh=args.refresh, csv_path=args.prices)
            if snapshot:
                print(f"Recorded {snapshot['observation_type']} at {snapshot['recorded_at_utc']}; prices as of {snapshot['as_of']}.")
                print("Entry candidates: " + (", ".join(snapshot["entry_candidates"]) or "none"))
                print("No brokerage orders or paper fills were created.")
            else:
                print("This price snapshot was already logged; no duplicate observation added.")
            return 0
        universe = ticker_list(args.tickers)
        config = TradingConfig(train_sessions=args.train_sessions, test_sessions=args.test_sessions,
                               cost_bps=args.cost_bps, entry_z=args.entry_z, exit_z=args.exit_z, stop_z=args.stop_z,
                               max_hold=args.max_hold, max_positions=args.max_positions, initial_capital=args.initial_capital,
                               market=args.market.strip().upper(), trend_filter=not args.no_trend_filter)
        start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
        if pd.isna(start) or pd.isna(end) or start.tzinfo or end.tzinfo or start != start.normalize() or end != end.normalize() or start >= end:
            raise ValueError("Use date-only start/end values, with start before end.")
        print("Trade Lab | " + ("SYNTHETIC DEMO" if args.demo else "HISTORICAL PRICE RESEARCH"), flush=True)
        folder, result = run_trading(universe, config, start=start.date().isoformat(), end=end.date().isoformat(),
                                     demo=args.demo, csv_path=args.prices, out=args.out, refresh=args.refresh, attach_to=args.attach_to)
        metrics = result["metrics"]
        print(f"Walk-forward net return: {metrics['total_return']:.2%}; max drawdown: {metrics['max_drawdown']:.2%}; closed trades: {metrics['trades']}")
        print(f"Modeled costs: ${metrics['total_cost']:,.2f}; average invested exposure: {metrics['mean_exposure']:.1%}")
        print(f"Report: {folder / 'report.html'}")
        print(f"Frozen rules: {folder / 'rules.json'}")
        if args.open:
            webbrowser.open((folder / "report.html").as_uri())
        return 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f"Trade Lab: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
