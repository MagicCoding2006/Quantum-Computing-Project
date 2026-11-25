#!/usr/bin/env python3
"""Run asset correlations and a small, training-only quantum selection study."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import webbrowser

import pandas as pd

from assetlab.data import DEFAULT_TICKERS, ticker_list
from assetlab.pipeline import run_study
from assetlab.selection import MAX_CANDIDATES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true", help="synthetic correlated assets; no downloads")
    source.add_argument("--prices", type=Path, help="CSV: date column first, adjusted-price ticker columns")
    parser.add_argument("--tickers", nargs="+", help="research universe; defaults to 20 example ETFs")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat(), help="exclusive end date")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--candidates", nargs="+", help=f"2-{MAX_CANDIDATES} tickers in fixed order; default first six of universe")
    parser.add_argument("--choose", type=int, default=3)
    parser.add_argument("--windows", nargs="+", type=int, default=[60, 120, 252])
    parser.add_argument("--layers", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh", action="store_true", help="ignore the 24-hour Yahoo snapshot cache")
    parser.add_argument("--cache", default="data_cache/asset_lab")
    parser.add_argument("--out", default="results_asset_lab")
    parser.add_argument("--open", action="store_true", help="open the self-contained report after the run")
    args = parser.parse_args(argv)
    try:
        tickers = ticker_list(args.tickers or DEFAULT_TICKERS)
        candidates = ticker_list(args.candidates) if args.candidates else tickers[:6]
        if not 2 <= len(candidates) <= MAX_CANDIDATES or any(t not in tickers for t in candidates):
            raise ValueError(f"Candidates must be 2-{MAX_CANDIDATES} distinct members of --tickers.")
        if not 1 <= args.choose < len(candidates):
            raise ValueError("--choose must be between 1 and the number of candidates minus one.")
        if not 0.5 <= args.train_fraction <= 0.85:
            raise ValueError("--train-fraction must be between 0.5 and 0.85.")
        if not 1 <= args.shots <= 100000 or not 0 <= args.seed <= 2 ** 32 - 1:
            raise ValueError("Use 1-100000 shots and a seed between 0 and 2^32-1.")
        if not args.windows or any(w < 2 for w in args.windows):
            raise ValueError("Rolling windows must contain at least two return observations.")
        start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
        if pd.isna(start) or pd.isna(end) or start.tzinfo or end.tzinfo or start != start.normalize() or end != end.normalize() or start >= end:
            raise ValueError("Supply date-only --start and --end values, with start before end.")
        config = {"tickers": tickers, "candidates": candidates, "choose": args.choose,
                  "start": start.date().isoformat(), "end": end.date().isoformat(),
                  "train_fraction": args.train_fraction, "windows": sorted(set(args.windows)),
                  "layers": args.layers, "shots": args.shots, "seed": args.seed, "demo": args.demo}
        print(f"Asset Lab | {'SYNTHETIC DEMO' if args.demo else 'ADJUSTED PRICE DATA'}", flush=True)
        print(f"{len(tickers)} assets; select {args.choose} of {len(candidates)} candidates; "
              f"{args.layers}-layer ideal QAOA simulation.", flush=True)
        output, result = run_study(config, csv_path=args.prices, cache_dir=args.cache,
                                   out=args.out, refresh=args.refresh)
    except (ValueError, OSError, ImportError, KeyError) as exc:
        print(f"Asset Lab: {exc}", file=sys.stderr)
        return 2
    print("Exact training selection: " + ", ".join(result["exact"][0]["assets"]))
    print(f"QAOA feasible shots: {result['quantum']['feasible_fraction']:.1%}; "
          f"optimal shots: {result['quantum']['optimal_fraction']:.1%}")
    print(f"Report: {output / 'report.html'}")
    print(f"Run manifest and Ising model: {output}")
    if args.open:
        webbrowser.open((output / "report.html").as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
