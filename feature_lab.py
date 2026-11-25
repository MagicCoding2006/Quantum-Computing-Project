#!/usr/bin/env python3
"""Study shallow quantum features against classical market predictors; explicit IBM execution."""

import argparse
from pathlib import Path
import shlex
import sys
import webbrowser

import numpy as np
import pandas as pd

from assetlab.feature_experiment import collect, plan, prepare, render_report, submit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    run = sub.add_parser('run', help='freeze and evaluate ideal and finite-shot feature studies locally')
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument('--prices', type=Path)
    source.add_argument('--demo', action='store_true')
    run.add_argument('--end', required=True, help='inclusive quote cutoff')
    run.add_argument('--target', default='SPY')
    run.add_argument('--market', default='QQQ')
    run.add_argument('--train', type=int, default=128)
    run.add_argument('--validation', type=int, default=64)
    run.add_argument('--test', type=int, default=64)
    run.add_argument('--horizon', type=int, default=10)
    run.add_argument('--shots', type=int, default=256)
    run.add_argument('--cost-bps', type=float, default=10.)
    run.add_argument('--seed', type=int, default=42)
    run.add_argument('--out', default='results_asset_lab/features')
    run.add_argument('--open', action='store_true')
    for action in ('plan', 'submit', 'collect', 'report'):
        command = sub.add_parser(action)
        command.add_argument('--run', required=True, type=Path)
        command.add_argument('--open', action='store_true')
        if action == 'plan': command.add_argument('--backend', required=True)
        if action == 'submit':
            command.add_argument('--plan', required=True, type=Path)
            command.add_argument('--max-execution-seconds', type=int, default=60)
    args = parser.parse_args(argv)
    try:
        if args.action == 'run':
            config = {key: getattr(args, key) for key in ('target', 'market', 'end', 'train', 'validation', 'test', 'horizon', 'shots', 'cost_bps', 'seed')}
            if args.demo:
                rng = np.random.default_rng(args.seed)
                common = rng.normal(.0002, .009, 4500)
                returns = np.c_[common, .8*common+rng.normal(.0001, .007, 4500)]
                prices = pd.DataFrame(100*np.exp(np.cumsum(returns, axis=0)), columns=[args.target, args.market],
                                      index=pd.bdate_range(end=args.end, periods=4500))
                source = {'kind': 'synthetic_demo'}
            else:
                from assetlab.search_experiment import digest
                prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
                prices.index = pd.DatetimeIndex(prices.index)
                source = {'kind': 'supplied_adjusted_price_csv', 'path': str(args.prices.resolve()), 'sha256': digest(args.prices)}
            folder = prepare(prices, config, out=args.out, source=source)
            print(f'Frozen feature study: {folder}')
        else:
            folder = args.run.resolve()
            if args.action == 'plan':
                path = plan(folder, args.backend)
                print(f'Native feature plan: {path}/plan.json; no QPU job submitted.')
                print('Submit command (uses QPU allocation):')
                print('python feature_lab.py submit --run '+shlex.quote(str(folder))+' --plan '+shlex.quote(str(path))+' --max-execution-seconds 60')
            elif args.action == 'submit':
                receipt = submit(folder, args.plan, max_seconds=args.max_execution_seconds)
                print(f"Submitted IBM feature job {receipt['job_id']} on {receipt['backend']}.")
                print('Collect later: python feature_lab.py collect --run '+shlex.quote(str(folder)))
            elif args.action == 'collect': collect(folder)
        report = render_report(folder)
        print(f'Report: {report}')
        if args.open: webbrowser.open(report.as_uri())
        return 0
    except ImportError:
        print('Install optional dependencies: python -m pip install -r requirements-features.txt -r requirements-quantum.txt', file=sys.stderr)
        return 2
    except (ValueError, KeyError, OSError) as exc:
        print(f'Feature experiment: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
