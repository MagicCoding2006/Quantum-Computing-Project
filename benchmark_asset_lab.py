#!/usr/bin/env python3
"""Run a local scaling benchmark from an existing frozen study; submits no IBM jobs."""

import argparse
from pathlib import Path
import sys

from assetlab.benchmark import run_benchmark


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[6, 10, 14])
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--shots", type=int, default=4096)
    args = parser.parse_args(argv)
    try:
        folder, result = run_benchmark(args.run, sizes=sorted(set(args.sizes)),
                                       layers=sorted(set(args.layers)), seeds=sorted(set(args.seeds)), shots=args.shots)
    except (ValueError, OSError, KeyError) as exc:
        print(f"Benchmark: {exc}", file=sys.stderr)
        return 2
    print(f"Saved {len(result['rows'])} local benchmark cases to {folder}")
    print(f"Updated {args.run / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
