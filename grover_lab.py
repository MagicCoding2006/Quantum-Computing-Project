#!/usr/bin/env python3
"""Prepare, simulate, plan, explicitly submit, and inspect a frozen Grover comparison."""

import argparse
from pathlib import Path
import shlex
import sys
import webbrowser

from assetlab.search_experiment import (collect_hardware, plan_hardware, prepare_experiment,
                                       render_report, simulate_experiment, submit_hardware, verify_native)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare", help="freeze comparison circuits from a saved search audit; offline")
    prepare.add_argument("--run", required=True, type=Path)
    prepare.add_argument("--iterations", nargs="+", type=int, default=[0, 1, 2])
    prepare.add_argument("--shots", type=int, default=256)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--cx-error", type=float, default=.001)
    prepare.add_argument("--one-qubit-error", type=float, default=.0001)
    prepare.add_argument("--readout-error", type=float, default=.01)
    prepare.add_argument("--simulate", action="store_true", help="also run local ideal and synthetic-noise sampling")
    prepare.add_argument("--open", action="store_true")
    for name in ("simulate", "plan", "verify", "submit", "collect", "report"):
        command = sub.add_parser(name)
        command.add_argument("--experiment", required=True, type=Path)
        command.add_argument("--open", action="store_true")
        if name == "plan":
            command.add_argument("--backend", required=True)
        if name in {"submit", "verify"}:
            command.add_argument("--plan", required=True, type=Path)
        if name == "verify":
            command.add_argument("--iterations", nargs="+", type=int, default=[1])
        if name == "submit":
            command.add_argument("--max-execution-seconds", type=int, default=60)
    args = parser.parse_args(argv)
    try:
        folder = args.experiment.resolve() if hasattr(args, "experiment") else None
        if args.action == "prepare":
            folder = prepare_experiment(args.run, iterations=args.iterations, shots=args.shots, seed=args.seed,
                cx_error=args.cx_error, one_qubit_error=args.one_qubit_error, readout_error=args.readout_error)
            print(f"Frozen experiment: {folder}", flush=True)
            if args.simulate:
                simulate_experiment(folder)
        elif args.action == "simulate":
            simulate_experiment(folder)
        elif args.action == "plan":
            plan = plan_hardware(folder, args.backend)
            print(f"Device-native plan: {plan / 'plan.json'}; no QPU job submitted.")
            print("Submit command (uses QPU allocation):")
            print("python grover_lab.py submit --experiment " + shlex.quote(str(folder)) +
                  " --plan " + shlex.quote(str(plan)) + " --max-execution-seconds 60")
        elif args.action == "verify":
            verify_native(folder, args.plan, iterations=args.iterations)
        elif args.action == "submit":
            receipt = submit_hardware(folder, args.plan, max_seconds=args.max_execution_seconds)
            print(f"Submitted IBM job {receipt['job_id']} on {receipt['backend']}.")
            print("Collect later: python grover_lab.py collect --experiment " + shlex.quote(str(folder)))
        elif args.action == "collect":
            collect_hardware(folder)
        report = render_report(folder)
        print(f"Report: {report}")
        if args.open:
            webbrowser.open(report.as_uri())
        return 0
    except ImportError:
        print("Install experiment dependencies: python -m pip install -r requirements-quantum.txt -r requirements-noise.txt", file=sys.stderr)
        return 2
    except (ValueError, OSError, KeyError) as exc:
        print(f"Grover experiment: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
