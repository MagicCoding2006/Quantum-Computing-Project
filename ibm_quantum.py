#!/usr/bin/env python3
"""Prepare, submit, or collect one fixed-parameter IBM quantum experiment.

Preparation is offline. Submission is always explicit and never happens as part
of an asset study. The QAOA angles come from the study's local training fit.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from assetlab.pipeline import code_provenance, write_json
from assetlab.quantum import qiskit_circuit
from assetlab.selection import summarize_counts


def load_run(path):
    path = Path(path).resolve()
    data = json.loads((path / "analysis.json").read_text(encoding="utf-8"))
    problem = json.loads((path / "problem.json").read_text(encoding="utf-8"))
    if problem != data["problem"]:
        raise ValueError("problem.json disagrees with the problem recorded in analysis.json.")
    return path, data, problem


def prepare(path):
    from qiskit import qasm3, qpy

    path, data, problem = load_run(path)
    circuit = qiskit_circuit(problem, data["quantum"]["parameters"])
    folder = path / "ibm"
    folder.mkdir(exist_ok=True)
    with (folder / "circuit.qpy").open("wb") as stream:
        qpy.dump(circuit, stream)
    # The elementary gates used here need no optional QASM importer/exporter.
    (folder / "circuit.qasm").write_text(qasm3.dumps(circuit), encoding="utf-8")
    write_json(folder / "preparation.json", {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_id": data["run_id"], "num_qubits": circuit.num_qubits,
        "depth_before_transpilation": circuit.depth(),
        "operations_before_transpilation": dict(circuit.count_ops()),
        "circuit_qpy_sha256": hashlib.sha256((folder / "circuit.qpy").read_bytes()).hexdigest(),
        "hardware_executed": False, "software": code_provenance(),
    })
    return path, data, problem, circuit


def service():
    from qiskit_ibm_runtime import QiskitRuntimeService

    # Neither tokens nor environment contents are written into result files.
    key = os.environ.get("IBM_QUANTUM_API_KEY")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE")
    if key:
        if not instance:
            raise ValueError("Set IBM_QUANTUM_INSTANCE to the intended instance CRN alongside the API key.")
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=key, instance=instance)
    return QiskitRuntimeService(instance=instance) if instance else QiskitRuntimeService()


def submit(path, backend_name, shots, max_seconds):
    from qiskit import qpy
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2

    path, data, problem = load_run(path)
    folder = path / "ibm"
    if (folder / "submission_attempt.json").exists():
        raise ValueError("This run already has a submission attempt. Use collect or inspect the IBM "
                         "dashboard before starting a new study; automatic retries are disabled.")
    _, _, _, circuit = prepare(path)
    runtime = service()
    backend = runtime.backend(backend_name)
    transpiler = generate_preset_pass_manager(backend=backend, optimization_level=2,
                                             seed_transpiler=data["config"]["seed"])
    isa = transpiler.run(circuit)
    with (folder / "transpiled.qpy").open("wb") as stream:
        qpy.dump(isa, stream)
    sampler = SamplerV2(mode=backend, options={"max_execution_time": max_seconds})
    # Persist the attempt before the network mutation. An uncertain response
    # must be reconciled in the IBM dashboard, never retried blindly.
    attempt = {"attempted_at_utc": datetime.now(timezone.utc).isoformat(),
               "source_run_id": data["run_id"], "backend": backend_name, "shots": shots,
               "max_execution_seconds": max_seconds, "depth": isa.depth(),
               "two_qubit_gates": sum(1 for item in isa.data if item.operation.num_qubits == 2),
               "problem_sha256": hashlib.sha256((path / "problem.json").read_bytes()).hexdigest(),
               "status": "submission_attempted"}
    # Exclusive create prevents concurrent callers from submitting the same run.
    with (folder / "submission_attempt.json").open("x", encoding="utf-8") as stream:
        json.dump(attempt, stream, indent=2)
    job = sampler.run([isa], shots=shots)
    receipt = {**attempt, "job_id": job.job_id(), "status": "submitted"}
    write_json(folder / "submission.json", receipt)
    print(f"Submitted IBM job {job.job_id()} on {backend_name}. Run collect later; no waiting worker required.")


def collect(path):
    path, _, problem = load_run(path)
    folder = path / "ibm"
    receipt_path = folder / "submission.json"
    if not receipt_path.exists():
        raise ValueError("No confirmed submission receipt. If an attempt exists, inspect the IBM dashboard "
                         "to reconcile its outcome before submitting again.")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256((path / "problem.json").read_bytes()).hexdigest()
    if digest != receipt["problem_sha256"]:
        raise ValueError("The model changed after submission; refusing to score against a different problem.")
    job = service().job(receipt["job_id"])
    status = str(job.status()).upper().split(".")[-1]
    print(f"IBM job {receipt['job_id']}: {status}")
    if status != "DONE":
        return
    raw_counts = job.result()[0].data.meas.get_counts()
    counts = {int(bits.replace(" ", ""), 2): int(count) for bits, count in raw_counts.items()}
    result = summarize_counts(counts, problem)
    result.update(backend=receipt["backend"], job_id=receipt["job_id"],
                  collected_at_utc=datetime.now(timezone.utc).isoformat(), hardware_executed=True)
    write_json(folder / "hardware_results.json", result)
    print(f"Feasible shots: {result['feasible_fraction']:.1%}; optimal shots: {result['optimal_fraction']:.1%}")
    print(f"Saved {folder / 'hardware_results.json'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "submit", "collect"):
        command = subparsers.add_parser(action)
        command.add_argument("--run", required=True, type=Path)
        if action == "submit":
            command.add_argument("--backend", required=True, help="explicit IBM backend name")
            command.add_argument("--shots", type=int, default=4096)
            command.add_argument("--max-execution-seconds", type=int, default=60)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            path, _, _, _ = prepare(args.run)
            print(f"Prepared local circuit: {path / 'ibm' / 'circuit.qasm'}; no hardware job submitted.")
        elif args.action == "submit":
            if not 1 <= args.shots <= 100000 or not 1 <= args.max_execution_seconds <= 600:
                raise ValueError("Use 1-100000 shots and a 1-600 second execution limit.")
            submit(args.run, args.backend, args.shots, args.max_execution_seconds)
        else:
            collect(args.run)
    except ImportError:
        print("Install the optional IBM dependencies: python3 -m pip install -r requirements-quantum.txt", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError) as exc:
        print(f"IBM experiment: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
