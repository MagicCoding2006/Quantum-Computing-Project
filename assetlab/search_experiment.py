"""Frozen Grover comparison protocols, synthetic noise, and explicit IBM jobs."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import uuid

import numpy as np

from .pipeline import code_provenance, write_json
from .search import classical_reference, grover_circuit, ideal_search, search_problem, with_encoding


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def unique_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def circuit_resources(circuit):
    return {"qubits": circuit.num_qubits, "depth": circuit.depth(),
            "two_qubit_gates": sum(item.operation.num_qubits == 2 for item in circuit.data),
            "operations": {str(k): int(v) for k, v in circuit.count_ops().items()}}


def load_protocol(folder):
    folder = Path(folder).resolve()
    manifest = read_json(folder / "manifest.json")
    for relative, expected in manifest["files_sha256"].items():
        if digest(folder / relative) != expected:
            raise ValueError(f"Frozen experiment file changed: {relative}. Prepare a new experiment.")
    protocol = read_json(folder / "protocol.json")
    if protocol.get("kind") != "grover_success_comparison":
        raise ValueError("This is not a Grover comparison experiment.")
    return folder, protocol


def load_circuit(path):
    from qiskit import qpy

    with Path(path).open("rb") as stream:
        circuits = qpy.load(stream)
    if len(circuits) != 1:
        raise ValueError("Expected exactly one circuit per frozen file.")
    return circuits[0]


def prepare_experiment(run, *, iterations=(0, 1, 2), shots=256, seed=42,
                       cx_error=.001, one_qubit_error=.0001, readout_error=.01):
    from qiskit import qpy, transpile

    run = Path(run).resolve()
    source = read_json(run / "problem.json")
    audit = read_json(run / "audit.json")
    if source != audit["problem"] or source.get("kind") != "grover_rule_threshold":
        raise ValueError("The source audit and problem must agree on a Grover search problem.")
    n = source["search_qubits"]
    if not 1 <= n <= 20 or len(source["episode_masks"]) > 64:
        raise ValueError("Bounded comparison preparation supports at most 20 conditions and 64 episodes.")
    iterations = list(iterations)
    if not iterations or len(set(iterations)) != len(iterations) or len(iterations) > 4 or any(k not in range(4) for k in iterations):
        raise ValueError("Choose up to four distinct iteration counts from 0 to 3.")
    if not 1 <= shots <= 4096 or not 0 <= seed < 2**32:
        raise ValueError("Use 1-4096 shots and a seed within 0-2^32-1.")
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in (cx_error, one_qubit_error, readout_error)):
        raise ValueError("Noise parameters must be finite values between zero and one.")
    features = [[(mask >> bit) & 1 for bit in range(n)] for mask in source["episode_masks"]]
    problem = search_problem(features, source["episode_scores"], names=source["condition_names"],
                             min_support=source["min_support"], min_score=source["min_score"])
    if source.get("predicate_sha256", problem["predicate_sha256"]) != problem["predicate_sha256"]:
        raise ValueError("The source predicate hash does not match its data and thresholds.")
    encodings = {name: with_encoding(problem, name) for name in ("baseline", "optimized")}
    if max(p["score_bits"] for p in encodings.values()) > 16:
        raise ValueError("Bounded comparison preparation supports at most 16 score bits per encoding.")
    folder = run / "experiments" / unique_id()
    (folder / "circuits").mkdir(parents=True, exist_ok=False)
    (folder / "source").mkdir()
    source_evidence = {}
    for filename in ("problem.json", "audit.json", "episodes.csv"):
        if (run / filename).exists():
            shutil.copyfile(run / filename, folder / "source" / filename)
            source_evidence[filename] = digest(folder / "source" / filename)
    reference = classical_reference(problem)
    cases = []
    for k in iterations:
        for implementation, encoded in encodings.items():
            circuit = grover_circuit(encoded, k, measure=False)
            circuit.measure_all()  # Includes workspace; output register is named meas.
            circuit = transpile(circuit, basis_gates=["cx", "rz", "sx", "x"],
                                optimization_level=1, seed_transpiler=seed)
            name = f"{implementation}_k{k}"
            circuit.name = name
            relative = f"circuits/{name}.qpy"
            with (folder / relative).open("wb") as stream:
                qpy.dump(circuit, stream)
            cases.append({"id": name, "implementation": implementation, "iterations": k,
                          "file": relative, "resources": circuit_resources(circuit),
                          "ideal": ideal_search(n, reference["marked_count"], k)})
    protocol = {"schema_version": 1, "kind": "grover_success_comparison", "created_at_utc": stamp(),
        "experiment_id": folder.name, "source_run": str(run), "source_mode": audit["source"]["mode"],
        "source_evidence_sha256": source_evidence,
        "source_problem_sha256": digest(run / "problem.json"), "problem": problem,
        "predicate_sha256": problem["predicate_sha256"], "shots_per_circuit": shots, "seed": seed,
        "cases": cases, "encodings": encodings,
        "marked_count": reference["marked_count"], "candidate_count": 1 << n,
        "noise_model": {"kind": "illustrative_depolarizing_and_symmetric_readout",
            "cx_depolarizing_parameter": cx_error, "sx_x_depolarizing_parameter": one_qubit_error,
            "readout_flip_probability": readout_error, "rz_error": 0,
            "warning": "Synthetic scenario, not IBM calibration. No relaxation, crosstalk, drift, or mitigation modeled."},
        "measurement": "Low bits encode selected conditions; high bits are workspace. No postselection in primary success rate.",
        "interpretation": "Success means passing a historical integer predicate, not realized market profit. Analytical marked counts are diagnostic, not free inputs to an unknown-solution search. No best iteration is selected after observing results."}
    write_json(folder / "protocol.json", protocol)
    write_json(folder / "manifest.json", {"created_at_utc": stamp(), "code": code_provenance(),
        "files_sha256": {p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob("*")) if p.is_file()}})
    render_report(folder)
    return folder


def wilson(successes, shots):
    z = 1.959963984540054
    p = successes / shots
    denominator = 1 + z*z/shots
    center = (p + z*z/(2*shots)) / denominator
    half = z * math.sqrt(p*(1-p)/shots + z*z/(4*shots*shots)) / denominator
    return [max(0., center-half), min(1., center+half)]


def score_counts(protocol, case, raw_counts):
    n = protocol["problem"]["search_qubits"]
    width = protocol["encodings"][case["implementation"]]["total_qubits"]
    marginal = np.zeros(1 << n)
    clean = np.zeros(1 << n)
    counts = {}
    for raw, count in raw_counts.items():
        bits = str(raw).replace(" ", "")
        if not bits or set(bits) - {"0", "1"} or len(bits) > width or not isinstance(count, int) or count < 0:
            raise ValueError("Invalid full-register measurement counts.")
        state = int(bits, 2)
        mask = state & ((1 << n) - 1)
        marginal[mask] += count
        if state >> n == 0:
            clean[mask] += count
        label = f"{state:0{width}b}"
        counts[label] = counts.get(label, 0) + count
    shots = int(marginal.sum())
    if shots != protocol["shots_per_circuit"]:
        raise ValueError("Measured shot count differs from the frozen protocol.")
    marked = classical_reference(protocol["problem"])["marked"]
    passing, clean_count = int(marginal[marked].sum()), int(clean.sum())
    expected = np.zeros(1 << n)
    good = int(marked.sum())
    probability = case["ideal"]["ideal_grover_success"]
    if good:
        expected[marked] = probability/good
    if good < len(expected):
        expected[~marked] = (1-probability)/(len(expected)-good)
    return {"case_id": case["id"], "shots": shots, "passing_shots": passing,
        "passing_fraction": passing/shots, "passing_wilson_95": wilson(passing, shots),
        "workspace_zero_fraction": clean_count/shots,
        "passing_and_workspace_zero_fraction": float(clean[marked].sum()/shots),
        "search_distribution_tv_from_ideal": float(np.abs(marginal/shots-expected).sum()/2),
        "joint_distribution_tv_from_ideal": float((np.abs(clean/shots-expected).sum() + 1-clean_count/shots)/2),
        "counts": counts,
        "uncertainty": "Wilson interval assumes independent Bernoulli shots; no correction for device drift or multiple comparisons."}


def simulate_experiment(folder):
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    folder, protocol = load_protocol(folder)
    if max(p["total_qubits"] for p in protocol["encodings"].values()) > 18:
        raise ValueError("Local statevector noise sampling is limited to 18 logical qubits. The frozen protocol can still be planned for IBM; no large statevector was allocated.")
    if (folder / "simulation.json").exists():
        cached = read_json(folder / "simulation.json")
        if cached["protocol_sha256"] != digest(folder / "protocol.json"):
            raise ValueError("Cached simulation belongs to a different protocol.")
        return cached
    noise = protocol["noise_model"]
    model = NoiseModel()
    if noise["cx_depolarizing_parameter"]:
        model.add_all_qubit_quantum_error(depolarizing_error(noise["cx_depolarizing_parameter"], 2), ["cx"])
    if noise["sx_x_depolarizing_parameter"]:
        model.add_all_qubit_quantum_error(depolarizing_error(noise["sx_x_depolarizing_parameter"], 1), ["sx", "x"])
    p = noise["readout_flip_probability"]
    if p:
        model.add_all_qubit_readout_error(ReadoutError([[1-p, p], [p, 1-p]]))
    output = {"created_at_utc": stamp(), "protocol_sha256": digest(folder / "protocol.json"),
              "hardware_executed": False, "noise_model": noise, "software": code_provenance(),
              "ideal_sampled": [], "synthetic_noise": []}
    for mode in ("ideal_sampled", "synthetic_noise"):
        simulator = AerSimulator(method="statevector", noise_model=model if mode == "synthetic_noise" else None,
                                 max_parallel_threads=2, max_parallel_shots=2, max_memory_mb=512)
        for index, case in enumerate(protocol["cases"]):
            print(f"Local {mode}: {case['id']}, {protocol['shots_per_circuit']} shots", flush=True)
            started = time.perf_counter()
            result = simulator.run(load_circuit(folder / case["file"]), shots=protocol["shots_per_circuit"],
                                   seed_simulator=protocol["seed"] + index).result()
            if not result.success:
                raise ValueError(f"Local simulation failed for {case['id']}.")
            scored = score_counts(protocol, case, result.get_counts())
            scored["elapsed_seconds"] = time.perf_counter() - started
            output[mode].append(scored)
    write_json(folder / "simulation.json", output)
    render_report(folder)
    return output


def plan_hardware(folder, backend_name):
    from qiskit import qpy
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from ibm_quantum import service

    folder, protocol = load_protocol(folder)
    backend = service().backend(backend_name)
    if backend.num_qubits < max(p["total_qubits"] for p in protocol["encodings"].values()):
        raise ValueError("The selected backend has too few qubits for this protocol.")
    transpiler = generate_preset_pass_manager(backend=backend, optimization_level=2, seed_transpiler=protocol["seed"])
    plan_folder = folder / "ibm" / "plans" / unique_id()
    plan_folder.mkdir(parents=True, exist_ok=False)
    cases = []
    for case in protocol["cases"]:
        isa = transpiler.run(load_circuit(folder / case["file"]))
        name = case["id"] + ".qpy"
        with (plan_folder / name).open("wb") as stream:
            qpy.dump(isa, stream)
        cases.append({"case_id": case["id"], "file": name, "sha256": digest(plan_folder/name),
                      "resources": circuit_resources(isa),
                      "final_layout": isa.layout.final_index_layout() if isa.layout else None})
    write_json(plan_folder / "plan.json", {"created_at_utc": stamp(), "backend": backend_name,
        "experiment_id": protocol["experiment_id"], "protocol_sha256": digest(folder / "protocol.json"),
        "shots_per_circuit": protocol["shots_per_circuit"], "cases": cases,
        "hardware_executed": False, "note": "Device-native transpilation only; no job or billing estimate. Placement may differ across encodings."})
    write_json(plan_folder / "manifest.json", {"files_sha256": {
        p.name: digest(p) for p in sorted(plan_folder.iterdir()) if p.is_file()}})
    render_report(folder)
    return plan_folder


def load_plan(folder, protocol, plan_folder):
    plan_folder = Path(plan_folder).resolve()
    for relative, expected in read_json(plan_folder / "manifest.json")["files_sha256"].items():
        if digest(plan_folder / relative) != expected:
            raise ValueError("The device plan changed after preparation; create a new plan.")
    plan = read_json(plan_folder / "plan.json")
    if plan["protocol_sha256"] != digest(folder / "protocol.json") or plan["experiment_id"] != protocol["experiment_id"]:
        raise ValueError("Hardware plan belongs to a different frozen experiment.")
    if [c["case_id"] for c in plan["cases"]] != [c["id"] for c in protocol["cases"]]:
        raise ValueError("Hardware plan cases differ from the frozen protocol.")
    return plan_folder, plan


def verify_native(folder, plan_folder, *, iterations=(1,)):
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit_aer import AerSimulator

    folder, protocol = load_protocol(folder)
    plan_folder, plan = load_plan(folder, protocol, plan_folder)
    cases = [case for case in protocol["cases"] if case["iterations"] in iterations]
    if not cases:
        raise ValueError("No frozen circuits match the requested verification iterations.")
    results = []
    for case in cases:
        item = next(c for c in plan["cases"] if c["case_id"] == case["id"])
        native = load_circuit(plan_folder / item["file"])
        active = sorted({native.find_bit(q).index for op in native.data
                         if op.operation.name != "barrier" for q in op.qubits})
        compact = QuantumCircuit(QuantumRegister(len(active), "q"), *native.cregs)
        mapping = {old: new for new, old in enumerate(active)}
        for op in native.data:
            if op.operation.name != "barrier":
                compact.append(op.operation,
                    [compact.qubits[mapping[native.find_bit(q).index]] for q in op.qubits],
                    [compact.clbits[native.find_bit(c).index] for c in op.clbits])
        print(f"Ideal native verification: {case['id']}, {len(active)} active physical wires", flush=True)
        result = AerSimulator(method="matrix_product_state", max_parallel_threads=2,
            max_memory_mb=512, matrix_product_state_truncation_threshold=1e-16).run(
            compact, shots=protocol["shots_per_circuit"], seed_simulator=protocol["seed"]).result()
        if not result.success:
            raise ValueError(f"Native circuit verification failed: {case['id']}.")
        scored = score_counts(protocol, case, result.get_counts())
        scored["active_physical_wires"] = len(active)
        results.append(scored)
    output = {"verified_at_utc": stamp(), "protocol_sha256": digest(folder / "protocol.json"),
              "plan_sha256": digest(plan_folder / "plan.json"), "hardware_executed": False,
              "method": "Ideal matrix product state after removing idle wires, with numerical truncation threshold 1e-16 and no bond-dimension cap.",
              "results": results, "software": code_provenance()}
    write_json(plan_folder / "verification.json", output)
    render_report(folder)
    return output


def submit_hardware(folder, plan_folder, *, max_seconds=60):
    from qiskit_ibm_runtime import SamplerV2
    from ibm_quantum import service

    folder, protocol = load_protocol(folder)
    plan_folder, plan = load_plan(folder, protocol, plan_folder)
    if not 1 <= max_seconds <= 600:
        raise ValueError("Execution limit must be between 1 and 600 seconds.")
    attempt_path = folder / "ibm" / "submission_attempt.json"
    if attempt_path.exists():
        raise ValueError("A submission was already attempted. Use collect or reconcile the IBM dashboard; no automatic retry.")
    circuits = []
    for case in plan["cases"]:
        if digest(plan_folder / case["file"]) != case["sha256"]:
            raise ValueError("A device-native circuit changed after planning.")
        circuits.append(load_circuit(plan_folder / case["file"]))
    backend = service().backend(plan["backend"])
    sampler = SamplerV2(mode=backend, options={"max_execution_time": max_seconds})
    receipt = {"attempted_at_utc": stamp(), "protocol_sha256": digest(folder / "protocol.json"),
               "plan_sha256": digest(plan_folder / "plan.json"), "plan_folder": str(plan_folder),
               "backend": plan["backend"], "shots_per_circuit": protocol["shots_per_circuit"],
               "max_execution_seconds": max_seconds, "case_ids": [c["id"] for c in protocol["cases"]]}
    with attempt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    job = sampler.run(circuits, shots=protocol["shots_per_circuit"])
    receipt.update(job_id=job.job_id(), status="submitted")
    write_json(folder / "ibm" / "submission.json", receipt)
    render_report(folder)
    return receipt


def collect_hardware(folder):
    from ibm_quantum import service

    folder, protocol = load_protocol(folder)
    receipt_path = folder / "ibm" / "submission.json"
    if not receipt_path.exists():
        raise ValueError("No confirmed IBM receipt. If an attempt exists, reconcile the IBM dashboard before retrying.")
    receipt = read_json(receipt_path)
    if receipt["protocol_sha256"] != digest(folder / "protocol.json"):
        raise ValueError("The protocol differs from the submitted job.")
    if (folder / "ibm" / "hardware_results.json").exists():
        render_report(folder)
        return read_json(folder / "ibm" / "hardware_results.json")
    job = service().job(receipt["job_id"])
    status = str(job.status()).upper().split(".")[-1]
    print(f"IBM job {receipt['job_id']}: {status}")
    write_json(folder / "ibm" / "job_status.json", {"checked_at_utc": stamp(), "job_id": receipt["job_id"], "status": status})
    if status != "DONE":
        render_report(folder)
        if status in {"ERROR", "CANCELLED"}:
            raise ValueError(f"IBM job {receipt['job_id']} ended with {status}. Inspect its workload details; no automatic resubmission.")
        return None
    results = job.result()
    if len(results) != len(protocol["cases"]):
        raise ValueError("IBM result count does not match the frozen circuit suite.")
    output = {"created_at_utc": stamp(), "protocol_sha256": receipt["protocol_sha256"],
              "backend": receipt["backend"], "job_id": receipt["job_id"], "hardware_executed": True,
              "results": [score_counts(protocol, case, result.data.meas.get_counts())
                          for case, result in zip(protocol["cases"], results)]}
    write_json(folder / "ibm" / "hardware_results.json", output)
    if callable(getattr(job, "metrics", None)):
        try:
            metrics = json.loads(json.dumps(job.metrics(), default=str))
            write_json(folder / "ibm" / "job_metrics.json", {"recorded_at_utc": stamp(),
                "job_id": receipt["job_id"], "metrics": metrics})
        except Exception as exc:
            # A secondary metadata failure must not discard confirmed measurements.
            write_json(folder / "ibm" / "metrics_status.json", {"error_type": type(exc).__name__, "job_id": receipt["job_id"]})
    render_report(folder)
    return output


def render_report(folder):
    folder, protocol = load_protocol(folder)
    data = {"protocol": protocol}
    for name, relative in (("simulation", "simulation.json"), ("hardware", "ibm/hardware_results.json")):
        path = folder / relative
        if path.exists():
            value = read_json(path)
            if value["protocol_sha256"] != digest(folder / "protocol.json"):
                raise ValueError("Saved results belong to a different protocol.")
            data[name] = value
    receipt = folder / "ibm" / "submission.json"
    if receipt.exists():
        data["submission"] = read_json(receipt)
        if data["submission"]["protocol_sha256"] != digest(folder / "protocol.json"):
            raise ValueError("Submission receipt belongs to a different protocol.")
    status_path = folder / "ibm" / "job_status.json"
    if status_path.exists():
        data["job_status"] = read_json(status_path)
        if "submission" not in data or data["job_status"]["job_id"] != data["submission"]["job_id"]:
            raise ValueError("Observed job status does not match the submission receipt.")
    if "hardware" in data and (not data["hardware"].get("hardware_executed") or
            "submission" not in data or data["hardware"]["job_id"] != data["submission"]["job_id"]):
        raise ValueError("Hardware results do not match a confirmed submission receipt.")
    metrics_path = folder / "ibm" / "job_metrics.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        if "submission" not in data or metrics["job_id"] != data["submission"]["job_id"]:
            raise ValueError("Job metrics do not match the submission receipt.")
        data["job_metrics"] = metrics["metrics"]
    plans = sorted((folder / "ibm" / "plans").glob("*/plan.json"))
    if "submission" in data:
        local = folder / "ibm" / "plans" / Path(data["submission"]["plan_folder"]).name / "plan.json"
        submitted = local if local.exists() else Path(data["submission"]["plan_folder"]) / "plan.json"
        if digest(submitted) != data["submission"]["plan_sha256"]:
            raise ValueError("The submitted native plan changed after submission.")
        plans = [submitted]
    if plans:
        plan = read_json(plans[-1])
        if plan["protocol_sha256"] != digest(folder / "protocol.json"):
            raise ValueError("Saved hardware plan belongs to a different protocol.")
        data["plan"] = plan
        verification = plans[-1].parent / "verification.json"
        if verification.exists():
            value = read_json(verification)
            if value["protocol_sha256"] != digest(folder / "protocol.json") or value["plan_sha256"] != digest(plans[-1]):
                raise ValueError("Native verification belongs to a different protocol or plan.")
            data["native_verification"] = value
    encoded = json.dumps(data, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = Path(__file__).with_name("search_experiment_report.html").read_text()
    (folder / "report.html").write_text(template.replace("__EXPERIMENT_DATA__", encoded), encoding="utf-8")
    write_json(folder / "report_metadata.json", {"rendered_at_utc": stamp(),
        "protocol_sha256": digest(folder / "protocol.json"), "report_sha256": digest(folder / "report.html"),
        "template_sha256": digest(Path(__file__).with_name("search_experiment_report.html")), "code": code_provenance()})
    return folder / "report.html"
