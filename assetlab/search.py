"""Reversible threshold search over conjunctions of observed market conditions.

The phase oracle computes support and a signed, quantized historical return
sum from observations. It never accepts a precomputed list of winning rules.
This is a feasibility experiment, not a forecast or a complete trading ledger.
"""

import hashlib
import json
import math
import time

import numpy as np


def search_problem(features, scores, *, min_support=2, min_score=1, names=None, implementation="optimized"):
    values = np.asarray(features)
    raw_scores = np.asarray(scores, dtype=float)
    if values.ndim != 2 or not 1 <= values.shape[1] <= 64 or not 1 <= len(values) <= 256:
        raise ValueError("Use 1-64 binary conditions and 1-256 training episodes.")
    if not np.isin(values, [0, 1]).all():
        raise ValueError("Conditions must contain only zero or one.")
    if raw_scores.shape != (len(values),) or not np.isfinite(raw_scores).all() or not np.equal(raw_scores, np.round(raw_scores)).all():
        raise ValueError("Each episode needs one finite integer return score.")
    if np.abs(raw_scores).max() > 1000000:
        raise ValueError("Use integer scores of at most 1,000,000 in absolute value.")
    if not isinstance(min_support, int) or not 1 <= min_support <= len(values):
        raise ValueError("Minimum support must be within the episode count.")
    if not isinstance(min_score, int) or abs(min_score) > 256000000:
        raise ValueError("Minimum score must be a bounded integer.")
    n = values.shape[1]
    names = list(names) if names is not None else [f"condition_{i}" for i in range(n)]
    if len(names) != n or len(set(names)) != n or any(not isinstance(s, str) or not s for s in names):
        raise ValueError("Supply one distinct name per condition.")
    scores = [int(v) for v in raw_scores]
    masks = [sum(int(bit) << i for i, bit in enumerate(row)) for row in values]
    offset = sum(-value for value in scores if value < 0)
    max_encoded_score = offset + sum(value for value in scores if value > 0)
    count_bits = len(values).bit_length()
    score_bits = max(1, max_encoded_score.bit_length())
    problem = {
        "schema_version": 2, "kind": "grover_rule_threshold", "condition_names": names,
        "episode_masks": masks, "episode_scores": scores,
        "min_support": min_support, "min_score": min_score,
        "score_offset": offset, "support_bits": count_bits, "score_bits": score_bits,
        "search_qubits": n, "workspace_qubits": count_bits + score_bits + 3,
        "total_qubits": n + count_bits + score_bits + 3,
        "rule": "A selected condition must be true in an episode; mask zero means always enter.",
        "predicate": "episode_count >= min_support AND sum(quantized_net_episode_return) >= min_score",
        "limitations": "Training-only aggregate return sum; no drawdown, risk normalization, or forward profitability claim.",
    }
    predicate = {key: problem[key] for key in ("condition_names", "episode_masks", "episode_scores", "min_support", "min_score")}
    problem["predicate_sha256"] = hashlib.sha256(json.dumps(predicate, sort_keys=True).encode()).hexdigest()
    return with_encoding(problem, implementation)


def encoding_plan(problem, implementation=None):
    implementation = implementation or problem.get("implementation", "baseline")
    if implementation not in {"baseline", "optimized"}:
        raise ValueError("Choose baseline or optimized oracle implementation.")
    rows = [{"mask": mask, "support": 1, "score": score}
            for mask, score in zip(problem["episode_masks"], problem["episode_scores"])]
    constant_support, constant_score = 0, 0
    grouped_count = len(rows)
    support_factor, score_factor = 1, 1
    if implementation == "optimized":
        groups = {}
        for row in rows:
            group = groups.setdefault(row["mask"], {"mask": row["mask"], "support": 0, "score": 0})
            group["support"] += row["support"]
            group["score"] += row["score"]
        grouped_count = len(groups)
        full_mask = (1 << problem["search_qubits"]) - 1
        always = groups.pop(full_mask, {"support": 0, "score": 0})
        constant_support, constant_score = always["support"], always["score"]
        rows = list(groups.values())
        support_factor = math.gcd(*(r["support"] for r in rows)) or 1
        score_factor = math.gcd(*(abs(r["score"]) for r in rows)) or 1
        rows = [{"mask": r["mask"], "support": r["support"] // support_factor,
                 "score": r["score"] // score_factor} for r in rows]
    support_threshold = -(-(problem["min_support"] - constant_support) // support_factor)
    score_threshold = -(-(problem["min_score"] - constant_score) // score_factor)
    offset = sum(-r["score"] for r in rows if r["score"] < 0)
    score_max = offset + sum(r["score"] for r in rows if r["score"] > 0)
    support_bits = max(1, sum(r["support"] for r in rows).bit_length())
    score_bits = max(1, score_max.bit_length())
    flags = 3 if implementation == "baseline" else 1
    return {"implementation": implementation, "groups": rows,
            "original_episode_count": len(problem["episode_masks"]), "grouped_observations": grouped_count,
            "dynamic_groups": len(rows), "constant_support": constant_support, "constant_score": constant_score,
            "support_factor": support_factor, "score_factor": score_factor,
            "support_threshold": support_threshold, "score_threshold": offset + score_threshold,
            "score_offset": offset, "support_bits": support_bits, "score_bits": score_bits,
            "workspace_qubits": support_bits + score_bits + flags,
            "total_qubits": problem["search_qubits"] + support_bits + score_bits + flags,
            "arithmetic": "reversible carry increments" if implementation == "baseline" else "exact Fourier additions; no QFT approximation"}


def with_encoding(problem, implementation):
    """Change the circuit representation, preserving every input and threshold."""
    result = dict(problem)
    plan = encoding_plan(problem, implementation)
    result["implementation"] = implementation
    result["encoding"] = plan
    for key in ("support_bits", "score_bits", "score_offset", "workspace_qubits", "total_qubits"):
        result[key] = plan[key]
    return result


def evaluate_rule(problem, mask):
    if not 0 <= mask < (1 << problem["search_qubits"]):
        raise ValueError("Rule mask is outside the search register.")
    count, score = 0, 0
    for row, value in zip(problem["episode_masks"], problem["episode_scores"]):
        if mask & row == mask:
            count += 1
            score += value
    return count, score, count >= problem["min_support"] and score >= problem["min_score"]


def classical_reference(problem):
    """Exploit conjunction structure via a superset sum, not N independent scans.

    Exhaustive small-case diagnostic only. Its marked count is never needed to
    construct the oracle and is not a free input to an unknown-solution search.
    """
    n = problem["search_qubits"]
    if n > 20:
        raise ValueError("Exhaustive diagnostics are capped at 20 conditions.")
    started = time.perf_counter()
    support = np.zeros(1 << n, dtype=np.int64)
    score = np.zeros(1 << n, dtype=np.int64)
    for mask, value in zip(problem["episode_masks"], problem["episode_scores"]):
        support[mask] += 1
        score[mask] += value
    for bit in range(n):
        width = 1 << bit
        for array in (support, score):
            blocks = array.reshape(-1, 2, width)
            blocks[:, 0, :] += blocks[:, 1, :]
    marked = (support >= problem["min_support"]) & (score >= problem["min_score"])
    return {"support": support, "score": score, "marked": marked,
            "marked_count": int(marked.sum()), "elapsed_seconds": time.perf_counter() - started,
            "method": "Classical superset-sum transform exploiting rule structure, O(episodes + n*2^n)."}


def coverage_reference(problem, *, max_states=100000):
    """Search distinct episode firing sets, merging observationally equal rules.

    At most min(2^conditions, 2^episodes) firing sets exist. Adding many binary
    conditions to very few episodes need not make the problem classically hard.
    A cap prevents this baseline itself allocating an unbounded state table.
    """
    started = time.perf_counter()
    count = len(problem["episode_masks"])
    outcomes = {(1 << count) - 1: 0}
    completed = True
    processed = 0
    for bit in range(problem["search_qubits"]):
        column = sum(((row >> bit) & 1) << i for i, row in enumerate(problem["episode_masks"]))
        for coverage, rule in list(outcomes.items()):
            subset = coverage & column
            if subset not in outcomes:
                if len(outcomes) >= max_states:
                    completed = False
                    break
                outcomes[subset] = rule | (1 << bit)
        processed = bit + 1
        if not completed:
            break
    passing, witness = 0, None
    for coverage, rule in outcomes.items():
        if coverage.bit_count() < problem["min_support"]:
            continue
        score = sum(value for i, value in enumerate(problem["episode_scores"]) if (coverage >> i) & 1)
        if score >= problem["min_score"]:
            passing += 1
            if witness is None:
                witness = {"rule_mask": rule, "support": coverage.bit_count(), "score_units": score}
    return {"method": "Merge rules producing identical episode firing sets; O(conditions * distinct_sets + episodes * distinct_sets).",
            "complete": completed, "conditions_processed": processed,
            "distinct_firing_sets": len(outcomes), "passing_firing_sets_seen": passing,
            "witness": witness, "state_cap": max_states,
            "elapsed_seconds": time.perf_counter() - started,
            "warning": "Firing sets are not equally numerous rule encodings; this is not a marked-rule probability."}


def ideal_search(n, marked_count, iterations):
    """Analytical ideal success, diagnostic use of a known marked count only."""
    if n < 1 or not 0 <= marked_count <= (1 << n) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("Invalid search size, marked count, or iteration count.")
    fraction = marked_count / (1 << n)
    theta = math.asin(math.sqrt(fraction))
    return {
        "iterations": iterations,
        "predicate_queries_with_final_verification": iterations + 1,
        "ideal_grover_success": math.sin((2 * iterations + 1) * theta) ** 2,
        "random_success_same_query_count": -math.expm1((iterations + 1) * math.log1p(-fraction)) if fraction < 1 else 1.,
        "marked_fraction": fraction,
        "warning": "Query comparison only; excludes oracle cost. Known marked count comes from exhaustive diagnostics.",
    }


def scaling_table(bits=(10, 20, 32, 40, 64, 128, 256)):
    """Single marked item, ideal black-box query scaling, no hardware claim."""
    rows = []
    for n in bits:
        if not isinstance(n, int) or not 1 <= n <= 256:
            raise ValueError("Scaling sizes must be integers from 1 to 256.")
        size = 1 << n
        queries = max(1, round(math.pi * math.sqrt(size) / 4))
        rows.append({"search_bits": n, "candidate_count": str(size),
                     "classical_linear_search_mean_queries": f"{size // 2}.5",
                     "ideal_grover_iterations_one_solution": str(queries),
                     "assumptions": "One marked rule, coherent error-free oracle; stronger classical structure excluded."})
    return rows


def _toggle_pattern(circuit, controls, values, target):
    zeros = [q for q, value in zip(controls, values) if not value]
    if zeros:
        circuit.x(zeros)
    if controls:
        circuit.mcx(list(controls), target)
    else:
        circuit.x(target)
    if zeros:
        circuit.x(zeros)


def _controlled_add(circuit, register, control, value):
    """Reversible constant addition modulo register size, without carry ancillas."""
    for start in range(len(register)):
        if not (abs(value) >> start) & 1:
            continue
        if value < 0:  # Inverse of the positive increment circuit.
            circuit.cx(control, register[start])
            for end in range(start + 1, len(register)):
                circuit.mcx([control, *register[start:end]], register[end])
        else:
            for end in range(len(register) - 1, start, -1):
                circuit.mcx([control, *register[start:end]], register[end])
            circuit.cx(control, register[start])


def _at_least_patterns(register, threshold):
    """Disjoint binary prefixes for an unsigned >= comparison."""
    if threshold <= 0:
        yield [], []
        return
    if threshold >= 1 << len(register):
        return
    prefix, values = [], []
    for bit in range(len(register) - 1, -1, -1):
        value = (threshold >> bit) & 1
        if not value:
            yield [*prefix, register[bit]], [*values, 1]
        prefix.append(register[bit])
        values.append(value)
    yield prefix, values


def _toggle_at_least(circuit, register, threshold, flag):
    for controls, values in _at_least_patterns(register, threshold):
        _toggle_pattern(circuit, controls, values, flag)


def _fourier_add(circuit, register, value, control=None):
    for bit, qubit in enumerate(register):
        modulus = 1 << (len(register) - bit)
        residue = value % modulus
        if residue:
            angle = 2 * math.pi * residue / modulus
            if control is None:
                circuit.p(angle, qubit)
            else:
                circuit.cp(angle, control, qubit)


def _phase_pattern(circuit, controls, values):
    zeros = [q for q, value in zip(controls, values) if not value]
    if zeros:
        circuit.x(zeros)
    if len(controls) == 1:
        circuit.z(controls[0])
    else:
        circuit.h(controls[-1])
        circuit.mcx(controls[:-1], controls[-1])
        circuit.h(controls[-1])
    if zeros:
        circuit.x(zeros)


def phase_oracle(problem):
    """Build compute/mark/uncompute from observations, not from known solutions."""
    from qiskit import QuantumCircuit

    plan = encoding_plan(problem)
    n = problem["search_qubits"]
    count = list(range(n, n + plan["support_bits"]))
    score = list(range(count[-1] + 1, count[-1] + 1 + plan["score_bits"]))
    if plan["implementation"] == "optimized":
        from qiskit.circuit.library import QFTGate

        event = score[-1] + 1
        compute = QuantumCircuit(plan["total_qubits"], name="fourier_episode_scores")
        compute.h([*count, *score])  # F|0> for each register.
        _fourier_add(compute, score, plan["score_offset"])
        for row in plan["groups"]:
            controls = [bit for bit in range(n) if not (row["mask"] >> bit) & 1]
            _toggle_pattern(compute, controls, [0] * len(controls), event)
            _fourier_add(compute, count, row["support"], event)
            _fourier_add(compute, score, row["score"], event)
            _toggle_pattern(compute, controls, [0] * len(controls), event)
        compute.append(QFTGate(len(count)).inverse(), count)
        compute.append(QFTGate(len(score)).inverse(), score)
        support_check = QuantumCircuit(plan["total_qubits"])
        _toggle_at_least(support_check, count, plan["support_threshold"], event)
        oracle = QuantumCircuit(plan["total_qubits"], name="optimized_rule_phase_oracle")
        oracle.compose(compute, inplace=True)
        oracle.compose(support_check, inplace=True)
        for controls, values in _at_least_patterns(score, plan["score_threshold"]):
            _phase_pattern(oracle, [event, *controls], [1, *values])
        oracle.compose(support_check.inverse(), inplace=True)
        oracle.compose(compute.inverse(), inplace=True)
        return oracle

    event, support_ok, score_ok = range(score[-1] + 1, score[-1] + 4)
    compute = QuantumCircuit(plan["total_qubits"], name="episode_scores")
    for bit, qubit in enumerate(score):
        if (plan["score_offset"] >> bit) & 1:
            compute.x(qubit)
    for row, value in zip(problem["episode_masks"], problem["episode_scores"]):
        # A rule fires iff none of its selected conditions is false in this row.
        controls = [bit for bit in range(n) if not (row >> bit) & 1]
        _toggle_pattern(compute, controls, [0] * len(controls), event)
        _controlled_add(compute, count, event, 1)
        _controlled_add(compute, score, event, value)
        _toggle_pattern(compute, controls, [0] * len(controls), event)
    compare = QuantumCircuit(plan["total_qubits"], name="thresholds")
    _toggle_at_least(compare, count, plan["support_threshold"], support_ok)
    _toggle_at_least(compare, score, plan["score_threshold"], score_ok)
    oracle = QuantumCircuit(plan["total_qubits"], name="rule_phase_oracle")
    oracle.compose(compute, inplace=True)
    oracle.compose(compare, inplace=True)
    oracle.cz(support_ok, score_ok)
    oracle.compose(compare.inverse(), inplace=True)
    oracle.compose(compute.inverse(), inplace=True)
    return oracle


def grover_circuit(problem, iterations=1, *, measure=True):
    from qiskit import QuantumCircuit

    if not isinstance(iterations, int) or not 0 <= iterations <= 32:
        raise ValueError("Offline circuit export supports 0-32 Grover iterations.")
    n = problem["search_qubits"]
    circuit = QuantumCircuit(encoding_plan(problem)["total_qubits"], n if measure else 0)
    circuit.h(range(n))
    oracle = phase_oracle(problem) if iterations else None
    for _ in range(iterations):
        circuit.compose(oracle, inplace=True)
        circuit.h(range(n))
        circuit.x(range(n))
        if n == 1:
            circuit.z(0)
        else:
            circuit.h(n - 1)
            circuit.mcx(list(range(n - 1)), n - 1)
            circuit.h(n - 1)
        circuit.x(range(n))
        circuit.h(range(n))
    if measure:
        circuit.measure(range(n), range(n))
    return circuit


def compiled_resources(circuit):
    """Deterministic reference synthesis; not IBM backend routing or billing."""
    from qiskit import transpile

    compiled = transpile(circuit, basis_gates=["cx", "rz", "sx", "x"],
                         optimization_level=1, seed_transpiler=42)
    operations = {str(k): int(v) for k, v in compiled.count_ops().items()}
    return {"qubits": compiled.num_qubits, "depth": compiled.depth(),
            "cx_gates": operations.get("cx", 0), "operations": operations,
            "compilation": "All-to-all reference synthesis into cx/rz/sx/x; no real backend routing, calibration, or error correction."}
