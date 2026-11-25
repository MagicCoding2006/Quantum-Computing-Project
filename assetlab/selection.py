"""Exact fixed-cardinality selection, QUBO, and equivalent Ising coefficients."""

from itertools import combinations

import numpy as np

MAX_CANDIDATES = 16


def encode_qubo(qubo, offset):
    """Upper-triangular QUBO to the shared circuit's Ising convention."""
    qubo = np.asarray(qubo, dtype=float)
    diagonal = np.diag(qubo)
    upper = np.triu(qubo, 1)
    h = -diagonal / 2 - (upper.sum(axis=0) + upper.sum(axis=1)) / 4
    states = binary_states(len(qubo))
    energies = np.einsum("bi,ij,bj->b", states, qubo, states) + offset
    return {"qubo_upper": qubo.tolist(), "qubo_offset": float(offset),
            "ising_h": h.tolist(), "ising_J_upper": (upper / 4).tolist(),
            "ising_offset": float(offset + diagonal.sum() / 2 + upper.sum() / 4),
            "phase_scale": max(float(energies.std()), 1e-12)}


def feasible_mask(mask, problem):
    if problem.get("kind") == "trade_selection":
        n = len(problem["decision_tickers"])
        selected = (mask & ((1 << n) - 1)).bit_count()
        return selected + (mask >> n) == problem["choose"]
    return mask.bit_count() == problem["choose"]


def decode_assets(mask, problem):
    return selected_assets(mask, problem.get("decision_tickers", problem["tickers"]))


def solution_objective(row):
    return row.get("objective_cost", row["train_annual_variance"])


def binary_states(n):
    return ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(float)


def build_problem(covariance, tickers, choose):
    n = len(tickers)
    covariance = np.asarray(covariance, dtype=float)
    if not 2 <= n <= MAX_CANDIDATES or not 1 <= choose < n:
        raise ValueError(f"Use 2-{MAX_CANDIDATES} candidate assets, and choose between 1 and n-1 assets.")
    if covariance.shape != (n, n) or not np.isfinite(covariance).all():
        raise ValueError("Covariance must be a finite n-by-n matrix.")
    if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=1e-10):
        raise ValueError("Covariance must be symmetric.")
    if np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError("Covariance must be positive semidefinite.")
    cov_scale = float(np.diag(covariance).max())
    if cov_scale <= 0:
        raise ValueError("Covariance must contain positive variance.")
    risk = covariance / cov_scale / choose ** 2
    # The penalty exceeds the absolute upper bound on risk for any bitstring.
    # Hence an infeasible bitstring cannot be the ground state.
    penalty = float(np.abs(risk).sum() + 1)
    diagonal = np.diag(risk) + penalty * (1 - 2 * choose)
    upper = np.triu(2 * risk + 2 * penalty, 1)
    qubo = upper + np.diag(diagonal)
    offset = penalty * choose ** 2
    h = -diagonal / 2 - (upper.sum(axis=0) + upper.sum(axis=1)) / 4
    coupling = upper / 4
    ising_offset = float(offset + diagonal.sum() / 2 + upper.sum() / 4)
    states = binary_states(n)
    energies = np.einsum("bi,ij,bj->b", states, qubo, states) + offset
    phase_scale = max(float(energies.std()), 1e-12)
    return {
        "schema_version": 1, "tickers": list(tickers), "choose": choose,
        "objective": "Minimum training covariance risk, fixed equal weights, exactly k assets",
        "annual_covariance": covariance.tolist(), "covariance_scale": cov_scale,
        "penalty": penalty, "qubo_upper": qubo.tolist(), "qubo_offset": float(offset),
        "ising_h": h.tolist(), "ising_J_upper": coupling.tolist(),
        "ising_offset": ising_offset, "phase_scale": phase_scale,
        "bit_order": "ticker i = qubit i = bit i of integer; display strings are q[n-1]...q[0]",
        "ising_convention": "x_i=(1-z_i)/2; E=offset+sum(h_i*z_i)+sum(i<j,J_ij*z_i*z_j)",
    }


def problem_energies(problem):
    states = binary_states(len(problem["tickers"]))
    return np.einsum("bi,ij,bj->b", states, np.asarray(problem["qubo_upper"]), states) + problem["qubo_offset"]


def selected_assets(mask, tickers):
    return [t for i, t in enumerate(tickers) if mask & (1 << i)]


def solve_exact(problem):
    if problem.get("kind") == "trade_selection":
        covariance = np.asarray(problem["horizon_covariance"])
        means = np.asarray(problem["expected_net_returns"])
        names, capacity = problem["decision_tickers"], problem["choose"]
        weight, aversion = problem["position_fraction"], problem["risk_aversion"]
        solutions = []
        for size in range(capacity + 1):
            for indices in combinations(range(len(names)), size):
                variance = float(covariance[np.ix_(indices, indices)].sum() * weight ** 2) if indices else 0.0
                expected_return = float(means[list(indices)].sum() * weight) if indices else 0.0
                mask = sum(1 << i for i in indices) | ((capacity - size) << len(names))
                annual_variance = variance * 252 / problem["horizon_sessions"]
                solutions.append({"mask": mask, "assets": [names[i] for i in indices],
                                  "objective_cost": aversion * variance - expected_return,
                                  "model_net_return": expected_return, "horizon_variance": variance,
                                  "train_annual_variance": annual_variance,
                                  "train_annual_volatility": float(np.sqrt(max(0.0, annual_variance)))})
        return sorted(solutions, key=lambda row: (row["objective_cost"], row["mask"]))
    tickers, k = problem["tickers"], problem["choose"]
    covariance = np.asarray(problem["annual_covariance"])
    solutions = []
    for indices in combinations(range(len(tickers)), k):
        variance = float(covariance[np.ix_(indices, indices)].sum() / k ** 2)
        solutions.append({
            "mask": sum(1 << i for i in indices), "assets": [tickers[i] for i in indices],
            "train_annual_variance": variance,
            "train_annual_volatility": float(np.sqrt(max(variance, 0))),
        })
    return sorted(solutions, key=lambda r: (r["train_annual_variance"], r["mask"]))


def summarize_counts(counts, problem):
    n, k = len(problem["tickers"]), problem["choose"]
    energies = problem_energies(problem)
    feasible = {int(mask): int(count) for mask, count in counts.items()
                if feasible_mask(int(mask), problem)}
    total = sum(int(v) for v in counts.values())
    if not total:
        raise ValueError("No measured shots were returned.")
    exact = solve_exact(problem)
    costs = np.array([solution_objective(r) for r in exact])
    optimum = {r["mask"] for r, value in zip(exact, costs)
               if np.isclose(value, costs.min(), atol=1e-12, rtol=1e-8)}
    best = min(feasible, key=lambda m: (energies[m], m)) if feasible else None
    return {
        "shots": total, "feasible_shots": sum(feasible.values()),
        "feasible_fraction": sum(feasible.values()) / total,
        "optimal_fraction": sum(int(counts.get(m, 0)) for m in optimum) / total,
        "uniform_feasible_probability": len(exact) / (1 << n),
        "uniform_optimal_probability": len(optimum) / (1 << n),
        "best_feasible_mask": best,
        "selected_assets": decode_assets(best, problem) if best is not None else [],
        "selection_rule": "Lowest recorded trade objective among measured feasible encodings; signal-date information only"
        if problem.get("kind") == "trade_selection" else "Lowest training objective among measured feasible bitstrings; no holdout inputs",
        "histogram": [
            {"bitstring": format(m, f"0{n}b"), "count": int(c),
             "feasible": feasible_mask(m, problem), "optimal": m in optimum,
             "assets": decode_assets(m, problem)}
            for m, c in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    }
