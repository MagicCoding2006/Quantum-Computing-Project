"""Exact fixed-cardinality selection, QUBO, and equivalent Ising coefficients."""

from itertools import combinations

import numpy as np


def binary_states(n):
    return ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(float)


def build_problem(covariance, tickers, choose):
    n = len(tickers)
    covariance = np.asarray(covariance, dtype=float)
    if not 2 <= n <= 10 or not 1 <= choose < n:
        raise ValueError("Use 2-10 candidate assets, and choose between 1 and n-1 assets.")
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
                if int(mask).bit_count() == k}
    total = sum(int(v) for v in counts.values())
    if not total:
        raise ValueError("No measured shots were returned.")
    exact = solve_exact(problem)
    variance = np.array([r["train_annual_variance"] for r in exact])
    optimum = {r["mask"] for r, v in zip(exact, variance)
               if np.isclose(v, variance.min(), atol=1e-12, rtol=1e-8)}
    best = min(feasible, key=lambda m: (energies[m], m)) if feasible else None
    return {
        "shots": total, "feasible_shots": sum(feasible.values()),
        "feasible_fraction": sum(feasible.values()) / total,
        "optimal_fraction": sum(int(counts.get(m, 0)) for m in optimum) / total,
        "uniform_feasible_probability": len(exact) / (1 << n),
        "uniform_optimal_probability": len(optimum) / (1 << n),
        "best_feasible_mask": best,
        "selected_assets": selected_assets(best, problem["tickers"]) if best is not None else [],
        "selection_rule": "Lowest training objective among measured feasible bitstrings; no holdout inputs",
        "histogram": [
            {"bitstring": format(m, f"0{n}b"), "count": int(c),
             "feasible": m.bit_count() == k, "optimal": m in optimum,
             "assets": selected_assets(m, problem["tickers"])}
            for m, c in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    }
