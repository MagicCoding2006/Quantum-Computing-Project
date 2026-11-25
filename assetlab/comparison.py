"""Compare measured bitstrings against the actual ideal circuit distribution."""

from math import comb

import numpy as np

from .selection import selected_assets, solve_exact


def measurement_comparison(problem, simulation, hardware):
    n, k = len(problem["tickers"]), problem["choose"]
    ideal = np.asarray(simulation["probabilities"], dtype=float)
    if len(ideal) != 1 << n or not np.isfinite(ideal).all() or (ideal < 0).any() or not np.isclose(ideal.sum(), 1):
        raise ValueError("The ideal distribution does not match the recorded problem.")
    counts = np.zeros(1 << n, dtype=np.int64)
    for row in hardware["histogram"]:
        bits, count = row["bitstring"], row["count"]
        if len(bits) != n or set(bits) - {"0", "1"} or not isinstance(count, int) or count < 0:
            raise ValueError("Invalid hardware bitstring or count.")
        counts[int(bits, 2)] += count
    shots = int(counts.sum())
    if shots <= 0 or shots != hardware["shots"]:
        raise ValueError("Hardware counts do not sum to the recorded shot count.")
    observed = counts / shots
    exact = solve_exact(problem)
    optimum = exact[0]["train_annual_variance"]
    feasible_lookup = {row["mask"]: row for row in exact}
    optimal = {row["mask"] for row in exact
               if np.isclose(row["train_annual_variance"], optimum, atol=1e-12, rtol=1e-8)}
    rows = []
    cardinality = [{"selected": i, "ideal": 0.0, "hardware": 0.0, "uniform_legal": float(i == k)} for i in range(n + 1)]
    for mask in range(1 << n):
        solution = feasible_lookup.get(mask)
        variance = solution["train_annual_variance"] if solution else None
        cardinality[mask.bit_count()]["ideal"] += float(ideal[mask])
        cardinality[mask.bit_count()]["hardware"] += float(observed[mask])
        rows.append({"mask": mask, "bitstring": format(mask, f"0{n}b"),
                     "assets": selected_assets(mask, problem["tickers"]),
                     "count": int(counts[mask]), "hardware": float(observed[mask]),
                     "ideal": float(ideal[mask]), "feasible": solution is not None,
                     "optimal": mask in optimal, "train_variance": variance,
                     "gap_percent": 100 * max(0.0, variance - optimum) / optimum
                     if variance is not None and optimum > 1e-15 else None,
                     "uniform_legal": 1 / comb(n, k) if solution else 0.0})
    feasible_masks = list(feasible_lookup)
    feasible_shots = int(counts[feasible_masks].sum())
    optimal_shots = int(counts[list(optimal)].sum())
    weighted_variance = sum(counts[m] * feasible_lookup[m]["train_annual_variance"] for m in feasible_masks)
    return {
        "rows": rows, "cardinality": cardinality, "shots": shots,
        "feasible_fraction": feasible_shots / shots, "optimal_fraction": optimal_shots / shots,
        "ideal_feasible_probability": float(ideal[feasible_masks].sum()),
        "ideal_optimal_probability": float(ideal[list(optimal)].sum()),
        "uniform_legal_optimal_probability": len(optimal) / comb(n, k),
        "total_variation_distance": float(np.abs(observed - ideal).sum() / 2),
        "feasible_mean_variance": float(weighted_variance / feasible_shots) if feasible_shots else None,
        "feasible_mean_gap_percent": float(100 * max(0.0, weighted_variance / feasible_shots - optimum) / optimum)
        if feasible_shots and optimum > 1e-15 else None,
    }
