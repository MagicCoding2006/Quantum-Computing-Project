"""Small ideal QAOA statevector simulator, plus the equivalent Qiskit circuit.

The simulator applies actual cost phases and X rotations; it is not a proxy
optimizer. Circuit parameters are optimized classically on training costs.
"""

import time

import numpy as np
from scipy.optimize import differential_evolution

from .selection import problem_energies, summarize_counts


def statevector(problem, parameters):
    n = len(problem["tickers"])
    if len(parameters) == 0 or len(parameters) % 2:
        raise ValueError("Parameters must be [gamma_0, beta_0, ...].")
    energy = problem_energies(problem) / problem["phase_scale"]
    state = np.ones(1 << n, dtype=complex) / np.sqrt(1 << n)
    for gamma, beta in np.asarray(parameters).reshape(-1, 2):
        state *= np.exp(-1j * gamma * energy)
        for qubit in range(n):
            blocks = state.reshape(-1, 2, 1 << qubit)
            zero, one = blocks[:, 0, :].copy(), blocks[:, 1, :].copy()
            blocks[:, 0, :] = np.cos(beta) * zero - 1j * np.sin(beta) * one
            blocks[:, 1, :] = -1j * np.sin(beta) * zero + np.cos(beta) * one
    return state


def simulate(problem, *, layers=1, shots=4096, seed=42):
    if not 1 <= layers <= 3 or not 1 <= shots <= 100000:
        raise ValueError("Use 1-3 QAOA layers and 1-100000 shots.")
    started = time.perf_counter()
    energy = problem_energies(problem) / problem["phase_scale"]

    def expectation(parameters):
        return float(np.abs(statevector(problem, parameters)) ** 2 @ energy)

    fit = differential_evolution(
        expectation, [(0, 2 * np.pi), (0, np.pi)] * layers,
        seed=seed, maxiter=60, popsize=8, polish=True, tol=1e-7,
    )
    probabilities = np.abs(statevector(problem, fit.x)) ** 2
    probabilities /= probabilities.sum()
    samples = np.random.default_rng(seed).multinomial(shots, probabilities)
    counts = {i: int(c) for i, c in enumerate(samples) if c}
    result = summarize_counts(counts, problem)
    result.update(
        backend="ideal_statevector", layers=layers, seed=seed,
        parameters=fit.x.tolist(), parameter_order="gamma_0,beta_0,gamma_1,beta_1,...",
        expected_scaled_energy=float(fit.fun), optimizer_evaluations=int(fit.nfev),
        optimizer_converged=bool(fit.success), optimizer_message=str(fit.message),
        elapsed_seconds=time.perf_counter() - started,
        probabilities=probabilities.tolist(),
    )
    return result


def qiskit_circuit(problem, parameters, *, measure=True):
    from qiskit import QuantumCircuit

    n = len(problem["tickers"])
    circuit = QuantumCircuit(n)
    circuit.h(range(n))
    scale = problem["phase_scale"]
    for gamma, beta in np.asarray(parameters).reshape(-1, 2):
        for i, h in enumerate(problem["ising_h"]):
            if abs(h) > 1e-14:
                circuit.rz(2 * gamma * h / scale, i)
        for i in range(n):
            for j in range(i + 1, n):
                coupling = problem["ising_J_upper"][i][j]
                if abs(coupling) > 1e-14:
                    circuit.rzz(2 * gamma * coupling / scale, i, j)
        circuit.rx(2 * beta, range(n))
    if measure:
        circuit.measure_all()
    return circuit
