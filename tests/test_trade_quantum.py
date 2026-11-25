from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from assetlab.quantum import qiskit_circuit, statevector
from assetlab.selection import binary_states, feasible_mask, problem_energies, solve_exact, summarize_counts
from assetlab.trade_quantum import build_trade_problem, candidate_input, create_trade_quantum
from assetlab.trading import TradingConfig
from assetlab.trading_io import run_trading
from test_trading import synthetic_relationships


def problem(means=(0.02, 0.03), capacity=2):
    return build_trade_problem([[0.001, 0.0002], [0.0002, 0.002]], ["A", "B"], means,
                               capacity=capacity, fraction=0.2, risk_aversion=5, horizon=10)


def fake_candidates(prices, universe, config, as_of, **kwargs):
    rows = [{"ticker": ticker, "z": -2.5, "peers": ["C"], "half_life": 5,
             "model_gross_return": gross, "assumed_round_trip_cost": .002,
             "model_net_return": gross - .002} for ticker, gross in [("A", .03), ("B", .04)]]
    return rows, np.diag([.001, .002]), {"as_of": str(as_of.date()),
                    "training_end": str(prices.index[125].date()), "market_filter_allows_entries": True,
                    "available_history_rows": len(prices.loc[:as_of])}


class TradeObjectiveTests(unittest.TestCase):
    def test_qubo_matches_raw_utility_and_weighted_capacity_on_every_state(self):
        p = problem()
        x = binary_states(len(p["tickers"]))
        real = x[:, :2]
        slack = x[:, 2] + 2 * x[:, 3]
        raw = 5 * .2 ** 2 * np.einsum("bi,ij,bj->b", real, p["horizon_covariance"], real)
        raw -= .2 * (real @ p["expected_net_returns"])
        expected = raw / p["objective_scale"] + p["penalty"] * (real.sum(axis=1) + slack - 2) ** 2
        np.testing.assert_allclose(problem_energies(p), expected, atol=1e-10)
        self.assertTrue(feasible_mask(int(np.argmin(expected)), p))
        feasible = [i for i in range(len(x)) if feasible_mask(i, p)]
        self.assertEqual(len(feasible), 4)
        subsets = [tuple(t["assets"]) for t in solve_exact(p)]
        self.assertEqual(len(set(subsets)), 4)
        self.assertIn((), subsets)

    def test_negative_net_scores_choose_cash_and_slack_does_not_become_an_asset(self):
        p = problem(means=(-.01, -.02))
        best = solve_exact(p)[0]
        self.assertEqual(best["assets"], [])
        self.assertAlmostEqual(best["objective_cost"], 0)
        counts = summarize_counts({best["mask"]: 100}, p)
        self.assertEqual(counts["selected_assets"], [])
        self.assertEqual(counts["feasible_fraction"], 1)
        self.assertEqual(counts["optimal_fraction"], 1)
        self.assertFalse(feasible_mask(0, p))  # Cash has a valid unused-slot encoding.

    def test_capacity_one_selects_best_model_score(self):
        p = problem(capacity=1)
        self.assertEqual(solve_exact(p)[0]["assets"], ["B"])
        self.assertFalse(feasible_mask(3, p))

    @unittest.skipUnless(importlib.util.find_spec("qiskit"), "optional Qiskit dependency")
    def test_trade_circuit_matches_simulator_with_binary_cash_bits(self):
        from qiskit.quantum_info import Statevector

        p = problem()
        angles = [.7, .3, .9, .5]
        actual = Statevector.from_instruction(qiskit_circuit(p, angles, measure=False)).probabilities()
        np.testing.assert_allclose(actual, np.abs(statevector(p, angles)) ** 2, atol=1e-12)

    def test_future_observations_cannot_change_a_saved_signal_model(self):
        prices = synthetic_relationships()
        cfg = TradingConfig(train_sessions=126, test_sessions=63, trend_filter=False)
        cache = {}
        for date in prices.index[126:-1]:
            first = candidate_input(prices, ["A", "B", "C"], cfg, date, model_cache=cache)
            if first[0]:
                break
        self.assertTrue(first[0])
        changed = prices.copy()
        changed.loc[changed.index > date, "A"] *= 10
        second = candidate_input(changed, ["A", "B", "C"], cfg, date)
        self.assertEqual(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        self.assertEqual(first[2], second[2])


class TradeQuantumWorkflowTests(unittest.TestCase):
    def make_run(self, temp):
        prices = synthetic_relationships()
        source = Path(temp) / "input.csv"
        prices.to_csv(source)
        config = TradingConfig(train_sessions=126, test_sessions=63, trend_filter=False, max_positions=3)
        path, _ = run_trading(["A", "B", "C"], config, start="2020-01-01", end="2022-01-01", csv_path=source, out=temp)
        return path

    def test_no_entry_signals_skip_quantum_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = self.make_run(temp)
            empty = ([], np.empty((0, 0)), {"as_of": "2021-07-13", "market_filter_allows_entries": True})
            with patch("assetlab.trade_quantum.candidate_input", return_value=empty), patch("assetlab.trade_quantum.simulate") as simulate:
                folder, result = create_trade_quantum(parent)
                self.assertIsNone(folder)
                self.assertIsNone(result)
                simulate.assert_not_called()
            self.assertEqual(json.loads((parent / "quantum" / "latest.json").read_text())["status"], "no_candidates")

    @unittest.skipUnless(importlib.util.find_spec("qiskit_ibm_runtime"), "optional IBM dependency")
    def test_trade_case_uses_existing_ibm_prepare_submit_collect_and_parent_refresh(self):
        import ibm_quantum
        from qiskit.providers.fake_provider import GenericBackendV2

        with tempfile.TemporaryDirectory() as temp:
            parent = self.make_run(temp)
            original_backtest = (parent / "trading.json").read_bytes()
            with patch("assetlab.trade_quantum.candidate_input", side_effect=fake_candidates):
                folder, data = create_trade_quantum(parent, example=True, shots=256)
            self.assertEqual(data["snapshot"]["mode"], "historical_example")
            self.assertEqual(data["snapshot"]["as_of"], "2020-06-24")
            ibm_quantum.prepare(folder)
            backend = GenericBackendV2(num_qubits=5, seed=42)
            with patch("ibm_quantum.service", return_value=SimpleNamespace(backend=lambda _: backend)), \
                 patch("qiskit_ibm_runtime.SamplerV2") as sampler:
                sampler.return_value.run.return_value.job_id.return_value = "test-trade-job"
                ibm_quantum.submit(folder, "fake_backend", 256, 60)
            job = SimpleNamespace(status=lambda: "DONE", result=lambda: [SimpleNamespace(data=SimpleNamespace(
                meas=SimpleNamespace(get_counts=lambda: {"0101": 10, "1000": 5})))])
            with patch("ibm_quantum.service", return_value=SimpleNamespace(job=lambda _: job)):
                ibm_quantum.collect(folder)
            hardware = json.loads((folder / "ibm" / "hardware_results.json").read_text())
            self.assertEqual(hardware["feasible_fraction"], 1)
            self.assertEqual(hardware["selected_assets"], ["A"])
            summary = json.loads((parent / "quantum" / "latest.json").read_text())
            self.assertEqual(summary["hardware"]["job_id"], "test-trade-job")
            self.assertEqual((parent / "trading.json").read_bytes(), original_backtest)
            self.assertIn('"hardware"', (folder / "report.html").read_text())


if __name__ == "__main__":
    unittest.main()
