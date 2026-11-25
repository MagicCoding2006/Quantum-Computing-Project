import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from assetlab.analysis import analyze, portfolio_metrics
from assetlab.data import align_returns, demo_prices, load_prices, ticker_list, validate_prices
from assetlab.pipeline import run_study
from assetlab.quantum import qiskit_circuit, simulate, statevector
from assetlab.selection import binary_states, build_problem, problem_energies, solve_exact, summarize_counts
from assetlab.comparison import measurement_comparison
from assetlab.report import refresh_report


def example_problem(n=4, choose=2):
    rng = np.random.default_rng(71)
    factor = rng.normal(size=(n, n))
    covariance = factor @ factor.T * 0.003
    return build_problem(covariance, [f"ASSET{i}" for i in range(n)], choose)


def config():
    return {"tickers": ["SPY", "QQQ", "TLT", "GLD"],
            "candidates": ["SPY", "QQQ", "TLT", "GLD"], "choose": 2,
            "start": "2020-01-01", "end": "2023-01-01", "train_fraction": 0.7,
            "windows": [60, 120, 252], "layers": 1, "shots": 512, "seed": 42, "demo": True}


class SelectionTests(unittest.TestCase):
    def test_expanded_candidate_limit_and_combination_count(self):
        p = build_problem(np.eye(16), [f"A{i}" for i in range(16)], 8)
        self.assertEqual(len(solve_exact(p)), 12870)
        with self.assertRaisesRegex(ValueError, "2-16"):
            build_problem(np.eye(17), [f"A{i}" for i in range(17)], 8)

    def test_qubo_and_ising_match_original_objective_for_every_state(self):
        for n in (3, 5, 6):
            for choose in (1, n - 1):
                p = example_problem(n, choose)
                x = binary_states(n)
                covariance = np.asarray(p["annual_covariance"])
                original = np.einsum("bi,ij,bj->b", x, covariance, x) / p["covariance_scale"] / choose ** 2
                original += p["penalty"] * (x.sum(axis=1) - choose) ** 2
                z = 1 - 2 * x
                ising = p["ising_offset"] + z @ p["ising_h"] + np.einsum(
                    "bi,ij,bj->b", z, np.asarray(p["ising_J_upper"]), z)
                np.testing.assert_allclose(problem_energies(p), original, atol=1e-11)
                np.testing.assert_allclose(ising, original, atol=1e-11)
                minimum = int(np.argmin(original))
                self.assertEqual(minimum.bit_count(), choose)

    def test_exact_solution_is_covariance_minimum_and_bit_order_is_consistent(self):
        p = build_problem(np.diag([0.01, 0.04, 0.09, 0.16]), ["A", "B", "C", "D"], 2)
        best = solve_exact(p)[0]
        self.assertEqual(best["assets"], ["A", "B"])
        self.assertEqual(best["mask"], 3)
        self.assertAlmostEqual(best["train_annual_variance"], 0.0125)
        result = summarize_counts({3: 5, 5: 3, 15: 2}, p)
        self.assertAlmostEqual(result["feasible_fraction"], 0.8)
        self.assertAlmostEqual(result["optimal_fraction"], 0.5)
        self.assertEqual(result["histogram"][0]["bitstring"], "0011")
        self.assertEqual(result["selected_assets"], ["A", "B"])

    def test_no_feasible_measurement_is_reported_without_inventing_selection(self):
        result = summarize_counts({0: 2, 15: 3}, example_problem())
        self.assertEqual(result["selected_assets"], [])
        self.assertIsNone(result["best_feasible_mask"])
        self.assertEqual(result["feasible_fraction"], 0)

    def test_non_psd_covariance_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            build_problem([[1, 2], [2, 1]], ["A", "B"], 1)


class DataTests(unittest.TestCase):
    def test_missing_prices_do_not_bridge_into_daily_returns(self):
        dates = pd.bdate_range("2024-01-01", periods=8)
        prices = pd.DataFrame({"A": [100, 102, np.nan, 108, 107, 112, 110, 115],
                               "B": [100, 101, 104, 103, 108, 107, 111, 110]}, index=dates)
        returns, _ = align_returns(prices, minimum=1)
        self.assertNotIn(dates[2], returns.index)
        self.assertNotIn(dates[3], returns.index)
        self.assertAlmostEqual(returns.loc[dates[4], "A"], 107 / 108 - 1)

    def test_bad_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            ticker_list(["SPY", "SPY"])
        with self.assertRaises(ValueError):
            ticker_list(["SPY", "../secret"])
        for prices in [pd.DataFrame({"A": [0], "B": [1]}, index=["2024-01-01"]),
                       pd.DataFrame({"A": [1, 2], "B": [1, 3]}, index=["2024-01-01"] * 2)]:
            with self.assertRaises(ValueError):
                validate_prices(prices, ["A", "B"])

    def test_demo_is_reproducible_independent_of_ticker_order(self):
        left = demo_prices(["SPY", "GLD"], "2020-01-01", "2021-01-01", 42)
        right = demo_prices(["GLD", "SPY"], "2020-01-01", "2021-01-01", 42)
        pd.testing.assert_frame_equal(left, right[left.columns])

    def test_csv_duplicate_headers_are_rejected_before_parser_mangling(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prices.csv"
            path.write_text("date,SPY,SPY,GLD\n2024-01-01,100,999,100\n")
            with self.assertRaisesRegex(ValueError, "distinct ticker columns"):
                load_prices(["SPY", "GLD"], "2024-01-01", "2025-01-01", csv_path=path)

    def test_drawdown_includes_initial_capital(self):
        metrics = portfolio_metrics(pd.DataFrame({"A": [-0.1, 0.01]}), ["A"])
        self.assertAlmostEqual(metrics["max_drawdown"], -0.1)


class QuantumTests(unittest.TestCase):
    def test_statevector_normalization_and_uniform_zero_angles(self):
        p = example_problem()
        for parameters in ([0, 0], [1.1, 0.4], [0.6, 0.3, 1.3, 0.8]):
            self.assertAlmostEqual(float(np.abs(statevector(p, parameters)) ** 2 @ np.ones(16)), 1)
        np.testing.assert_allclose(np.abs(statevector(p, [0, 0])) ** 2, np.full(16, 1 / 16))

    @unittest.skipUnless(importlib.util.find_spec("qiskit"), "optional Qiskit dependency not installed")
    def test_qiskit_circuit_matches_numpy_simulator_with_multiple_layers(self):
        from qiskit.quantum_info import Statevector

        p = example_problem(5, 2)
        for parameters in ([0.7, 0.3], [0.7, 0.3, 1.9, 0.8]):
            qiskit = Statevector.from_instruction(qiskit_circuit(p, parameters, measure=False))
            expected = np.abs(statevector(p, parameters)) ** 2
            np.testing.assert_allclose(qiskit.probabilities(), expected, atol=1e-12)

    def test_fit_samples_are_seeded_and_probabilities_normalized(self):
        p = example_problem()
        a = simulate(p, shots=512, seed=42)
        b = simulate(p, shots=512, seed=42)
        self.assertEqual(a["histogram"], b["histogram"])
        self.assertAlmostEqual(sum(a["probabilities"]), 1)
        self.assertEqual(sum(row["count"] for row in a["histogram"]), 512)


class PipelineTests(unittest.TestCase):
    def test_distribution_comparison_includes_invalid_shots_and_stronger_baseline(self):
        p = build_problem(np.diag([1.0, 2.0]), ["A", "B"], 1)
        q = {"probabilities": [0.25] * 4}
        h = {"shots": 8, "histogram": [{"bitstring": "01", "count": 4}, {"bitstring": "10", "count": 4}]}
        c = measurement_comparison(p, q, h)
        self.assertEqual(c["feasible_fraction"], 1)
        self.assertEqual(c["optimal_fraction"], 0.5)
        self.assertEqual(c["ideal_feasible_probability"], 0.5)
        self.assertEqual(c["uniform_legal_optimal_probability"], 0.5)
        self.assertEqual(c["total_variation_distance"], 0.5)
        self.assertAlmostEqual(c["feasible_mean_gap_percent"], 50)
        self.assertEqual(c["cardinality"][1]["hardware"], 1)
        h["histogram"][1] = {"bitstring": "00", "count": 4}
        self.assertEqual(measurement_comparison(p, q, h)["feasible_fraction"], 0.5)
        h["shots"] = 9
        with self.assertRaisesRegex(ValueError, "shot count"):
            measurement_comparison(p, q, h)

    def test_local_benchmark_exports_cases_and_retains_parent_analysis(self):
        from assetlab.benchmark import run_benchmark

        with tempfile.TemporaryDirectory() as temp:
            output, _ = run_study(config(), out=temp)
            before = (output / "analysis.json").read_bytes()
            folder, b = run_benchmark(output, sizes=[4], layers=[1, 2], seeds=[42], shots=256)
            self.assertEqual(len(b["rows"]), 2)
            self.assertTrue((folder / "results.csv").exists())
            self.assertEqual((output / "analysis.json").read_bytes(), before)
            for row in b["rows"]:
                self.assertFalse(row["hardware_executed"])
                self.assertEqual(row["legal_portfolios"], 6)
                self.assertTrue((output / row["case_run"] / "problem.json").exists())
                self.assertAlmostEqual(row["random_legal"]["expected_optimal_probability"], 1 / 6)
            self.assertIn('"benchmark"', (output / "report.html").read_text())

    def test_classical_greedy_swap_baseline_finds_known_diagonal_optimum(self):
        from assetlab.benchmark import classical_baselines

        p = build_problem(np.diag([0.01, 0.04, 0.09, 0.16]), ["A", "B", "C", "D"], 2)
        result = classical_baselines(p, solve_exact(p), shots=128, seed=42)
        self.assertEqual(result["greedy_swap"]["assets"], ["A", "B"])
        self.assertAlmostEqual(result["greedy_swap"]["best_gap_percent"], 0)

    def test_holdout_changes_cannot_change_optimization_inputs(self):
        c = config()
        prices = demo_prices(c["tickers"], c["start"], c["end"], c["seed"])
        original, _ = analyze(prices, {"source": "synthetic"}, c)
        changed = prices.copy()
        first_test_price = 1 + int((len(prices) - 1) * c["train_fraction"])
        changed.iloc[first_test_price:, 0] *= np.exp(np.linspace(0.1, 1.0, len(changed) - first_test_price))
        altered, _ = analyze(changed, {"source": "synthetic"}, c)
        self.assertEqual(original["problem"], altered["problem"])
        self.assertEqual(original["exact"], altered["exact"])
        self.assertEqual(original["quantum"]["parameters"], altered["quantum"]["parameters"])
        self.assertNotEqual(original["portfolios"]["All candidates"]["total_return"],
                            altered["portfolios"]["All candidates"]["total_return"])

    def test_complete_run_artifacts_and_csv_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            output, result = run_study(config(), out=temp)
            manifest = json.loads((output / "manifest.json").read_text())
            for name, digest in manifest["files_sha256"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            self.assertIn("SYNTHETIC DEMO", (output / "report.html").read_text())
            self.assertNotIn("__REPORT_DATA__", (output / "report.html").read_text())
            c = {**config(), "demo": False}
            _, replay = run_study(c, csv_path=output / "prices.csv", out=temp)
            self.assertEqual(result["exact"][0]["assets"], replay["exact"][0]["assets"])
            np.testing.assert_allclose(result["problem"]["annual_covariance"],
                                       replay["problem"]["annual_covariance"], atol=1e-12)


@unittest.skipUnless(importlib.util.find_spec("qiskit_ibm_runtime"), "optional IBM dependency not installed")
class IBMTests(unittest.TestCase):
    def test_missing_credentials_have_actionable_error_without_exposing_values(self):
        from qiskit_ibm_runtime.accounts.exceptions import AccountNotFoundError
        import ibm_quantum

        with patch.dict("os.environ", {"IBM_QUANTUM_API_KEY": "", "IBM_QUANTUM_INSTANCE": ""}), \
             patch("qiskit_ibm_runtime.QiskitRuntimeService", side_effect=AccountNotFoundError("missing")):
            with self.assertRaisesRegex(ValueError, "python ibm_quantum.py login"):
                ibm_quantum.service()

    def test_login_saves_credentials_without_connecting_or_printing_key(self):
        from contextlib import redirect_stdout
        from io import StringIO
        import ibm_quantum

        output = StringIO()
        with patch("ibm_quantum.getpass", return_value="example-secret"), \
             patch("builtins.input", return_value="crn:example:open-instance"), \
             patch("qiskit_ibm_runtime.QiskitRuntimeService") as runtime, redirect_stdout(output):
            self.assertEqual(ibm_quantum.main(["login"]), 0)
            runtime.save_account.assert_called_once_with(
                channel="ibm_quantum_platform", token="example-secret",
                instance="crn:example:open-instance", set_as_default=True)
            runtime.assert_not_called()
        self.assertNotIn("example-secret", output.getvalue())

    def test_login_rejects_instance_name_and_preserves_existing_credentials(self):
        from qiskit_ibm_runtime.accounts.exceptions import AccountAlreadyExistsError
        import ibm_quantum

        with patch("ibm_quantum.getpass", return_value="example-secret"), \
             patch("builtins.input", return_value="open-instance"), \
             patch("qiskit_ibm_runtime.QiskitRuntimeService") as runtime:
            with self.assertRaisesRegex(ValueError, "full instance CRN"):
                ibm_quantum.login()
            runtime.save_account.assert_not_called()
        with patch("ibm_quantum.getpass", return_value="example-secret"), \
             patch("builtins.input", return_value="crn:example:open-instance"), \
             patch("qiskit_ibm_runtime.QiskitRuntimeService") as runtime:
            runtime.save_account.side_effect = AccountAlreadyExistsError("exists")
            with self.assertRaisesRegex(ValueError, "existing credentials were kept"):
                ibm_quantum.login()

    def test_offline_preparation_and_duplicate_submission_protection(self):
        from qiskit.providers.fake_provider import GenericBackendV2
        from types import SimpleNamespace
        import ibm_quantum

        with tempfile.TemporaryDirectory() as temp:
            output, _ = run_study(config(), out=temp)
            ibm_quantum.prepare(output)
            self.assertTrue((output / "ibm" / "circuit.qasm").exists())
            fake_backend = GenericBackendV2(num_qubits=5, seed=42)
            fake_service = SimpleNamespace(backend=lambda _: fake_backend)
            with patch("ibm_quantum.service", return_value=fake_service), patch("qiskit_ibm_runtime.SamplerV2") as sampler:
                sampler.return_value.run.return_value.job_id.return_value = "test-job"
                ibm_quantum.submit(output, "fake_backend", 512, 60)
                with self.assertRaisesRegex(ValueError, "already has a submission attempt"):
                    ibm_quantum.submit(output, "fake_backend", 512, 60)
                self.assertEqual(sampler.return_value.run.call_count, 1)
            fake_job = SimpleNamespace(status=lambda: "DONE", result=lambda: [
                SimpleNamespace(data=SimpleNamespace(meas=SimpleNamespace(get_counts=lambda: {"0011": 20, "0000": 5})))])
            with patch("ibm_quantum.service", return_value=SimpleNamespace(job=lambda _: fake_job)):
                ibm_quantum.collect(output)
            result = json.loads((output / "ibm" / "hardware_results.json").read_text())
            self.assertEqual(result["shots"], 25)
            self.assertAlmostEqual(result["feasible_fraction"], 0.8)
            report = (output / "report.html").read_text()
            self.assertIn('"hardware"', report)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["files_sha256"]["report.html"], hashlib.sha256((output / "report.html").read_bytes()).hexdigest())
            self.assertTrue(manifest["report_updates"])
            (output / "ibm" / "hardware_results.json").write_text(json.dumps({**result, "job_id": "wrong-job"}))
            with self.assertRaisesRegex(ValueError, "does not match"):
                refresh_report(output)

    def test_uncertain_submission_is_not_automatically_retried(self):
        from qiskit.providers.fake_provider import GenericBackendV2
        from types import SimpleNamespace
        import ibm_quantum

        with tempfile.TemporaryDirectory() as temp:
            output, _ = run_study(config(), out=temp)
            fake_service = SimpleNamespace(backend=lambda _: GenericBackendV2(num_qubits=5, seed=42))
            with patch("ibm_quantum.service", return_value=fake_service), patch("qiskit_ibm_runtime.SamplerV2") as sampler:
                sampler.return_value.run.side_effect = TimeoutError("uncertain network response")
                with self.assertRaises(TimeoutError):
                    ibm_quantum.submit(output, "fake_backend", 512, 60)
                with self.assertRaisesRegex(ValueError, "already has a submission attempt"):
                    ibm_quantum.submit(output, "fake_backend", 512, 60)
                self.assertEqual(sampler.return_value.run.call_count, 1)
                self.assertFalse((output / "ibm" / "submission.json").exists())


if __name__ == "__main__":
    unittest.main()
