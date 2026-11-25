import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from assetlab.search import (classical_reference, coverage_reference, evaluate_rule, grover_circuit,
                            ideal_search, phase_oracle, scaling_table, search_problem)
from quantum_search import main, price_episodes


class SearchReferenceTests(unittest.TestCase):
    def test_structured_scoring_matches_episode_by_episode_for_every_rule(self):
        rng = np.random.default_rng(71)
        p = search_problem(rng.integers(0, 2, (15, 7)), rng.integers(-9, 10, 15), min_support=3, min_score=-2)
        reference = classical_reference(p)
        for mask in range(128):
            count, score, passes = evaluate_rule(p, mask)
            self.assertEqual((reference['support'][mask], reference['score'][mask], reference['marked'][mask]),
                             (count, score, passes))

    def test_invalid_and_empty_data_are_rejected(self):
        for features, scores in [([], []), ([[0, 2]], [1]), ([[0, 1]], [.5]), ([[1]], [np.nan])]:
            with self.subTest(features=features), self.assertRaises(ValueError):
                search_problem(features, scores, min_support=1)

    def test_many_conditions_can_collapse_to_few_episode_patterns(self):
        features = np.tile([[1, 0], [0, 1]], (1, 32))
        p = search_problem(features, [-1, 2], min_support=1, min_score=2)
        result = coverage_reference(p)
        self.assertTrue(result['complete'])
        self.assertEqual(result['distinct_firing_sets'], 4)
        self.assertTrue(evaluate_rule(p, result['witness']['rule_mask'])[2])
        capped = coverage_reference(p, max_states=2)
        self.assertFalse(capped['complete'])
        self.assertLessEqual(capped['distinct_firing_sets'], 2)

    def test_zero_and_all_solutions_have_correct_ideal_probabilities(self):
        for k in range(4):
            self.assertEqual(ideal_search(4, 0, k)['ideal_grover_success'], 0)
            self.assertAlmostEqual(ideal_search(4, 16, k)['ideal_grover_success'], 1)
        self.assertAlmostEqual(ideal_search(4, 1, 0)['random_success_same_query_count'], 1/16)
        self.assertGreater(ideal_search(10, 1, 25)['ideal_grover_success'], .99)
        self.assertEqual(scaling_table([256])[0]['candidate_count'], str(1 << 256))

    def test_price_episodes_do_not_use_future_prices_and_have_delayed_nonoverlapping_fills(self):
        rng = np.random.default_rng(17)
        dates = pd.bdate_range('2020-01-01', periods=160)
        prices = pd.DataFrame(100*np.exp(np.cumsum(rng.normal(0, .01, (160, 2)), axis=0)),
                              index=dates, columns=['SPY', 'QQQ'])
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'prices.csv'
            prices.to_csv(source)
            end = dates[100].date().isoformat()
            first, names = price_episodes(source, training_end=end, episodes=4)
            prices.loc[prices.index > dates[100]] *= 1000
            prices.to_csv(source)
            second, _ = price_episodes(source, training_end=end, episodes=4)
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(len(names), 6)
            for i, row in first.iterrows():
                self.assertLess(row['signal_date'], row['entry_date'])
                self.assertLess(row['entry_date'], row['exit_date'])
                self.assertLessEqual(row['exit_date'], end)
                if i:
                    self.assertLess(first.iloc[i-1]['exit_date'], row['entry_date'])


@unittest.skipUnless(importlib.util.find_spec('qiskit'), 'optional Qiskit dependency')
class ReversibleOracleTests(unittest.TestCase):
    def test_oracle_marks_correct_phase_and_cleans_workspace_for_every_basis_rule(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        for scores, threshold, min_support in [([-1, 2], 1, 1), ([-2, -1], -2, 1),
                                                ([0, 0], 0, 2), ([1, 2], 99, 1)]:
            p = search_problem([[1, 0], [0, 1]], scores, min_score=threshold, min_support=min_support)
            with patch('assetlab.search.classical_reference', side_effect=AssertionError('No precomputed winners')):
                oracle = phase_oracle(p)
            for mask in range(4):
                with self.subTest(scores=scores, threshold=threshold, mask=mask):
                    circuit = QuantumCircuit(p['total_qubits'])
                    for bit in range(2):
                        if (mask >> bit) & 1:
                            circuit.x(bit)
                    circuit.compose(oracle, inplace=True)
                    actual = Statevector.from_instruction(circuit).data
                    expected = np.zeros_like(actual)
                    expected[mask] = -1 if evaluate_rule(p, mask)[2] else 1
                    np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_grover_circuit_matches_ideal_amplification_and_uncomputes_workspace(self):
        from qiskit.quantum_info import Statevector

        p = search_problem([[1, 0, 0], [0, 1, 0]], [-1, 2], min_support=1, min_score=2)
        marked = classical_reference(p)['marked']
        for k in [0, 1, 2]:
            state = Statevector.from_instruction(grover_circuit(p, k, measure=False)).probabilities()
            self.assertAlmostEqual(float(state[8:].sum()), 0., places=10)
            self.assertAlmostEqual(float(state[:8][marked].sum()), ideal_search(3, int(marked.sum()), k)['ideal_grover_success'], places=10)

    def test_offline_cli_exports_circuit_and_hashes_without_runtime_access(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'episodes.csv'
            pd.DataFrame({'A': [1, 0], 'B': [0, 1], 'score_units': [-1, 2]}).to_csv(source, index=False)
            with patch('ibm_quantum.service', side_effect=AssertionError('No cloud access')):
                status = main(['--episodes-csv', str(source), '--min-support', '1', '--min-score', '2',
                               '--out', str(Path(temp)/'outputs')])
            self.assertEqual(status, 0)
            folder = next((Path(temp)/'outputs').iterdir())
            report = json.loads((folder/'audit.json').read_text())
            self.assertFalse(report['hardware_executed'])
            self.assertEqual(report['classical_reference']['marked_count'], 1)
            self.assertGreater(report['oracle_resources']['cx_gates'], 0)
            self.assertTrue((folder/'grover.qpy').is_file())
            manifest = json.loads((folder/'manifest.json').read_text())
            self.assertIn('grover.qpy', manifest['files_sha256'])
            self.assertIn('quantum_search.py', manifest['code']['source_sha256'])


if __name__ == '__main__':
    unittest.main()
