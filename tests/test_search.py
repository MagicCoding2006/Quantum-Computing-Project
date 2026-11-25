import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from assetlab.search import (classical_reference, coverage_reference, evaluate_rule, grover_circuit,
                            ideal_search, phase_oracle, scaling_table, search_problem,
                            compiled_resources, with_encoding)
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
            selected, selected_names = price_episodes(source, training_end=end, episodes=4,
                feature_assets=['QQQ', 'UNUSED_MISSING_ASSET'], conditions=8)
            self.assertEqual(len(selected_names), 8)
            pd.testing.assert_frame_equal(first, selected[first.columns])
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
    def test_both_encodings_match_the_predicate_on_superpositions_with_clean_workspace(self):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        fixtures = [
            ([[1, 0], [1, 0], [0, 1], [0, 1]], [2, -2, 4, 2], 3, 3),
            ([[1, 1], [1, 0], [1, 0], [0, 1]], [5, -2, -2, 2], 2, 2),
            ([[1, 1], [1, 1]], [-2, 3], 1, 0),
            ([[1, 0], [1, 0], [0, 1], [0, 1]], [-2, -2, -4, -2], 1, -5),
        ]
        rng = np.random.default_rng(121)
        for _ in range(4):
            fixtures.append((rng.integers(0, 2, (4, 3)), rng.integers(-3, 4, 4), 2, 1))
        for features, scores, support, threshold in fixtures:
            p = search_problem(features, scores, min_support=support, min_score=threshold)
            n = p['search_qubits']
            for implementation in ('baseline', 'optimized'):
                with self.subTest(features=features, implementation=implementation):
                    encoded = with_encoding(p, implementation)
                    self.assertEqual(encoded['predicate_sha256'], p['predicate_sha256'])
                    circuit = QuantumCircuit(encoded['total_qubits'])
                    circuit.h(range(n))
                    circuit.compose(phase_oracle(encoded), inplace=True)
                    actual = Statevector.from_instruction(circuit).data
                    expected = np.zeros_like(actual)
                    expected[:1 << n] = [(-1 if evaluate_rule(p, mask)[2] else 1) / np.sqrt(1 << n)
                                               for mask in range(1 << n)]
                    np.testing.assert_allclose(actual, expected, atol=1e-10)

    def test_user_demo_reduces_workspace_and_compiled_two_qubit_gates(self):
        rng = np.random.default_rng(42)
        p = search_problem(rng.integers(0, 2, (8, 4)), [2, -1, 1, -2, 3, -1, 2, -1])
        old = compiled_resources(phase_oracle(with_encoding(p, 'baseline')))
        new = compiled_resources(phase_oracle(p))
        self.assertLess(new['qubits'], old['qubits'])
        self.assertLess(new['cx_gates'], old['cx_gates'])

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

    def test_market_scaling_benchmark_uses_fixed_cases_and_matching_predicates(self):
        dates = pd.bdate_range('2020-01-01', periods=100)
        rng = np.random.default_rng(53)
        prices = pd.DataFrame(100*np.exp(np.cumsum(rng.normal(0, .01, (100, 3)), axis=0)),
                              index=dates, columns=['SPY', 'QQQ', 'IWM'])
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'prices.csv'
            prices.to_csv(source)
            status = main(['--benchmark', '--prices', str(source), '--training-end', '2020-05-15',
                '--feature-assets', 'QQQ', 'IWM', '--condition-sizes', '6', '8',
                '--episode-sizes', '2', '4', '--min-support', '1', '--out', str(Path(temp)/'output')])
            self.assertEqual(status, 0)
            folder = next((Path(temp)/'output/benchmarks').iterdir())
            summary = json.loads((folder/'summary.json').read_text())
            self.assertEqual(len(summary['cases']), 4)
            self.assertFalse(summary['hardware_executed'])
            for case in summary['cases']:
                audit = json.loads((folder/case['path']/'audit.json').read_text())
                implementations = audit['implementations']
                self.assertEqual(implementations['baseline']['predicate_sha256'],
                                 implementations['optimized']['predicate_sha256'])
                frame = pd.read_csv(folder/case['path']/'episodes.csv')
                self.assertEqual(len(frame), case['episodes'])
                self.assertTrue((frame['exit_date'] <= summary['training_end']).all())
            # More conditions use the same episode decisions and outcomes.
            small = pd.read_csv(folder/'conditions_6_episodes_4/episodes.csv')
            large = pd.read_csv(folder/'conditions_8_episodes_4/episodes.csv')
            pd.testing.assert_frame_equal(small, large[small.columns])


if __name__ == '__main__':
    unittest.main()
