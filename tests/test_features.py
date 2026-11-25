import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from assetlab import features


def prices():
    rng = np.random.default_rng(19)
    common = rng.normal(.0003, .01, 700)
    returns = np.c_[common, .7*common+rng.normal(0, .005, 700)]
    return pd.DataFrame(100*np.exp(np.cumsum(returns, axis=0)), columns=['SPY', 'QQQ'],
                        index=pd.bdate_range('2020-01-01', periods=700))


class FeatureNumericalTests(unittest.TestCase):
    def test_time_boundaries_training_only_scaler_and_future_price_independence(self):
        panel = prices()
        end = panel.index[-31].date().isoformat()
        args = dict(end=end, train=16, validation=8, test=8)
        first = features.dataset(panel, **args)
        changed = panel.copy()
        changed.loc[changed.index > end] *= 100
        second = features.dataset(changed, **args)
        pd.testing.assert_frame_equal(first[1], second[1])
        np.testing.assert_allclose(first[2], second[2])
        episodes = first[1]
        self.assertLessEqual(episodes.iloc[15].exit_index, episodes.iloc[16].signal_index)
        self.assertLessEqual(episodes.iloc[23].exit_index, episodes.iloc[24].signal_index)
        np.testing.assert_allclose(first[2][:16].mean(axis=0), 0., atol=1e-12)
        np.testing.assert_allclose(first[2][:16].std(axis=0), 1., atol=1e-12)
        modified = episodes[features.FEATURES].to_numpy()
        expected = (modified-modified[:16].mean(axis=0))/modified[:16].std(axis=0)
        np.testing.assert_allclose(first[2], expected)

    def test_single_and_joint_measurements_preserve_correlations_and_bit_order(self):
        mixed = features.count_features({'000000': 5, '111111': 5}, shots=10)
        np.testing.assert_array_equal(mixed[:6], 0)
        np.testing.assert_array_equal(mixed[6:], 1)
        one = features.count_features({'000001': 10}, shots=10)
        self.assertEqual(one[0], -1)
        self.assertEqual(one[1], 1)
        self.assertEqual(one[6], -1)
        with self.assertRaises(ValueError): features.count_features({'0': 1}, shots=10)

    def test_delayed_entry_and_both_fees_reconcile_with_daily_drawdown(self):
        panel = pd.DataFrame({'SPY': [100., 80., 90., 100.]}, index=pd.bdate_range('2020-01-01', periods=4))
        episodes = pd.DataFrame([{'signal_index': 0, 'entry_index': 1, 'exit_index': 3}])
        result = features.backtest(panel, episodes, [True], target='SPY', cost_bps=10)
        expected = 100/80*.999/1.001-1
        self.assertAlmostEqual(result['total_return'], expected)
        self.assertLess(result['equity'][1], 1.)
        self.assertLess(result['max_drawdown'], 0.)
        changed = panel.copy()
        changed.iloc[0, 0] = 1
        self.assertEqual(result, features.backtest(changed, episodes, [True], target='SPY', cost_bps=10))
        cash = features.backtest(panel, episodes, [False], target='SPY', cost_bps=10)
        self.assertEqual(cash['total_return'], 0.)
        self.assertEqual(cash['total_fees_as_initial_equity'], 0.)

    @unittest.skipUnless(importlib.util.find_spec('sklearn'), 'optional predictive models')
    def test_test_labels_cannot_choose_hyperparameters_or_change_fitted_predictions(self):
        rng = np.random.default_rng(25)
        x = rng.normal(size=(32, 6))
        y = .01*x[:, 0]+rng.normal(0, .005, 32)
        grid = [{'kind': 'ridge', 'alpha': a} for a in (.01, 1., 100.)]
        first = features.fit_model(x, y, grid, train=16, validation=8, seed=42)
        y[24:] = 1000
        second = features.fit_model(x, y, grid, train=16, validation=8, seed=42)
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    @unittest.skipUnless(importlib.util.find_spec('qiskit'), 'optional quantum circuit dependency')
    def test_fast_probabilities_match_qiskit_for_both_shallow_variants(self):
        from qiskit.quantum_info import Statevector

        angles = np.random.default_rng(12).normal(size=(5, 6))
        for entangled in (False, True):
            circuit = features.feature_circuit(entangled=entangled, measure=False)
            self.assertEqual(circuit.num_qubits, 6)
            self.assertEqual(circuit.count_ops().get('cz', 0), 5 if entangled else 0)
            actual = [Statevector.from_instruction(circuit.assign_parameters(dict(zip(circuit.parameters, row)))).probabilities()
                      for row in features.bind_values(circuit, angles)]
            np.testing.assert_allclose(actual, features.probabilities(angles, entangled=entangled), atol=1e-12)


@unittest.skipUnless(importlib.util.find_spec('qiskit') and importlib.util.find_spec('sklearn'), 'optional study dependencies')
class FeatureWorkflowTests(unittest.TestCase):
    def make_run(self, temp):
        from assetlab.feature_experiment import prepare

        panel = prices()
        config = dict(target='SPY', market='QQQ', end=panel.index[-1].date().isoformat(),
                      train=16, validation=8, test=8, horizon=10, shots=16, cost_bps=10., seed=42)
        return prepare(panel, config, out=temp, source={'kind': 'synthetic_test'})

    def test_input_edits_are_rejected_before_cloud_access(self):
        from assetlab.feature_experiment import load_run

        with tempfile.TemporaryDirectory() as temp:
            folder = self.make_run(temp)
            with (folder/'episodes.csv').open('a') as stream: stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'Frozen feature-study input changed'):
                load_run(folder)

    @unittest.skipUnless(importlib.util.find_spec('qiskit_ibm_runtime'), 'optional IBM dependency')
    def test_batched_parameters_counts_collection_and_duplicate_protection(self):
        from qiskit.primitives.containers import BitArray
        from qiskit.providers.fake_provider import GenericBackendV2
        from assetlab.feature_experiment import collect, inputs, load_run, plan, submit

        with tempfile.TemporaryDirectory() as temp:
            folder = self.make_run(temp)
            backend = GenericBackendV2(num_qubits=8, seed=42)
            with patch('ibm_quantum.service', return_value=SimpleNamespace(backend=lambda _: backend)), \
                 patch('qiskit_ibm_runtime.SamplerV2') as sampler:
                path = plan(folder, 'fake_backend')
                sampler.assert_not_called()
                sampler.return_value.run.return_value.job_id.return_value = 'test-feature-job'
                submit(folder, path)
                args = sampler.return_value.run.call_args
                self.assertEqual(len(args.args[0]), 2)
                self.assertEqual(args.args[0][0][1].shape, (32, 6))
                self.assertEqual(args.kwargs['shots'], 16)
                with self.assertRaisesRegex(ValueError, 'already attempted'):
                    submit(folder, path)
                sampler.return_value.run.assert_called_once()
            _, protocol = load_run(folder)
            _, _, _, angles = inputs(folder)
            order = json.loads((path/'plan.json').read_text())['execution_row_order']
            rng = np.random.default_rng(10)
            pubs = []
            for variant in ('product', 'entangled'):
                probs = features.probabilities(angles[order], entangled=variant == 'entangled')
                samples = np.array([rng.choice(64, 16, p=row) for row in probs], dtype=np.uint8)
                bits = BitArray(samples[:, :, None], num_bits=6)
                pubs.append(SimpleNamespace(data=SimpleNamespace(meas=bits)))
            queued = SimpleNamespace(status=lambda: 'QUEUED', result=lambda: self.fail('No waiting for queued jobs'))
            with patch('ibm_quantum.service', return_value=SimpleNamespace(job=lambda _: queued)):
                self.assertIsNone(collect(folder))
            done = SimpleNamespace(status=lambda: 'DONE', result=lambda: pubs)
            with patch('ibm_quantum.service', return_value=SimpleNamespace(job=lambda _: done)):
                result = collect(folder)
            self.assertTrue(result['hardware_executed'])
            self.assertEqual(result['job_id'], 'test-feature-job')
            self.assertEqual(len(result['models']), 8)
            self.assertIn('moment_rmse', result['fidelity']['entangled'])
            with patch('ibm_quantum.service', side_effect=AssertionError('Already collected')):
                self.assertEqual(result, collect(folder))
            self.assertTrue((folder/'ibm/hardware_counts.json').is_file())
            measured = json.loads((folder/'ibm/hardware_counts.json').read_text())
            self.assertEqual(measured['variants']['product'][0], pubs[0].data.meas.get_counts(order.index(0)))
            self.assertEqual(protocol['row_count'], 32)


if __name__ == '__main__': unittest.main()
