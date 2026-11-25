import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from assetlab.pipeline import write_json
from assetlab.search import search_problem
from assetlab.search_experiment import (collect_hardware, load_protocol, plan_hardware, prepare_experiment,
    score_counts, simulate_experiment, submit_hardware, verify_native)


@unittest.skipUnless(importlib.util.find_spec('qiskit'), 'optional Qiskit dependency')
class SearchExperimentTests(unittest.TestCase):
    def make_experiment(self, temp):
        root = Path(temp)/'source'
        root.mkdir()
        problem = search_problem([[1, 0], [0, 1]], [-1, 2], min_support=1, min_score=2)
        write_json(root/'problem.json', problem)
        write_json(root/'audit.json', {'problem': problem, 'source': {'mode': 'synthetic'}})
        return prepare_experiment(root, shots=32, iterations=(0, 1), cx_error=0,
                                  one_qubit_error=0, readout_error=0)

    def test_primary_success_includes_workspace_errors_instead_of_postselecting(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = self.make_experiment(temp)
            _, p = load_protocol(folder)
            p['shots_per_circuit'] = 4
            case = next(c for c in p['cases'] if c['id'] == 'optimized_k1')
            result = score_counts(p, case, {'0000000': 1, '0000010': 1, '1000010': 2})
            self.assertEqual(result['passing_fraction'], .75)
            self.assertEqual(result['workspace_zero_fraction'], .5)
            self.assertEqual(result['passing_and_workspace_zero_fraction'], .25)
            self.assertAlmostEqual(result['search_distribution_tv_from_ideal'], .25)
            self.assertAlmostEqual(result['joint_distribution_tv_from_ideal'], .75)
            self.assertLess(result['passing_wilson_95'][0], .75)
            self.assertGreater(result['passing_wilson_95'][1], .75)
            with self.assertRaisesRegex(ValueError, 'shot count'):
                score_counts(p, case, {'0': 1})

    def test_frozen_protocol_and_circuit_edits_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = self.make_experiment(temp)
            protocol = json.loads((folder/'protocol.json').read_text())
            protocol['shots_per_circuit'] *= 2
            write_json(folder/'protocol.json', protocol)
            with self.assertRaisesRegex(ValueError, 'Frozen experiment file changed'):
                load_protocol(folder)

    @unittest.skipUnless(importlib.util.find_spec('qiskit_aer'), 'optional local noise simulator')
    def test_large_hardware_protocol_does_not_allocate_local_statevector(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = self.make_experiment(temp)
            _, protocol = load_protocol(folder)
            protocol['encodings']['baseline']['total_qubits'] = 19
            with patch('assetlab.search_experiment.load_protocol', return_value=(folder, protocol)), \
                 patch('qiskit_aer.AerSimulator') as simulator:
                with self.assertRaisesRegex(ValueError, 'limited to 18'):
                    simulate_experiment(folder)
                simulator.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec('qiskit_aer'), 'optional local noise simulator')
    def test_zero_noise_simulation_reproduces_exact_amplification_and_is_cached(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = self.make_experiment(temp)
            with patch('ibm_quantum.service', side_effect=AssertionError('No cloud access')):
                result = simulate_experiment(folder)
            self.assertFalse(result['hardware_executed'])
            for mode in ('ideal_sampled', 'synthetic_noise'):
                for case in result[mode]:
                    self.assertEqual(case['workspace_zero_fraction'], 1.)
                    if case['case_id'].endswith('k1'):
                        self.assertEqual(case['passing_fraction'], 1.)
            self.assertEqual(result, simulate_experiment(folder))
            self.assertIn('Grover', (folder/'report.html').read_text())

    @unittest.skipUnless(importlib.util.find_spec('qiskit_ibm_runtime'), 'optional IBM dependency')
    def test_planned_batch_submission_collection_and_duplicate_protection(self):
        from qiskit.providers.fake_provider import GenericBackendV2

        with tempfile.TemporaryDirectory() as temp:
            folder = self.make_experiment(temp)
            backend = GenericBackendV2(num_qubits=10, seed=42)
            runtime = SimpleNamespace(backend=lambda _: backend)
            with patch('ibm_quantum.service', return_value=runtime), patch('qiskit_ibm_runtime.SamplerV2') as sampler:
                plan = plan_hardware(folder, 'fake_backend')
                sampler.assert_not_called()
                if importlib.util.find_spec('qiskit_aer'):
                    verified = verify_native(folder, plan)
                    for case in verified['results']:
                        self.assertEqual(case['passing_fraction'], 1.)
                        self.assertEqual(case['workspace_zero_fraction'], 1.)
                sampler.return_value.run.return_value.job_id.return_value = 'test-grover-job'
                receipt = submit_hardware(folder, plan)
                self.assertEqual(receipt['job_id'], 'test-grover-job')
                self.assertEqual(sampler.return_value.run.call_args.kwargs['shots'], 32)
                self.assertEqual(len(sampler.return_value.run.call_args.args[0]), 4)
                with self.assertRaisesRegex(ValueError, 'already attempted'):
                    submit_hardware(folder, plan)
                sampler.return_value.run.assert_called_once()
            _, protocol = load_protocol(folder)
            queued = SimpleNamespace(status=lambda: 'QUEUED', result=lambda: self.fail('Do not wait for queued results'))
            with patch('ibm_quantum.service', return_value=SimpleNamespace(job=lambda _: queued)):
                self.assertIsNone(collect_hardware(folder))
            self.assertEqual(json.loads((folder/'ibm/job_status.json').read_text())['status'], 'QUEUED')
            self.assertIn('QUEUED', (folder/'report.html').read_text())
            counts = [SimpleNamespace(data=SimpleNamespace(meas=SimpleNamespace(
                get_counts=lambda: {'10': 32}))) for _ in protocol['cases']]
            job = SimpleNamespace(status=lambda: 'DONE', result=lambda: counts)
            with patch('ibm_quantum.service', return_value=SimpleNamespace(job=lambda _: job)):
                output = collect_hardware(folder)
            self.assertTrue(output['hardware_executed'])
            self.assertEqual(len(output['results']), 4)
            with patch('ibm_quantum.service', side_effect=AssertionError('Already collected')):
                self.assertEqual(output, collect_hardware(folder))

    @unittest.skipUnless(importlib.util.find_spec('qiskit_ibm_runtime'), 'optional IBM dependency')
    def test_uncertain_submission_is_not_retried_and_has_no_confirmed_receipt(self):
        from qiskit.providers.fake_provider import GenericBackendV2

        with tempfile.TemporaryDirectory() as temp:
            folder = self.make_experiment(temp)
            backend = GenericBackendV2(num_qubits=10, seed=42)
            with patch('ibm_quantum.service', return_value=SimpleNamespace(backend=lambda _: backend)), \
                 patch('qiskit_ibm_runtime.SamplerV2') as sampler:
                plan = plan_hardware(folder, 'fake_backend')
                sampler.return_value.run.side_effect = ConnectionError('Uncertain response')
                with self.assertRaises(ConnectionError):
                    submit_hardware(folder, plan)
                with self.assertRaisesRegex(ValueError, 'already attempted'):
                    submit_hardware(folder, plan)
                sampler.return_value.run.assert_called_once()
            with self.assertRaisesRegex(ValueError, 'No confirmed IBM receipt'):
                collect_hardware(folder)


if __name__ == '__main__':
    unittest.main()
