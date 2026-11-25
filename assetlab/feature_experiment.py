"""Frozen shallow-feature studies and batched IBM feature measurements."""

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import features
from .pipeline import code_provenance, write_json
from .search_experiment import circuit_resources, digest, load_circuit, read_json, stamp, unique_id


def load_run(folder):
    folder = Path(folder).resolve()
    manifest = read_json(folder/'manifest.json')
    for relative, expected in manifest['files_sha256'].items():
        if digest(folder/relative) != expected:
            raise ValueError(f'Frozen feature-study input changed: {relative}. Create a new study.')
    protocol = read_json(folder/'protocol.json')
    if protocol['kind'] != 'shallow_quantum_features' or digest(features.__file__) != protocol['numerical_source_sha256']:
        raise ValueError('The numerical study procedure changed; create a new study before evaluation.')
    return folder, protocol


def inputs(folder):
    prices = pd.read_csv(folder/'prices.csv', index_col=0, parse_dates=True)
    episodes = pd.read_csv(folder/'episodes.csv')
    with np.load(folder/'inputs.npz', allow_pickle=False) as archive:
        z, angles = archive['z'], archive['angles']
    return prices, episodes, z, angles


def prepare(prices, config, *, out='results_asset_lab/features', source=None):
    from qiskit import qpy

    required = ('target', 'market', 'end', 'train', 'validation', 'test', 'horizon', 'shots', 'cost_bps', 'seed')
    if any(key not in config for key in required) or not 0 <= config['cost_bps'] < 10000:
        raise ValueError('Supply the complete study configuration and finite nonnegative costs.')
    if not 1 <= config['shots'] <= 4096 or 2*config['shots']*(config['train']+config['validation']+config['test']) > 262144:
        raise ValueError('Use 1-4096 shots and at most 262,144 total measurement shots per study.')
    if not 0 <= config['seed'] < 2**32:
        raise ValueError('Use a seed from zero to 2^32-1.')
    prices, episodes, z, angles, scaler = features.dataset(prices, **{k: config[k] for k in
        ('target', 'market', 'end', 'train', 'validation', 'test', 'horizon')})
    folder = Path(out).resolve()/unique_id()
    (folder/'circuits').mkdir(parents=True, exist_ok=False)
    prices.to_csv(folder/'prices.csv', index_label='date')
    episodes.to_csv(folder/'episodes.csv', index=False)
    np.savez_compressed(folder/'inputs.npz', z=z, angles=angles)
    cases = []
    for variant in ('product', 'entangled'):
        circuit = features.feature_circuit(entangled=variant == 'entangled')
        circuit.metadata = {'feature_variant': variant}
        relative = f'circuits/{variant}.qpy'
        with (folder/relative).open('wb') as stream:
            qpy.dump(circuit, stream)
        cases.append({'id': variant, 'file': relative, 'resources': circuit_resources(circuit)})
    protocol = {'schema_version': 1, 'kind': 'shallow_quantum_features', 'created_at_utc': stamp(),
        'run_id': folder.name, 'config': config, 'source': source, 'input_scaler': scaler,
        'numerical_source_sha256': digest(features.__file__), 'cases': cases,
        'condition_order': features.FEATURES, 'tuning_grids': features.model_specs(),
        'row_count': len(episodes), 'shots_per_observation': config['shots'],
        'total_measurement_shots': 2*len(episodes)*config['shots'],
        'max_native_two_qubit_gates': 20, 'max_native_depth': 100,
        'selection': 'Choose each hyperparameter and the overall model by validation MSE only; refit on train plus validation before the test block.',
        'trading': 'Single target, long or cash, no leverage. Signal at close t, entry t+1 close, exit ten sessions later by default. Trade iff forecast return net of declared round-trip costs is positive. Cash interest zero.',
        'evidence_status': 'Exploratory chronological holdout on previously inspected historical project data, not pristine unseen data or forward profit.',
        'quantum_claim': 'Six-qubit projected features, no search-speedup guarantee. Small circuits are readily classically simulated. Classical learning and data preparation remain part of every pipeline.'}
    write_json(folder/'protocol.json', protocol)
    write_json(folder/'manifest.json', {'code': code_provenance(), 'files_sha256': {
        p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}})
    ideal = features.representations(z, angles, seed=config['seed'])
    sampled = features.representations(z, angles, seed=config['seed'], shots=config['shots'])
    for name, matrices in (('ideal', ideal), ('shot_simulated', sampled)):
        started = time.perf_counter()
        result = features.evaluate(prices, episodes, matrices, config)
        result['fidelity'] = {variant: {'moment_rmse': float(np.sqrt(np.mean((matrices['quantum_'+variant]-ideal['quantum_'+variant])**2)))}
                              for variant in ('product', 'entangled')}
        result.update(source=name, hardware_executed=False, protocol_sha256=digest(folder/'protocol.json'),
                      evaluated_at_utc=stamp(), elapsed_seconds=time.perf_counter()-started)
        write_json(folder/f'{name}_analysis.json', result)
    render_report(folder)
    return folder


def choose_path(backend, length=6):
    """Bounded beam search using device errors, never market outcomes."""
    target = backend.target
    gate = next((name for name in ('cz', 'ecr', 'cx') if name in target.operation_names), None)
    if gate is None:
        raise ValueError('Backend has no supported two-qubit coupling gate.')
    edges, adjacency = {}, {i: set() for i in range(backend.num_qubits)}
    for pair, properties in target[gate].items():
        if pair is None:
            continue
        a, b = pair
        error = properties.error if properties is not None and properties.error is not None else .01
        edge = tuple(sorted((a, b)))
        edges[edge] = min(edges.get(edge, 1.), float(error))
        adjacency[a].add(b)
        adjacency[b].add(a)
    readout = {}
    for q in adjacency:
        prop = target['measure'].get((q,)) if 'measure' in target.operation_names else None
        readout[q] = float(prop.error) if prop is not None and prop.error is not None else .01
    paths = [(.2*readout[q], (q,)) for q in adjacency]
    for _ in range(length-1):
        candidates = [(score+edges[tuple(sorted((path[-1], q)))]+.2*readout[q], (*path, q))
                      for score, path in paths for q in sorted(adjacency[path[-1]]) if q not in path]
        paths = sorted(candidates)[:256]
        if not paths:
            raise ValueError('Could not find a connected six-qubit path on this backend.')
    score, path = paths[0]
    return list(path), {'native_edge_gate': gate, 'heuristic_score': score,
        'edge_errors': [edges[tuple(sorted(pair))] for pair in zip(path[:-1], path[1:])],
        'readout_errors': [readout[q] for q in path], 'method': 'Deterministic beam width 256; edge error plus 0.2 times readout error. Shared initial layout for both variants.'}


def plan(folder, backend_name):
    from qiskit import qpy
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from ibm_quantum import service

    folder, protocol = load_run(folder)
    backend = service().backend(backend_name)
    physical, selection = choose_path(backend)
    transpiler = generate_preset_pass_manager(backend=backend, initial_layout=physical,
                                             optimization_level=2, seed_transpiler=protocol['config']['seed'])
    output = folder/'ibm/plans'/unique_id()
    output.mkdir(parents=True, exist_ok=False)
    _, _, _, angles = inputs(folder)
    execution_order = np.random.default_rng(protocol['config']['seed']).permutation(len(angles))
    cases, bindings = [], {}
    for case in protocol['cases']:
        native = transpiler.run(load_circuit(folder/case['file']))
        resources = circuit_resources(native)
        if resources['two_qubit_gates'] > protocol['max_native_two_qubit_gates'] or resources['depth'] > protocol['max_native_depth']:
            raise ValueError('Native circuit exceeds the frozen shallow-circuit gate/depth budget; no job submitted.')
        filename = case['id']+'.qpy'
        with (output/filename).open('wb') as stream:
            qpy.dump(native, stream)
        bindings[case['id']] = features.bind_values(native, angles[execution_order])
        cases.append({'id': case['id'], 'file': filename, 'resources': resources,
                      'parameter_order': [str(p) for p in native.parameters],
                      'final_layout': native.layout.final_index_layout() if native.layout else None})
    np.savez_compressed(output/'bindings.npz', **bindings)
    write_json(output/'plan.json', {'created_at_utc': stamp(), 'backend': backend_name,
        'protocol_sha256': digest(folder/'protocol.json'), 'run_id': protocol['run_id'],
        'shared_physical_qubits': physical, 'layout_selection': selection, 'cases': cases,
        'execution_row_order': execution_order.tolist(),
        'row_count': protocol['row_count'], 'shots_per_observation': protocol['shots_per_observation'],
        'total_measurement_shots': protocol['total_measurement_shots'], 'hardware_executed': False})
    write_json(output/'manifest.json', {'files_sha256': {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}})
    verify_plan(folder, output)
    render_report(folder)
    return output


def load_plan(folder, protocol, path):
    path = Path(path).resolve()
    for filename, expected in read_json(path/'manifest.json')['files_sha256'].items():
        if digest(path/filename) != expected:
            raise ValueError('A frozen native feature plan changed; create a new plan.')
    planned = read_json(path/'plan.json')
    if planned['protocol_sha256'] != digest(folder/'protocol.json') or planned['run_id'] != protocol['run_id']:
        raise ValueError('The hardware plan belongs to a different feature study.')
    if [c['id'] for c in planned['cases']] != [c['id'] for c in protocol['cases']]:
        raise ValueError('Hardware variants differ from the frozen study.')
    return path, planned


def verify_plan(folder, path):
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.quantum_info import Statevector

    folder, protocol = load_run(folder)
    path, planned = load_plan(folder, protocol, path)
    _, _, _, angles = inputs(folder)
    order = planned['execution_row_order']
    if sorted(order) != list(range(len(angles))):
        raise ValueError('Hardware row order is not a permutation of the frozen observations.')
    result = {'verified_at_utc': stamp(), 'protocol_sha256': digest(folder/'protocol.json'),
              'plan_sha256': digest(path/'plan.json'), 'hardware_executed': False, 'variants': {}}
    with np.load(path/'bindings.npz', allow_pickle=False) as archive:
        for case in planned['cases']:
            native = load_circuit(path/case['file'])
            active = sorted({native.find_bit(q).index for item in native.data if item.operation.name != 'barrier' for q in item.qubits})
            if len(active) != 6:
                raise ValueError('The shallow plan unexpectedly uses more than six active physical qubits.')
            compact = QuantumCircuit(QuantumRegister(6, 'q'), *native.cregs)
            mapping = {old: new for new, old in enumerate(active)}
            measurement = {}
            for item in native.data:
                if item.operation.name == 'measure':
                    measurement[native.find_bit(item.clbits[0]).index] = mapping[native.find_bit(item.qubits[0]).index]
                elif item.operation.name != 'barrier':
                    compact.append(item.operation, [compact.qubits[mapping[native.find_bit(q).index]] for q in item.qubits], [])
            if sorted(measurement) != list(range(6)):
                raise ValueError('The six feature measurements were not preserved by native compilation.')
            exact = features.probabilities(angles[order], entangled=case['id'] == 'entangled')@features.moment_signs()
            actual = []
            for values in archive[case['id']]:
                state = Statevector.from_instruction(compact.assign_parameters(dict(zip(compact.parameters, values))))
                actual.append(state.probabilities(qargs=[measurement[i] for i in range(6)])@features.moment_signs())
            error = float(np.max(np.abs(np.asarray(actual)-exact)))
            if error > 1e-9:
                raise ValueError('Native feature circuits disagree with the frozen ideal representation.')
            result['variants'][case['id']] = {'rows_verified': len(actual), 'max_moment_error': error}
    write_json(path/'verification.json', result)
    return result


def submit(folder, path, *, max_seconds=60):
    from qiskit_ibm_runtime import SamplerV2
    from ibm_quantum import service

    folder, protocol = load_run(folder)
    path, planned = load_plan(folder, protocol, path)
    if not 1 <= max_seconds <= 600:
        raise ValueError('Use an execution limit of 1-600 seconds.')
    attempt = folder/'ibm/submission_attempt.json'
    if attempt.exists():
        raise ValueError('Submission already attempted; collect or reconcile the IBM dashboard instead of retrying.')
    verification = read_json(path/'verification.json')
    if verification['protocol_sha256'] != digest(folder/'protocol.json') or verification['plan_sha256'] != digest(path/'plan.json'):
        raise ValueError('The native-circuit verification does not match this plan.')
    pubs = []
    with np.load(path/'bindings.npz', allow_pickle=False) as archive:
        for case in planned['cases']:
            native = load_circuit(path/case['file'])
            values = archive[case['id']]
            if values.shape != (protocol['row_count'], 6):
                raise ValueError('Parameter batch dimensions differ from the frozen dataset.')
            pubs.append((native, values))
    backend = service().backend(planned['backend'])
    sampler = SamplerV2(mode=backend, options={'max_execution_time': max_seconds,
        'dynamical_decoupling': {'enable': False}, 'twirling': {'enable_gates': False, 'enable_measure': False}})
    receipt = {'attempted_at_utc': stamp(), 'backend': planned['backend'],
        'protocol_sha256': digest(folder/'protocol.json'), 'plan_sha256': digest(path/'plan.json'),
        'plan_folder': str(path), 'shots_per_observation': protocol['shots_per_observation'],
        'total_measurement_shots': protocol['total_measurement_shots'], 'max_execution_seconds': max_seconds,
        'verification_sha256': digest(path/'verification.json'),
        'variant_order': [c['id'] for c in protocol['cases']], 'dynamical_decoupling': False, 'twirling': False}
    with attempt.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
    job = sampler.run(pubs, shots=protocol['shots_per_observation'])
    receipt.update(job_id=job.job_id(), status='submitted')
    write_json(folder/'ibm/submission.json', receipt)
    render_report(folder)
    return receipt


def collect(folder):
    from ibm_quantum import service

    folder, protocol = load_run(folder)
    receipt_path = folder/'ibm/submission.json'
    if not receipt_path.exists():
        raise ValueError('No confirmed receipt. Reconcile any uncertain submission in the IBM dashboard first.')
    receipt = read_json(receipt_path)
    if receipt['protocol_sha256'] != digest(folder/'protocol.json'):
        raise ValueError('Submitted protocol does not match this study.')
    plan_folder = folder/'ibm/plans'/Path(receipt['plan_folder']).name
    _, planned = load_plan(folder, protocol, plan_folder)
    if digest(plan_folder/'plan.json') != receipt['plan_sha256']:
        raise ValueError('The submitted parameter-batch plan changed.')
    if (folder/'hardware_analysis.json').exists():
        render_report(folder)
        return read_json(folder/'hardware_analysis.json')
    counts_path = folder/'ibm/hardware_counts.json'
    if counts_path.exists():
        measured = read_json(counts_path)
        if measured['job_id'] != receipt['job_id'] or measured['protocol_sha256'] != receipt['protocol_sha256']:
            raise ValueError('Cached feature measurements belong to a different submitted job.')
    else:
        job = service().job(receipt['job_id'])
        status = str(job.status()).upper().split('.')[-1]
        write_json(folder/'ibm/job_status.json', {'checked_at_utc': stamp(), 'job_id': receipt['job_id'], 'status': status})
        print(f"IBM feature job {receipt['job_id']}: {status}", flush=True)
        if status != 'DONE':
            render_report(folder)
            if status in {'ERROR', 'CANCELLED'}:
                raise ValueError(f'IBM job ended with {status}; no automatic resubmission.')
            return None
        result = job.result()
        if len(result) != len(protocol['cases']):
            raise ValueError('Feature PUB result count does not match the frozen variants.')
        measured = {'collected_at_utc': stamp(), 'hardware_executed': True, 'job_id': receipt['job_id'],
                    'backend': receipt['backend'], 'protocol_sha256': receipt['protocol_sha256'], 'variants': {}}
        for case, pub in zip(protocol['cases'], result):
            bits = pub.data.meas
            if tuple(bits.shape) != (protocol['row_count'],) or bits.num_bits != 6:
                raise ValueError('Expected one six-bit count distribution per chronological observation.')
            order = planned['execution_row_order']
            if sorted(order) != list(range(protocol['row_count'])):
                raise ValueError('Invalid frozen observation execution order.')
            rows = [None]*protocol['row_count']
            for location, observation in enumerate(order):
                rows[observation] = bits.get_counts(location)
            for row in rows:
                features.count_features(row, shots=protocol['shots_per_observation'])
            measured['variants'][case['id']] = rows
        write_json(counts_path, measured)  # Preserve paid measurements before model evaluation.
        if callable(getattr(job, 'metrics', None)):
            try:
                write_json(folder/'ibm/job_metrics.json', {'job_id': receipt['job_id'],
                    'metrics': json.loads(json.dumps(job.metrics(), default=str))})
            except Exception:
                pass
    prices, episodes, z, angles = inputs(folder)
    matrices = features.representations(z, angles, seed=protocol['config']['seed'])
    fidelity = {}
    for variant, rows in measured['variants'].items():
        key = 'quantum_'+variant
        actual = np.array([features.count_features(row, shots=protocol['shots_per_observation']) for row in rows])
        ideal = matrices[key]
        fidelity[variant] = {'moment_rmse': float(np.sqrt(np.mean((actual-ideal)**2))),
            'moment_mae': float(np.mean(np.abs(actual-ideal))), 'mean_bias': float(np.mean(actual-ideal))}
        matrices[key] = actual
    evaluated = features.evaluate(prices, episodes, matrices, protocol['config'])
    evaluated.update(source='hardware', hardware_executed=True, job_id=receipt['job_id'],
                     backend=receipt['backend'], protocol_sha256=receipt['protocol_sha256'],
                     evaluated_at_utc=stamp(), fidelity=fidelity, software=code_provenance())
    write_json(folder/'hardware_analysis.json', evaluated)
    render_report(folder)
    return evaluated


def render_report(folder):
    folder, protocol = load_run(folder)
    payload = {'protocol': protocol}
    for source in ('ideal', 'shot_simulated', 'hardware'):
        file = folder/f'{source}_analysis.json'
        if file.exists():
            result = read_json(file)
            if result['protocol_sha256'] != digest(folder/'protocol.json'):
                raise ValueError('Feature results belong to a different frozen study.')
            payload[source] = result
    receipt_path = folder/'ibm/submission.json'
    receipt = read_json(receipt_path) if receipt_path.exists() else None
    if receipt:
        if receipt['protocol_sha256'] != digest(folder/'protocol.json'):
            raise ValueError('Receipt belongs to a different frozen study.')
        payload['submission'] = receipt
    if 'hardware' in payload and (not receipt or payload['hardware']['job_id'] != receipt['job_id'] or not payload['hardware']['hardware_executed']):
        raise ValueError('Hardware feature results lack a matching confirmed submission.')
    plans = sorted((folder/'ibm/plans').glob('*/plan.json'))
    if receipt:
        submitted = folder/'ibm/plans'/Path(receipt['plan_folder']).name/'plan.json'
        if digest(submitted) != receipt['plan_sha256']:
            raise ValueError('Submitted native feature plan changed.')
        plans = [submitted]
    if plans:
        payload['plan'] = read_json(plans[-1])
        verified = plans[-1].parent/'verification.json'
        if verified.exists(): payload['verification'] = read_json(verified)
    for name, filename in (('job_status', 'job_status.json'), ('job_metrics', 'job_metrics.json')):
        path = folder/'ibm'/filename
        if path.exists():
            value = read_json(path)
            if not receipt or value['job_id'] != receipt['job_id']:
                raise ValueError('Job metadata belongs to a different submission.')
            payload[name] = value
    encoded = json.dumps(payload, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = Path(__file__).with_name('feature_report.html')
    (folder/'report.html').write_text(template.read_text().replace('__FEATURE_DATA__', encoded), encoding='utf-8')
    write_json(folder/'report_metadata.json', {'rendered_at_utc': stamp(), 'report_sha256': digest(folder/'report.html'),
        'template_sha256': digest(template), 'protocol_sha256': digest(folder/'protocol.json'), 'code': code_provenance()})
    return folder/'report.html'
