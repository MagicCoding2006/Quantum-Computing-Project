"""Shallow six-qubit market features and a fixed chronological prediction study."""

import numpy as np
import pandas as pd


FEATURES = ['return_1', 'return_5', 'return_20', 'volatility_20', 'market_return_5', 'correlation_60']
ALPHAS = np.logspace(-5, 4, 12).tolist()


def dataset(prices, *, target='SPY', market='QQQ', end, train=128, validation=64, test=64, horizon=10):
    if min(train, validation, test) < 8 or max(train, validation, test) > 512 or not 1 <= horizon <= 63:
        raise ValueError('Use 8-512 episodes per split and a 1-63 session holding period.')
    if target == market:
        raise ValueError('Choose distinct target and market assets for these six features.')
    cutoff = pd.Timestamp(end)
    if pd.isna(cutoff) or prices.index.has_duplicates or not prices.index.is_monotonic_increasing or prices.index.hasnans:
        raise ValueError('Use a valid cutoff and valid, unique, increasing price dates.')
    prices = prices.loc[:cutoff, [target, market]].copy()
    if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any().any():
        raise ValueError('Prices must be complete, finite, and positive; missing closes are not filled.')
    returns = prices.pct_change(fill_method=None)
    observed = pd.DataFrame({
        'return_1': returns[target], 'return_5': prices[target].pct_change(5, fill_method=None),
        'return_20': prices[target].pct_change(20, fill_method=None),
        'volatility_20': returns[target].rolling(20).std(),
        'market_return_5': prices[market].pct_change(5, fill_method=None),
        'correlation_60': returns[target].rolling(60).corr(returns[market]),
    })
    rows = []
    for signal in range(60, len(prices)-horizon-1, horizon+1):
        values = observed.iloc[signal].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError('Undefined trailing features; use a source with nonconstant price variation.')
        entry, exit_ = signal+1, signal+1+horizon
        rows.append({'signal_date': prices.index[signal].date().isoformat(),
            'entry_date': prices.index[entry].date().isoformat(), 'exit_date': prices.index[exit_].date().isoformat(),
            'signal_index': signal, 'entry_index': entry, 'exit_index': exit_,
            **dict(zip(FEATURES, values)), 'gross_return': float(prices[target].iloc[exit_]/prices[target].iloc[entry]-1)})
    total = train+validation+test
    if len(rows) < total:
        raise ValueError(f'Need {total} completed, nonoverlapping episodes; this source provides {len(rows)}.')
    episodes = pd.DataFrame(rows[-total:]).reset_index(drop=True)
    episodes['split'] = ['train']*train + ['validation']*validation + ['test']*test
    x = episodes[FEATURES].to_numpy(dtype=float)
    center = x[:train].mean(axis=0)
    scale = x[:train].std(axis=0)
    scale[scale < 1e-12] = 1.
    standardized = (x-center)/scale
    return prices, episodes, standardized, np.arctan(standardized), {'center': center.tolist(), 'scale': scale.tolist(), 'fit_rows': train}


def feature_circuit(*, entangled=True, measure=True):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    x = ParameterVector('x', 6)
    circuit = QuantumCircuit(6, name='entangled_features' if entangled else 'product_features')
    for i in range(6):
        circuit.ry(x[i], i)
        circuit.rz(x[i], i)
    if entangled:
        for start in (0, 1):
            for i in range(start, 5, 2):
                circuit.cz(i, i+1)
    for i in range(6):
        circuit.ry(np.pi/4+x[i]/2, i)
    if measure:
        circuit.measure_all()
    return circuit


def bind_values(circuit, angles):
    names = [str(p) for p in circuit.parameters]
    if set(names) != {f'x[{i}]' for i in range(6)}:
        raise ValueError('Unexpected feature-circuit parameter order.')
    return np.asarray(angles)[:, [int(name[2:-1]) for name in names]]


def probabilities(angles, *, entangled=True):
    angles = np.asarray(angles, dtype=float)
    if angles.ndim != 2 or angles.shape[1] != 6 or not np.isfinite(angles).all():
        raise ValueError('Expected finite observation rows with six angles.')
    states = np.zeros((len(angles), 64), dtype=complex)
    states[:, 0] = 1
    bits = (np.arange(64)[:, None] >> np.arange(6)) & 1

    def rotate(bit, theta):
        blocks = states.reshape(len(states), -1, 2, 1 << bit)
        zero, one = blocks[:, :, 0, :].copy(), blocks[:, :, 1, :].copy()
        c, s = np.cos(theta/2)[:, None, None], np.sin(theta/2)[:, None, None]
        blocks[:, :, 0, :], blocks[:, :, 1, :] = c*zero-s*one, s*zero+c*one

    for i in range(6):
        rotate(i, angles[:, i])
        states *= np.exp(-.5j*angles[:, i, None]*(1-2*bits[:, i]))
    if entangled:
        states *= np.prod(1-2*bits[:, :-1]*bits[:, 1:], axis=1)
    for i in range(6):
        rotate(i, np.pi/4+angles[:, i]/2)
    result = np.abs(states)**2
    return result/result.sum(axis=1, keepdims=True)


def moment_signs():
    z = 1-2*((np.arange(64)[:, None] >> np.arange(6)) & 1)
    return np.c_[z, z[:, :-1]*z[:, 1:]]


def count_features(raw_counts, *, shots):
    counts = np.zeros(64)
    for raw, value in raw_counts.items():
        bits = str(raw).replace(' ', '')
        if not bits or len(bits) > 6 or set(bits)-{'0', '1'} or not isinstance(value, int) or value < 0:
            raise ValueError('Invalid six-qubit feature counts.')
        counts[int(bits, 2)] += value
    if counts.sum() != shots:
        raise ValueError('Feature shot count differs from the frozen protocol.')
    return counts@moment_signs()/shots


def representations(z, angles, *, seed=42, shots=None):
    rng = np.random.default_rng(seed)
    matrices = {'raw': np.asarray(z), 'trigonometric': np.c_[np.sin(angles), np.cos(angles),
        np.sin(angles[:, :-1])*np.sin(angles[:, 1:]), np.cos(angles[:, :-1])*np.cos(angles[:, 1:])]}
    weights, phases = rng.normal(size=(6, 32)), rng.uniform(0, 2*np.pi, size=32)
    matrices['random_features'] = np.sqrt(2/32)*np.cos(z@weights+phases)
    for entangled in (False, True):
        probs = probabilities(angles, entangled=entangled)
        if shots is not None:
            probs = np.array([rng.multinomial(shots, row)/shots for row in probs])
        matrices['quantum_entangled' if entangled else 'quantum_product'] = probs@moment_signs()
    return matrices


def backtest(prices, episodes, positions, *, target, cost_bps):
    if not 0 <= cost_bps < 10000 or len(positions) != len(episodes):
        raise ValueError('Invalid trading costs or episode position count.')
    begin, end = int(episodes.iloc[0].signal_index), int(episodes.iloc[-1].exit_index)
    curve, exposure = np.full(end-begin+1, np.nan), np.zeros(end-begin+1)
    closes, cash, previous = prices[target].to_numpy(dtype=float), 1., 0
    fee, total_fees, net = cost_bps/10000, 0., []
    for (_, row), invest in zip(episodes.iterrows(), positions):
        lo, hi = int(row.entry_index)-begin, int(row.exit_index)-begin
        if lo < previous or hi <= lo:
            raise ValueError('Evaluation trades must have nonoverlapping, delayed holding intervals.')
        curve[previous:lo] = cash
        before = cash
        if invest:
            units = cash/(closes[lo+begin]*(1+fee))
            total_fees += units*(closes[lo+begin]+closes[hi+begin])*fee
            curve[lo:hi+1] = units*closes[lo+begin:hi+begin+1]
            exposure[lo:hi] = 1
            cash = units*closes[hi+begin]*(1-fee)
            curve[hi] = cash
        else:
            curve[lo:hi+1] = cash
        net.append(cash/before-1)
        previous = hi+1
    if not np.isfinite(curve).all():
        raise ValueError('Incomplete daily equity ledger.')
    daily = curve[1:]/curve[:-1]-1
    vol = float(np.std(daily, ddof=1)*np.sqrt(252)) if len(daily)>1 else 0.
    return {'total_return': float(cash-1), 'cagr': float(cash**(252/max(1, len(daily)))-1),
        'max_drawdown': float(np.min(curve/np.maximum.accumulate(curve)-1)), 'annual_volatility': vol,
        'sharpe_zero_rf': float(np.mean(daily)*252/vol) if vol>1e-12 else None,
        'mean_exposure': float(exposure.mean()), 'trades': int(np.sum(positions)),
        'total_fees_as_initial_equity': float(total_fees), 'episode_returns': net,
        'equity': curve.tolist(), 'dates': [date.date().isoformat() for date in prices.index[begin:end+1]]}


def buy_hold(prices, episodes, *, target, cost_bps):
    first, last = episodes.iloc[0].copy(), episodes.iloc[-1]
    first.exit_index, first.exit_date = last.exit_index, last.exit_date
    return backtest(prices, pd.DataFrame([first]), [True], target=target, cost_bps=cost_bps)


def model_specs():
    specs = {
        'historical_mean': ('raw', [{'kind': 'mean'}]),
        'linear_ridge': ('raw', [{'kind': 'ridge', 'alpha': a} for a in ALPHAS]),
        'rbf_kernel': ('raw', [{'kind': 'rbf', 'gamma': g, 'alpha': a} for g in (.1, 1., 10.) for a in (.001, .01, .1, 1.)]),
        'boosted_trees': ('raw', [{'kind': 'boosted', 'max_depth': d, 'l2_regularization': a} for d in (1, 2, 3) for a in (.01, .1, 1., 10.)]),
    }
    for representation in ('trigonometric', 'random_features', 'quantum_product', 'quantum_entangled'):
        specs[representation] = (representation, [{'kind': 'ridge', 'alpha': a} for a in ALPHAS])
    return specs


def estimator(parameters, seed):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    params = dict(parameters)
    kind = params.pop('kind')
    if kind == 'mean': model = DummyRegressor(strategy='mean')
    elif kind == 'ridge': model = Ridge(**params)
    elif kind == 'rbf': model = KernelRidge(kernel='rbf', **params)
    else: model = HistGradientBoostingRegressor(**params, max_iter=100, learning_rate=.05,
                min_samples_leaf=8, early_stopping=False, random_state=seed)
    return TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), model), transformer=StandardScaler())


def fit_model(x, y, parameters, *, train, validation, seed):
    losses = []
    for candidate in parameters:
        model = estimator(candidate, seed).fit(x[:train], y[:train])
        losses.append(float(np.mean((model.predict(x[train:train+validation])-y[train:train+validation])**2)))
    chosen = int(np.argmin(losses))
    model = estimator(parameters[chosen], seed).fit(x[:train+validation], y[:train+validation])
    return {'parameters': parameters[chosen], 'validation_mse': losses[chosen],
            'tuning_trials': len(parameters), 'all_validation_losses': losses}, model.predict(x[train+validation:])


def evaluate(prices, episodes, matrices, config):
    from threadpoolctl import threadpool_limits

    train, validation = config['train'], config['validation']
    test_rows = episodes.iloc[train+validation:]
    y = episodes.gross_return.to_numpy(dtype=float)
    target, cost = config['target'], config['cost_bps']
    models = {}
    with threadpool_limits(limits=2):
        for name, (representation, grid) in model_specs().items():
            selection, prediction = fit_model(matrices[representation], y, grid, train=train,
                                              validation=validation, seed=config['seed'])
            net_prediction = (1+prediction)*(1-cost/10000)/(1+cost/10000)-1
            positions = net_prediction>0
            models[name] = {**selection, 'test_predictions': prediction.tolist(),
                'test_mse': float(np.mean((prediction-y[train+validation:])**2)),
                'test_direction_accuracy': float(np.mean((prediction>0)==(y[train+validation:]>0))),
                'trading': backtest(prices, test_rows, positions, target=target, cost_bps=cost),
                'cost_sensitivity_total_return': {str(fee): backtest(prices, test_rows, positions,
                    target=target, cost_bps=fee)['total_return'] for fee in (0, 5, 10, 25)}}
    winner = min(models, key=lambda name: models[name]['validation_mse'])
    classical = min((name for name in models if not name.startswith('quantum_')), key=lambda name: models[name]['validation_mse'])
    reference = np.array(models[classical]['trading']['episode_returns'])
    rng = np.random.default_rng(config['seed']+1)
    length, block = len(reference), min(4, len(reference))
    starts = rng.integers(length, size=(1000, int(np.ceil(length/block))))
    indices = ((starts[:, :, None]+np.arange(block))%length).reshape(1000, -1)[:, :length]
    for name in ('quantum_product', 'quantum_entangled'):
        delta = np.array(models[name]['trading']['episode_returns'])-reference
        models[name]['mean_episode_return_difference_vs_classical'] = float(delta.mean())
        models[name]['paired_block_bootstrap_95'] = np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist()
    return {'models': models, 'validation_selected_model': winner, 'validation_selected_classical': classical,
        'test_dates': {'first_signal': test_rows.iloc[0].signal_date, 'first_entry': test_rows.iloc[0].entry_date,
                       'last_exit': test_rows.iloc[-1].exit_date},
        'benchmarks': {'cash': backtest(prices, test_rows, np.zeros(len(test_rows), dtype=bool), target=target, cost_bps=cost),
            'always_long_same_episodes': backtest(prices, test_rows, np.ones(len(test_rows), dtype=bool), target=target, cost_bps=cost),
            'buy_and_hold': buy_hold(prices, test_rows, target=target, cost_bps=cost)},
        'limitations': 'Exploratory chronological holdout, previously inspected project data. One fixed test block, not forward returns. Return comparisons are not risk/exposure-adjusted alpha. Bootstrap intervals use four-episode circular blocks and are not corrected for research selection or multiple comparisons.'}
