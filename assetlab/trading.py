"""Long-only relative-value research with explicit signal and fill timestamps.

This models adjusted-close fills, not brokerage orders. Formation uses only
the trailing training window. Parameters are fixed across walk-forward folds.
"""

from dataclasses import asdict, dataclass
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from .data import validate_prices


@dataclass(frozen=True)
class TradingConfig:
    train_sessions: int = 504
    test_sessions: int = 63
    peers: int = 3
    min_correlation: float = 0.60
    coint_pvalue: float = 0.05
    min_half_life: float = 2.0
    max_half_life: float = 60.0
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    max_hold: int = 10
    stop_loss: float = 0.05
    max_positions: int = 5
    initial_capital: float = 100000.0
    cost_bps: float = 10.0
    market: str = "SPY"
    trend_filter: bool = True

    def validate(self):
        if self.train_sessions < 126 or self.test_sessions < 2:
            raise ValueError("Use at least 126 formation sessions and two sessions per test fold.")
        if self.peers < 1 or not 0 < self.min_correlation < 1 or not 0 < self.coint_pvalue < 1:
            raise ValueError("Peer count must be positive; correlation and p-value cutoffs must be between 0 and 1.")
        if not 0 <= self.exit_z < self.entry_z < self.stop_z:
            raise ValueError("Require 0 <= exit-z < entry-z < stop-z.")
        if not 0 < self.min_half_life < self.max_half_life or self.max_hold < 1 or self.max_positions < 1:
            raise ValueError("Use positive holding/position limits and ordered positive half-life bounds.")
        if not 0 < self.stop_loss < 1 or self.initial_capital <= 0 or not 0 <= self.cost_bps <= 1000:
            raise ValueError("Use a positive capital amount, a stop loss between 0 and 1, and costs of 0-1000 bps.")
        if not all(np.isfinite(v) for v in asdict(self).values() if isinstance(v, (float, int))):
            raise ValueError("Trading configuration values must be finite.")


def check_panel(prices, universe, config):
    config.validate()
    required = list(dict.fromkeys([*universe, config.market]))
    prices = validate_prices(prices, required)
    if prices.isna().any().any():
        raise ValueError("Trading requires complete common-session prices. Missing quotes must be resolved "
                         "explicitly; this backtester never fills prices or bridges missing rows.")
    if len(universe) < 3 or len(prices) < config.train_sessions + 3:
        raise ValueError("Need at least three tradable assets and formation history plus three test sessions.")
    return prices


def fit_relationships(history, universe, config):
    """Fit each peer basket on historical data only; retain rejection reasons."""
    history = history.iloc[-config.train_sessions:]
    returns = history[universe].pct_change(fill_method=None).iloc[1:]
    correlations = returns.corr()
    models, audit = {}, []
    for ticker in universe:
        ranked = correlations[ticker].drop(ticker).dropna().sort_values(ascending=False, kind="stable")
        peers = ranked[ranked >= config.min_correlation].index[:config.peers].tolist()
        row = {"ticker": ticker, "train_start": history.index[0].date().isoformat(),
               "train_end": history.index[-1].date().isoformat(), "peers": peers,
               "accepted": False, "reason": "insufficient_correlation"}
        if not peers:
            audit.append(row)
            continue
        x = np.log(history[peers]).mean(axis=1).to_numpy()
        y = np.log(history[ticker]).to_numpy()
        design = np.column_stack([np.ones(len(x)), x])
        intercept, beta = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - intercept - beta * x
        sigma = float(residual.std(ddof=1))
        if sigma < 1e-6 or beta <= 0 or np.std(x) < 1e-8:
            row["reason"] = "degenerate_or_negative_relationship"
            audit.append(row)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, pvalue, _ = coint(y, x, trend="c", maxlag=5, autolag="aic")
        ar = np.linalg.lstsq(np.column_stack([np.ones(len(residual) - 1), residual[:-1]]), residual[1:], rcond=None)[0]
        phi = float(ar[1])
        half_life = float(np.log(0.5) / np.log(phi)) if 0 < phi < 1 else None
        basket_return = returns[peers].mean(axis=1)
        midpoint = len(returns) // 2
        first_corr = float(returns[ticker].iloc[:midpoint].corr(basket_return.iloc[:midpoint]))
        second_corr = float(returns[ticker].iloc[midpoint:].corr(basket_return.iloc[midpoint:]))
        row.update(pvalue=float(pvalue), half_life=half_life, beta=float(beta),
                   first_half_correlation=first_corr, second_half_correlation=second_corr)
        if not np.isfinite(pvalue) or pvalue > config.coint_pvalue:
            row["reason"] = "cointegration_filter"
        elif half_life is None or not config.min_half_life <= half_life <= config.max_half_life:
            row["reason"] = "half_life_filter"
        elif not np.isfinite(first_corr + second_corr) or min(first_corr, second_corr) < config.min_correlation * 0.75:
            row["reason"] = "unstable_return_correlation"
        else:
            row.update(accepted=True, reason="accepted")
            models[ticker] = {**row, "intercept": float(intercept), "residual_mean": float(residual.mean()),
                              "residual_std": sigma}
        audit.append(row)
    return models, audit


def model_z(model, close):
    basket = float(np.log(close[model["peers"]]).mean())
    residual = float(np.log(close[model["ticker"]])) - model["intercept"] - model["beta"] * basket
    return (residual - model["residual_mean"]) / model["residual_std"]


def market_allowed(prices, index, config):
    if not config.trend_filter:
        return True
    recent = prices[config.market].iloc[max(0, index - 199):index + 1]
    return len(recent) == 200 and float(recent.iloc[-1]) > float(recent.mean())


def formation_folds(prices, universe, config):
    folds = []
    for start in range(config.train_sessions, len(prices), config.test_sessions):
        stop = min(start + config.test_sessions, len(prices))
        # A last single session cannot support an entry and later exit.
        models, audit = fit_relationships(prices.iloc[start - config.train_sessions:start], universe, config)
        folds.append({"id": len(folds) + 1, "start": start, "stop": stop, "models": models, "audit": audit,
                      "train_end": prices.index[start - 1].date().isoformat(),
                      "test_start": prices.index[start].date().isoformat(),
                      "test_end": prices.index[stop - 1].date().isoformat()})
    return folds


def simulate_trades(prices, folds, config, *, cost_bps=None):
    """Cash/units ledger. Decisions at close t execute at close t+1."""
    rate = (config.cost_bps if cost_bps is None else cost_bps) / 10000
    cash = config.initial_capital
    positions, trades, daily, fills, fold_rows = {}, [], [], [], []
    previous_equity = config.initial_capital

    def close_position(ticker, execution, signal, reason, z):
        nonlocal cash
        position = positions.pop(ticker)
        price = float(prices[ticker].iloc[execution])
        notional = position["units"] * price
        fee = notional * rate
        cash += notional - fee
        trade = {**position, "ticker": ticker, "exit_signal_date": prices.index[signal].date().isoformat(),
                 "exit_date": prices.index[execution].date().isoformat(), "exit_price": price,
                 "exit_cost": fee, "exit_z": z, "exit_reason": reason,
                 "held_sessions": execution - position["entry_index"],
                 "gross_pnl": notional - position["entry_notional"],
                 "net_pnl": notional - fee - position["entry_notional"] - position["entry_cost"]}
        trade["net_return"] = trade["net_pnl"] / (position["entry_notional"] + position["entry_cost"])
        trades.append(trade)
        fills.append({"date": trade["exit_date"], "signal_date": trade["exit_signal_date"], "ticker": ticker,
                      "side": "SELL", "notional": notional, "cost": fee, "reason": reason})
        return fee, notional

    for fold in folds:
        fold_begin = previous_equity
        fold_trade_count = len(trades)
        models = fold["models"]
        for signal in range(fold["start"] - 1, fold["stop"] - 1):
            execution = signal + 1
            known = prices.iloc[signal]
            zs = {ticker: model_z(model, known) for ticker, model in models.items()}
            equity_at_signal = cash + sum(p["units"] * float(known[t]) for t, p in positions.items())
            exits = []
            for ticker, position in positions.items():
                z = zs[ticker]
                if z >= -config.exit_z:
                    reason = "spread_recovered"
                elif z <= -config.stop_z:
                    reason = "spread_stop"
                elif float(known[ticker]) / position["entry_price"] - 1 <= -config.stop_loss:
                    reason = "price_stop"
                elif execution - position["entry_index"] >= config.max_hold:
                    reason = "time_exit"
                else:
                    continue
                exits.append((ticker, reason, z))
            # No re-entry into an asset just exited on the same close.
            just_exited = {ticker for ticker, _, _ in exits}
            candidates = sorted((z, ticker) for ticker, z in zs.items()
                                if ticker not in positions and ticker not in just_exited
                                and -config.stop_z < z <= -config.entry_z)
            if not market_allowed(prices, signal, config) or execution == fold["stop"] - 1:
                candidates = []
            slots = max(0, config.max_positions - len(positions) + len(exits))
            candidates = candidates[:slots]
            costs = turnover = 0.0
            for ticker, reason, z in exits:
                fee, notional = close_position(ticker, execution, signal, reason, z)
                costs += fee
                turnover += notional
            for z, ticker in candidates:
                notional = min(equity_at_signal / config.max_positions, cash / (1 + rate))
                if notional <= 1e-8:
                    continue
                price = float(prices[ticker].iloc[execution])
                fee = notional * rate
                cash -= notional + fee
                if abs(cash) < 1e-8:
                    cash = 0.0
                positions[ticker] = {"units": notional / price, "entry_index": execution,
                                     "entry_date": prices.index[execution].date().isoformat(),
                                     "entry_signal_date": prices.index[signal].date().isoformat(),
                                     "entry_price": price, "entry_notional": notional, "entry_cost": fee,
                                     "entry_z": z, "fold": fold["id"], "peers": models[ticker]["peers"]}
                fills.append({"date": prices.index[execution].date().isoformat(),
                              "signal_date": prices.index[signal].date().isoformat(), "ticker": ticker,
                              "side": "BUY", "notional": notional, "cost": fee, "reason": "relative_value_entry"})
                costs += fee
                turnover += notional
            if execution == fold["stop"] - 1:
                # Fold boundaries are fixed in advance, not selected from returns.
                for ticker in list(positions):
                    fee, notional = close_position(ticker, execution, signal, "scheduled_fold_end", zs[ticker])
                    costs += fee
                    turnover += notional
            invested = sum(p["units"] * float(prices[t].iloc[execution]) for t, p in positions.items())
            equity = cash + invested
            daily.append({"date": prices.index[execution].date().isoformat(), "fold": fold["id"],
                          "equity": equity, "net_return": equity / previous_equity - 1,
                          "cost": costs, "turnover": turnover / previous_equity,
                          "cash": cash, "exposure": invested / equity if equity else 0.0,
                          "positions": len(positions)})
            previous_equity = equity
        fold_rows.append({"fold": fold["id"], "train_end": fold["train_end"],
                          "test_start": fold["test_start"], "test_end": fold["test_end"],
                          "accepted_models": len(models), "trades": len(trades) - fold_trade_count,
                          "net_return": previous_equity / fold_begin - 1})
    return {"daily": daily, "trades": trades, "fills": fills, "folds": fold_rows}


def performance(daily, capital, trades=None):
    if not daily:
        raise ValueError("No evaluated trading sessions.")
    returns = np.asarray([row["net_return"] for row in daily])
    wealth = np.r_[capital, [row["equity"] for row in daily]]
    std = float(returns.std(ddof=1)) if len(returns) > 1 else 0
    costs = sum(row.get("cost", 0) for row in daily)
    result = {"total_return": float(wealth[-1] / capital - 1),
              "cagr": float((wealth[-1] / capital) ** (252 / len(returns)) - 1),
              "annual_volatility": std * np.sqrt(252),
              "sharpe_zero_rf": float(returns.mean() / std * np.sqrt(252)) if std > 1e-12 else None,
              "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
              "ending_equity": float(wealth[-1]), "total_cost": float(costs),
              "annual_oneway_turnover": float(sum(row.get("turnover", 0) for row in daily) * 252 / len(daily)),
              "mean_exposure": float(np.mean([row.get("exposure", 1) for row in daily])),
              "sessions": len(daily)}
    if trades is not None:
        pnl = np.asarray([trade["net_pnl"] for trade in trades])
        result.update(trades=len(trades), win_rate=float((pnl > 0).mean()) if len(pnl) else None,
                      profit_factor=float(pnl[pnl > 0].sum() / -pnl[pnl < 0].sum()) if (pnl < 0).any() else None)
    return result


def buy_and_hold(prices, assets, start, config, cost_bps):
    panel = prices[assets].iloc[start:]
    rate = cost_bps / 10000
    each = config.initial_capital / len(assets) / (1 + rate)
    units = each / panel.iloc[0]
    previous = config.initial_capital
    rows = []
    for i, (date, close) in enumerate(panel.iterrows()):
        equity = float((close * units).sum())
        costs = len(assets) * each * rate if i == 0 else 0
        if i == len(panel) - 1:
            costs += equity * rate
            equity *= 1 - rate
        rows.append({"date": date.date().isoformat(), "equity": equity,
                     "net_return": equity / previous - 1, "cost": costs, "exposure": 1.0})
        previous = equity
    return {"daily": rows, "metrics": performance(rows, config.initial_capital)}


def backtest(prices, universe, config):
    prices = check_panel(prices, universe, config)
    folds = formation_folds(prices, universe, config)
    costs = sorted(set([0.0, 5.0, config.cost_bps, 25.0]))
    scenarios = []
    baseline_runs = {}
    for cost in costs:
        result = simulate_trades(prices, folds, config, cost_bps=cost)
        metrics = performance(result["daily"], config.initial_capital, result["trades"])
        scenarios.append({"cost_bps": cost, "metrics": metrics,
                          "equity": [r["equity"] for r in result["daily"]]})
        if cost == config.cost_bps:
            main = result
            main_metrics = metrics
    baseline_runs["Equal-weight buy and hold"] = buy_and_hold(prices, universe, config.train_sessions, config, config.cost_bps)
    baseline_runs[config.market + " buy and hold"] = buy_and_hold(prices, [config.market], config.train_sessions, config, config.cost_bps)
    return {"config": asdict(config), "universe": list(universe), "strategy": main,
            "metrics": main_metrics, "cost_scenarios": scenarios, "benchmarks": baseline_runs,
            "model_audit": [dict(row, fold=fold["id"]) for fold in folds for row in fold["audit"]],
            "dates": [row["date"] for row in main["daily"]],
            "execution": "Signals at close t; modeled adjusted-close fills at close t+1. No shorts or leverage.",
            "assumptions": [
                "Adjusted-close total-return proxies and fractional units; fills are modeled, not broker quotes.",
                "One-way costs include assumed spread, slippage, and commission, charged on buys and sells.",
                "Stops are evaluated at a close and filled at the next close; losses can exceed the stop threshold.",
                "Cash and risk-free return are set to zero; Sharpe uses that convention.",
                "Holdings are liquidated at predetermined fold boundaries and the final close, including costs.",
                "Long-only positions carry market exposure. This is not a market-neutral pairs strategy.",
                "Cointegration p-values are training filters after peer selection, not corrected significance claims.",
                "Fixed present-day universe; historical walk-forward results remain exploratory.",
                "Cost sensitivity is reported without selecting the best cost scenario or retuning the strategy.",
                "No brokerage orders are placed and no quantum hardware is used by this trading module.",
            ]}
