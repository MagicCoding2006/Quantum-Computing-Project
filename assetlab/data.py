"""Price sources and explicit, conservative return alignment."""

from datetime import datetime, timezone
import csv
import hashlib
import io
from pathlib import Path
import re
import time
import zlib

import numpy as np
import pandas as pd


DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "TLT", "IEF", "GLD", "SLV", "XLE", "XLF", "XLK",
    "XLV", "XLI", "XLP", "XLU", "VNQ", "EFA", "EEM", "HYG", "LQD", "DBC",
]


def ticker_list(values):
    tickers = [v.strip().upper() for v in values]
    if len(tickers) < 2 or len(set(tickers)) != len(tickers):
        raise ValueError("Supply at least two distinct tickers.")
    if any(not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^=\-]{0,19}", t) for t in tickers):
        raise ValueError("Tickers must be 1-20 letters, numbers, '.', '-', '^', or '='.")
    if len(tickers) > 100:
        raise ValueError("This prototype supports at most 100 assets per run.")
    return tickers


def validate_prices(prices, tickers):
    prices = prices.copy()
    if prices.columns.duplicated().any():
        raise ValueError("Price input contains duplicate ticker columns.")
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError("No price column for: " + ", ".join(missing))
    prices = prices[tickers]
    prices.index = pd.to_datetime(prices.index, errors="raise", utc=True).tz_convert(None)
    if prices.index.hasnans or prices.index.duplicated().any():
        raise ValueError("Price dates must be present and unique.")
    if (prices.index != prices.index.normalize()).any():
        raise ValueError("Use daily prices with date-only timestamps.")
    prices = prices.sort_index().apply(pd.to_numeric, errors="raise").astype(float)
    values = prices.to_numpy()
    if np.isinf(values).any() or (values[np.isfinite(values)] <= 0).any():
        raise ValueError("Adjusted prices must be positive and finite, or missing.")
    if prices.empty or prices.isna().all().any():
        raise ValueError("Each requested asset needs price observations.")
    return prices


def demo_prices(tickers, start, end, seed=42):
    """Synthetic factor returns, with shared relationships and a regime change.

    Dates are weekdays, not an exchange calendar. These are never market quotes.
    """
    dates = pd.bdate_range(start, end, inclusive="left")
    rng = np.random.default_rng(seed)
    factors = rng.normal(0, [0.009, 0.004, 0.008], (len(dates), 3))
    factors[int(len(dates) * 0.75):, 1] += factors[int(len(dates) * 0.75):, 0] * 0.3
    groups = {
        "SPY": 0, "QQQ": 0, "IWM": 0, "TLT": 1, "IEF": 1,
        "GLD": 2, "SLV": 2, "LQD": 1, "HYG": 0, "DBC": 2,
    }
    output = {}
    for ticker in tickers:
        key = zlib.crc32(ticker.encode())
        local = np.random.default_rng(np.random.SeedSequence([seed, key]))
        group = groups.get(ticker, 0 if key % 3 else 2)
        loading = np.array([0.1, 0.05, 0.05])
        loading[group] += local.uniform(0.7, 1.2)
        ret = 0.0001 + factors @ loading + local.normal(0, 0.003, len(dates))
        output[ticker] = 100 * np.exp(np.cumsum(ret))
    return pd.DataFrame(output, index=dates).rename_axis("date")


def load_prices(tickers, start, end, *, demo=False, csv_path=None,
                cache_dir="data_cache/asset_lab", refresh=False, seed=42):
    provenance = {"requested_start": start, "requested_end_exclusive": end}
    if demo:
        prices = demo_prices(tickers, start, end, seed)
        provenance.update(source="synthetic", seed=seed, cache_hit=False)
    elif csv_path:
        raw = Path(csv_path).read_bytes()
        header = next(csv.reader(io.StringIO(raw.decode("utf-8-sig"))), [])
        if len(header) < 3 or len(set(header[1:])) != len(header[1:]):
            raise ValueError("Price CSV needs a date column and at least two distinct ticker columns.")
        prices = pd.read_csv(io.BytesIO(raw), index_col=0)
        provenance.update(source="csv", input_sha256=hashlib.sha256(raw).hexdigest(),
                          input_name=Path(csv_path).name, cache_hit=False)
    else:
        import yfinance as yf

        key = hashlib.sha256(repr((tickers, start, end, "adjusted-v1")).encode()).hexdigest()[:20]
        cache = Path(cache_dir) / f"{key}.csv"
        age = time.time() - cache.stat().st_mtime if cache.exists() else float("inf")
        cached = not refresh and 0 <= age < 24 * 60 * 60
        if cached:
            prices = pd.read_csv(cache, index_col=0)
        else:
            raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                              progress=False, threads=False, timeout=20)
            if raw is None or raw.empty:
                raise ValueError("Yahoo returned no prices. Retry later, use --prices, or run --demo.")
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
            prices = validate_prices(prices, tickers)
            cache.parent.mkdir(parents=True, exist_ok=True)
            temp = cache.with_suffix(".tmp.csv")
            prices.to_csv(temp)
            temp.replace(cache)
        provenance.update(source="yahoo_adjusted_close", cache_hit=cached,
                          provider_version=yf.__version__,
                          cache_timestamp_utc=datetime.fromtimestamp(
                              cache.stat().st_mtime, timezone.utc).isoformat())
    prices = validate_prices(prices, tickers)
    prices = prices.loc[(prices.index >= pd.Timestamp(start)) & (prices.index < pd.Timestamp(end))]
    if prices.empty:
        raise ValueError("No price observations fall inside the requested dates.")
    provenance["snapshot_sha256"] = hashlib.sha256(prices.to_csv().encode()).hexdigest()
    return prices, provenance


def align_returns(prices, minimum=252):
    # Calculate before dropping gaps: dropping missing prices first would turn a
    # multi-session move into a daily return. Never fill a missing quote.
    raw = prices.pct_change(fill_method=None)
    returns = raw.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if len(returns) < minimum:
        raise ValueError(f"Need at least {minimum} common daily returns; found {len(returns)}. "
                         "Use a longer history or assets with better overlapping coverage.")
    constant = returns.columns[returns.std() < 1e-12].tolist()
    if constant:
        raise ValueError("Cannot correlate assets with constant returns: " + ", ".join(constant))
    quality = {
        "price_rows": len(prices), "return_rows": len(returns),
        "excluded_return_rows": len(raw) - len(returns),
        "missing_prices": {t: int(prices[t].isna().sum()) for t in prices},
        "observed_start": prices.index.min().date().isoformat(),
        "observed_end": prices.index.max().date().isoformat(),
        "return_alignment": "Complete common return rows; no price filling or gap bridging.",
    }
    return returns, quality
