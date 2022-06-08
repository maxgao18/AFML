import math

import pandas as pd
import numpy as np

import c


def apply_tick_rule(prices: pd.Series):
    direction = np.full(prices.shape[0], c.Dir.U)
    previous_prices = prices.shift(1)
    direction[0] = c.Dir.B
    direction[prices > previous_prices] = c.Dir.B
    direction[prices < previous_prices] = c.Dir.S

    for i, e in enumerate(direction):
        if e == c.Dir.U:
            direction[i] = direction[i - 1]

    return direction


def rolls_spread_estimate(trade_prices: pd.Series):
    diff = trade_prices.diff().dropna()
    return 2 * np.sqrt(-diff.autocorr() * diff.var())


def volatility_estimator(
    numerator_column_name: str,
    denominator_column_name: str,
    k: float,
    bars: pd.DataFrame,
    subtract_mean=True,
    period=None,
    use_index=True,
    rolling_window=None,
    min_periods=10,
    **kwargs
):
    def volatility(bars):
        observations = np.log(
            bars[numerator_column_name] / bars[denominator_column_name]
        )
        if subtract_mean:
            observations -= observations.mean()
        volatility = np.sqrt(np.square(observations).mean() / k)
        if period is not None:
            if isinstance(bars.index, pd.DatetimeIndex) and use_index:
                average_bar_length_years = (
                    bars.index[-1] - bars.index[0]
                ).total_seconds() / len(bars.index)
                volatility *= np.sqrt(period.total_seconds() / average_bar_length_years)
            else:
                volatility *= np.sqrt(period)
        return volatility

    if rolling_window is None:
        return volatility(bars)

    rolling = bars.rolling(rolling_window, min_periods=min_periods, **kwargs)
    vol = rolling["High"].apply(lambda x: volatility(bars.loc[x.index]), raw=False)
    vol.name = "Volatility"
    return vol


def high_low_volatility_estimator(*args, **kwargs):
    k = 4 * np.log(2)
    return volatility_estimator("High", "Low", k, *args, subtract_mean=False, **kwargs)


def open_close_volatility_estimator(*args, **kwargs):
    return volatility_estimator("Close", "Open", 1, *args, **kwargs)


def close_close_volatility_estimator(bars, *args, **kwargs):
    bars = bars.copy()
    bars["prev_close"] = bars["Close"].shift(1)
    bars = bars.iloc[1:]
    return volatility_estimator("Close", "prev_close", 1, bars, *args, **kwargs)
