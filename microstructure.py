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


def corwin_schultz_spread_estimate(
    bars: pd.DataFrame, rolling_window, as_percent_of_price: bool = False, **kwargs
):
    hi = bars["High"]
    lo = bars["Low"]

    def beta():
        log_hi_lo_sq = np.square(np.log(hi / lo))
        return (
            (log_hi_lo_sq + log_hi_lo_sq.shift(1))
            .rolling(rolling_window, **kwargs)
            .mean()
        )

    def gamma():
        hi_window = pd.DataFrame([hi, hi.shift(1)]).max()
        lo_window = pd.DataFrame([lo, lo.shift(1)]).min()

        return np.square(np.log(hi_window / lo_window))

    def alpha():
        c1 = (np.sqrt(2) - 1) / (3 - 2 * np.sqrt(2))
        c2 = 1 / np.sqrt(3 - 2 * np.sqrt(2))
        a = c1 * np.sqrt(beta()) - c2 * np.sqrt(gamma())
        return a.clip(lower=0)

    def spread_as_percent():
        a = np.exp(alpha())
        return 2 * (a - 1) / (1 + a)

    spread = spread_as_percent()
    if not as_percent_of_price:
        spread *= bars["Close"]
    return spread


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
