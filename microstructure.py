import math

import pandas as pd
import numpy as np
import statsmodels.api as sm

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


def amihuds_lambda_estimator(
    bars,
    rolling_window=None,
    use_close_close=True,
    with_t_value: bool = False,
    **kwargs
):
    def amihuds_lambda(bars):
        bars = bars.dropna()
        if len(bars.index) <= 2:
            if with_t_value:
                return np.nan, np.nan
            return np.nan

        fit = sm.OLS(bars.iloc[:, 0].to_numpy(), bars.iloc[:, 1].to_numpy()).fit()
        if with_t_value:
            return fit.params[0], fit.tvalues[0]
        return fit.params[0]

    log_price_change = (
        np.log(bars["Open"])
        if "Open" in bars and not use_close_close
        else np.log(bars["Close"].shift(1))
    )
    log_price_change = np.abs(log_price_change - np.log(bars["Close"]))
    dv = bars["dv"] if "dv" in bars else bars["Close"] * bars["Volume"]

    data = pd.concat([log_price_change, dv], axis=1)
    if rolling_window is None:
        return amihuds_lambda(data)

    # This code is garbage
    result = []
    rolling = data.iloc[:, 0].rolling(rolling_window, **kwargs)
    rolling.apply(
        lambda x: result.append(amihuds_lambda(data.loc[x.index])) or 0, raw=False
    )
    index = rolling.apply(lambda _: 1, raw=False)

    if with_t_value:
        d = pd.Series([s[0] for s in result], index=index.dropna().index)
        d.name = "Amihuds Lambda"
        t = pd.Series([s[1] for s in result], index=index.dropna().index)
        t.name = "t-stat"
        return d, t
    return pd.Series(result, index=i.index)


def volume_probability_of_informed_trading(bars, **kwargs):
    buy_volume = None
    use_mean = True
    if "buy_volume" in bars:
        buy_volume = bars["buy_volume"]
    elif "avg_dir" in bars:
        buy_volume = bars["Volume"] * (bars["avg_dir"] + 1) / 2
    else:
        buy_volume = (bars["Volume"] * bars["dir"]).clip(lower=0)
        use_mean = False

    if buy_volume is None:
        return None

    sell_volume = bars["Volume"] - buy_volume
    inbalance = sell_volume - buy_volume
    if use_mean:
        inbalance = inbalance.abs()
    inbalance = inbalance.rolling(**kwargs).sum().abs()
    total_volume = bars["Volume"].rolling(**kwargs).sum()

    return inbalance / total_volume
