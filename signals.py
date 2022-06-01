import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as sts

from collections.abc import Iterable

import c


def get_cusum_indices(diff: pd.Series, threshold: float):
    indices = np.full(diff.size, False)

    positive_cusum = 0.0
    negative_cusum = 0.0
    for e, (_, r) in enumerate(diff.iteritems()):
        positive_cusum = max(0, positive_cusum + r)
        negative_cusum = min(0, negative_cusum + r)
        if positive_cusum > threshold or negative_cusum < -threshold:
            positive_cusum = 0
            negative_cusum = 0
            indices[e] = True
    return indices


def get_cusum_indices_on_pct_returns(
    prices: pd.Series, percent_threshold: float = 0.05
):
    returns = prices.shift(1) / prices - 1
    returns.fillna(0)
    return get_cusum_indices(returns, percent_threshold)


def _price_target_stop_loss_indices(
    prices: pd.Series, target: float, side, price_target=None, stop_loss=None
):
    if side == c.Dir.S:
        price_target, stop_loss = stop_loss, price_target

    upper_idx = pd.NaT
    lower_idx = pd.NaT

    if price_target is not None:
        upper_target = prices[0] * (1 + target * price_target)
        for i, p in prices.iteritems():
            if p >= upper_target:
                upper_idx = i
                break

    if stop_loss is not None:
        lower_target = prices[0] * (1 - target * stop_loss)
        for i, p in prices.iteritems():
            if p <= lower_target:
                lower_idx = i
                break

    if side == c.Dir.S:
        return (lower_idx, upper_idx)

    return (upper_idx, lower_idx)


def get_price_target_stop_loss_indices(
    data: pd.DataFrame,
    indices,
    target: float,
    side=c.Dir.B,
    price_target=None,
    stop_loss=None,
    vertical_barrier=None,
):
    prices = data["close"]
    barriers = pd.DataFrame(columns=["pt_dt", "sl_dt"])

    if not isinstance(side, Iterable):
        side = np.full(len(indices), side)

    for s, i in zip(side, indices):
        cond = data.index >= i
        if vertical_barrier is not None:
            cond = cond & (data.index <= i + vertical_barrier)
        u, l = _price_target_stop_loss_indices(
            prices.loc[cond], target, s, price_target, stop_loss
        )
        barriers.loc[i] = pd.Series(data={"pt_dt": u, "sl_dt": l}, name=data.index[0])

    barriers["side"] = side
    barriers["vb"] = vertical_barrier
    return barriers


def add_profit_target_stop_loss_outcome(barriers: pd.DataFrame):
    pt = pd.notnull(barriers["pt_dt"])
    sl = pd.notnull(barriers["sl_dt"])
    barriers["pt_sl"] = (
        ((barriers["pt_dt"] < barriers["sl_dt"]) | (pt & ~sl)) * c.TripleBarrier.PT
        + ((barriers["pt_dt"] > barriers["sl_dt"]) | (~pt & sl)) * c.TripleBarrier.SL
        + (~pt & ~sl) * c.TripleBarrier.N
    )
    barriers["pt"] = barriers["pt_sl"] == c.TripleBarrier.PT
    return barriers


def _get_fractional_differentiation_weights(derivative, threshold=None, size=None):
    w = [1.0]
    if size is not None:
        for k in range(1, size):
            w_ = -w[-1] * (derivative - k + 1) / k
            w.append(w_)
    else:
        k = 1
        while abs(w[-1]) > threshold:
            w_ = -w[-1] * (derivative - k + 1) / k
            w.append(w_)
            k += 1
    return np.array(w)


def get_fixed_window_fractional_differentiated_series(
    series: pd.Series, derivative: float, threshold=1e-5, size=None
):
    weights = _get_fractional_differentiation_weights(derivative, threshold, size)
    weights = np.flip(weights)
    size = len(weights)
    values = np.full(series.size, np.nan)

    for i in range(size, series.size):
        values[i] = np.dot(weights, series.iloc[i - size : i])
    return values


def get_minimally_fractional_differentiated_series(
    series: pd.Series,
    tol=1e-2,
    p_value_threshold=0.01,
    window_threshold=1e-5,
    window_size=None,
):
    p = sts.adfuller(series)[1]
    if p < p_value_threshold:
        return series, 0, p

    lower_bound, upper_bound = 0, 2
    while upper_bound - lower_bound > tol:
        mid = (upper_bound + lower_bound) / 2
        fd = get_fixed_window_fractional_differentiated_series(
            series, mid, threshold=window_threshold, size=window_size
        )
        fd = fd[~np.isnan(fd)]
        p = st.adfuller(fd)[1]
        if p < p_value_threshold:
            upper_bound = mid
        else:
            lower_bound = mid

    fd = get_fixed_window_fractional_differentiated_series(
        series, upper_bound, threshold=window_threshold, size=window_size
    )
    fd_test = fd[~np.isnan(fd)]
    p = st.adfuller(fd_test)[1]
    return fd, upper_bound, p


def brown_durbin_evans_residuals(
    predictor: pd.DataFrame,
    response: pd.Series,
    num_residuals: int,
    add_constant: bool = True,
    normalize: bool = True,
):
    index = predictor.index
    predictor = predictor.to_numpy()
    if add_constant:
        predictor = sm.add_constant(predictor)
    response = response.to_numpy()

    start_index = predictor.shape[0] - num_residuals - 1
    models = [
        sm.OLS(response[: start_index + i], predictor[: start_index + i])
        for i in range(num_residuals)
    ]
    results = []
    fit = None
    for m, x, y in zip(models, predictor[-num_residuals:], response[-num_residuals:]):
        fit = m.fit()
        error = y - fit.predict(x)[0]
        vcov = fit.cov_params()
        if isinstance(vcov, pd.DataFrame):
            vcov = vcov.to_numpy()
        se = np.sqrt(1 + np.dot(x, np.dot(x, vcov)))
        results.append(error / se)
    results /= np.sqrt(fit.scale)
    results = pd.Series(results, index=index[-num_residuals:])
    if normalize:
        results /= results.std()
    return results


def chu_stinchcombe_white_departure(series: pd.Series, start_index=None):
    partial_series = series if start_index is None else series[start_index:]

    rolling_std = np.square(series.diff().dropna()).cumsum()
    rolling_std /= np.arange(1, series.size)
    rolling_std = np.sqrt(rolling_std)

    change = partial_series - partial_series.iloc[0]
    scaled_std = rolling_std[change.index] * np.sqrt(np.arange(0, change.size))
    scaled_std.replace(0, np.nan, inplace=True)

    return (change / scaled_std).fillna(0)


def chu_stinchcombe_white_critical_value(
    series: pd.Series, b_alpha=4.6, start_index=None
):
    series = series if start_index is None else series[start_index:]
    critical_values = np.sqrt(b_alpha + np.log(np.arange(1, len(series.index))))
    return pd.Series(critical_values, index=series.iloc[1:].index)
