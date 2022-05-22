import pandas as pd
import numpy as np

import c


def get_cusum_indices(prices: pd.Series, percentage_threshold: float = 0.05):
    returns = prices.shift(1) / prices - 1
    returns.fillna(0)
    indices = np.full(prices.size, False)

    positive_cusum = 0.0
    negative_cusum = 0.0
    for e, (_, r) in enumerate(returns.iteritems()):
        positive_cusum = max(0, positive_cusum + r)
        negative_cusum = min(0, negative_cusum + r)
        if (
            positive_cusum > percentage_threshold
            or negative_cusum < -percentage_threshold
        ):
            positive_cusum = 0
            negative_cusum = 0
            indices[e] = True
    return indices


def _price_target_stop_loss_indices(
    prices: pd.Series, target: float, side=c.Dir.B, price_target=None, stop_loss=None
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

    for i in indices:
        cond = data.index >= i
        if vertical_barrier is not None:
            cond = cond & (data.index <= i + vertical_barrier)
        u, l = _price_target_stop_loss_indices(
            prices.loc[cond], target, side, price_target, stop_loss
        )
        barriers.loc[i] = pd.Series(data={"pt_dt": u, "sl_dt": l}, name=data.index[0])

    return barriers


def add_profit_target_stop_loss_outcome(barriers: pd.DataFrame):
    pt = pd.isnull(barriers["pt_dt"])
    sl = pd.isnull(barriers["sl_dt"])
    barriers["pt_sl"] = (
        ((barriers["pt_dt"] < barriers["sl_dt"]) | (~pt & sl)) * c.TripleBarrier.PT
        + ((barriers["pt_dt"] > barriers["sl_dt"]) | (pt & ~sl)) * c.TripleBarrier.SL
        + (~pt & ~sl) * c.TripleBarrier.N
    )
    return barriers
