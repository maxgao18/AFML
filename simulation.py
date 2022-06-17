import itertools

import numpy as np
import pandas as pd

import c
import u
import signals


def _simulate_price_target_stop_loss_outcome(
    side, barrier_args, price_data_args, with_price: bool
):
    d = u.create_price_data(**price_data_args)
    barrier = signals.get_price_target_stop_loss_indices(
        d, [d.index[0]], **barrier_args
    )
    barrier = signals.add_profit_target_stop_loss_outcome(barrier)

    r = barrier.iloc[0]
    o = r["pt_sl"]
    if o == c.TripleBarrier.N:
        price = d.iloc[-1, 0]
    else:
        t = r["pt_dt"] if o == c.TripleBarrier.PT else r["sl_dt"]
        price = d.loc[t].iloc[0]

    profit = (price - d.iloc[0, 0]) * side
    if with_price:
        return profit, price
    return profit


def simulate_price_target_stop_loss_mesh(
    target: float,
    price_targets: list,
    stop_losses: list,
    price_data_args: dict,
    side=c.Dir.B,
    n_samples_per_point=1000,
    barrier_args: dict = {},
):
    results = []

    for pt, sl in itertools.product(price_targets, stop_losses):
        barrier_args_ = {
            "target": target,
            "price_target": pt,
            "stop_loss": sl,
            "side": side,
        }
        barrier_args_.update(barrier_args)

        profit = np.array(
            [
                _simulate_price_target_stop_loss_outcome(
                    side, barrier_args_, price_data_args, with_price=False
                )
                for _ in range(n_samples_per_point)
            ]
        )
        results.append([pt, sl, profit, profit.mean(), profit.std()])

    return pd.DataFrame(
        results, columns=["pt", "sl", "profit", "profit_mean", "profit_stddev"]
    )


def half_life_to_rho(hl: float):
    return 2 ** (-1 / hl)

def rho_to_half_life(rho: float):
    return - np.log(2) / np.log(rho)


def estimate_ou_parameters(series: pd.Series, target: float, as_series: bool = False):
    prices = series.iloc[1:].to_numpy()
    deviance = series.iloc[:-1].to_numpy() - target
    vcov = np.cov(prices, deviance)
    rho = vcov[0, 1] / vcov[1, 1]
    error = (prices - target) - rho * deviance
    stddev = error.std()
    if as_series:
        return pd.Series({"rho": rho, "stddev": stddev})
    return rho, stddev
