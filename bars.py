import sys

import numpy as np
import pandas as pd


def _create_bar_indices(values: pd.Series, thres: float):
    indices = np.full(values.size, False, dtype=bool)
    csum = 0.0
    for e, (_, r) in enumerate(values.iteritems()):
        if csum >= thres:
            csum = 0.0
            indices[e] = True
        csum += r
    return indices


def _create_imbalance_bar_indices(
    values: pd.Series,
    expected_length: float,
    expected_abs_inbalance: float,
    min_observation_weight: float = 0.01,
    max_length: int = sys.maxsize,
):
    observations = 0
    indices = np.full(values.size, False)
    current_length = 0
    current_inbalance = 0.0
    for i, (_, r) in enumerate(values.iteritems()):
        if (
            abs(current_inbalance) > np.sqrt(expected_length) * expected_abs_inbalance
            or current_length >= max_length
        ):
            indices[i] = True
            observations += 1
            w = max(min_observation_weight, 1 / observations)
            mean_inbalance = current_inbalance / current_length

            expected_length = w * current_length + (1 - w) * expected_length
            expected_abs_inbalance = (
                w * abs(mean_inbalance) + (1 - w) * expected_abs_inbalance
            )

            current_inbalance = 0.0
            current_length = 0

        current_inbalance += r
        current_length += 1
    return indices


def _group_bars(data, indices):
    groups = data.reset_index().groupby(indices.cumsum())
    bars = groups[["volume"]].sum()
    return groups, bars


def _get_bars(group):
    groups, bars = group
    bars.set_index(groups["index"].first(), inplace=True)
    return bars


def _with_dv(group):
    groups, bars = group
    bars["dv"] = groups["dv"].sum()
    return groups, bars


def _with_open(group):
    groups, bars = group
    bars["open"] = groups["close"].first()
    return groups, bars


def _with_low(group):
    groups, bars = group
    bars["low"] = groups["close"].min()
    return groups, bars


def _with_high(group):
    groups, bars = group
    bars["high"] = groups["close"].max()
    return groups, bars


def _with_close(group):
    groups, bars = group
    bars["close"] = groups["close"].last()
    return groups, bars


def _create_bars(data, indices):
    g = _group_bars(data, indices)
    g = _with_open(g)
    g = _with_close(g)
    g = _with_high(g)
    g = _with_low(g)
    return g


def create_tick_bars(data, rate):
    data["tmp"] = 1
    indices = _create_bar_indices(data["tmp"], rate)
    g = _create_bars(data, indices)
    return _get_bars(g)


def create_dollar_volume_bars(data, rate):
    indices = _create_bar_indices(data["dv"], rate)
    g = _create_bars(data, indices)
    g = _with_dv(g)
    return _get_bars(g)


def create_volume_bars(data, rate):
    indices = _create_bar_indices(data["volume"], rate)
    g = _create_bars(data, indices)
    g = _with_dv(g)
    return _get_bars(g)


def create_tick_imbalance_bars(data, **kwargs):
    indices = _create_imbalance_bar_indices(data["dir"], **kwargs)
    g = _create_bars(data, indices)
    return _get_bars(g)


def create_volume_imbalance_bars(data, **kwargs):
    indices = _create_imbalance_bar_indices(data["volume"] * data["dir"], **kwargs)
    g = _create_bars(data, indices)
    g = _with_dv(g)
    return _get_bars(g)


def create_dollar_volume_imbalance_bars(data, **kwargs):
    indices = _create_imbalance_bar_indices(data["dv"] * data["dir"], **kwargs)
    g = _create_bars(data, indices)
    g = _with_dv(g)
    return _get_bars(g)
