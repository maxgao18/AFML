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
        if pd.notnull(r):
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


def _column_or_default(group, colname, default: str = "Close"):
    return group[colname] if colname in group.obj.columns else group[default]


def _group_bars(data, indices):
    groups = data.reset_index().groupby(indices.cumsum())
    bars = groups[["Volume"]].sum()
    return groups, bars


def _get_bars(group):
    groups, bars = group
    bars.set_index(groups["Date"].first(), inplace=True)
    return bars


def _with_dv(group):
    groups, bars = group
    bars["dv"] = groups["dv"].sum()
    return groups, bars


def _with_open(group):
    groups, bars = group
    bars["Open"] = _column_or_default(groups, "Open").first()
    # bars["Open"] = groups["Close"].first()
    return groups, bars


def _with_low(group):
    groups, bars = group
    bars["Low"] = _column_or_default(groups, "Low").min()
    # bars["Low"] = groups["Close"].min()
    return groups, bars


def _with_high(group):
    groups, bars = group
    bars["High"] = _column_or_default(groups, "High").max()
    return groups, bars


def _with_close(group):
    groups, bars = group
    bars["Close"] = _column_or_default(groups, "Close").last()
    return groups, bars


def _apply(group, apply):
    groups, bars = group
    if apply is not None:
        bars = apply(groups, bars)
    return groups, bars


def _create_bars(data, indices):
    g = _group_bars(data, indices)
    g = _with_open(g)
    g = _with_close(g)
    g = _with_high(g)
    g = _with_low(g)
    return g


def create_tick_bars(data, rate, apply=None):
    data["tmp"] = 1
    indices = _create_bar_indices(data["tmp"], rate)
    g = _create_bars(data, indices)
    g = _apply(g, apply)
    return _get_bars(g)


def create_dollar_volume_bars(data, rate, apply=None):
    if "dv" not in data:
        data["dv"] = data["Volume"] * data["Close"]
    indices = _create_bar_indices(data["dv"], rate)
    g = _create_bars(data, indices)
    g = _with_dv(g)
    g = _apply(g, apply)
    return _get_bars(g)


def create_volume_bars(data, rate, apply=None):
    indices = _create_bar_indices(data["Volume"], rate)
    g = _create_bars(data, indices)
    g = _with_dv(g)
    g = _apply(g, apply)
    return _get_bars(g)


def create_tick_imbalance_bars(data, **kwargs):
    indices = _create_imbalance_bar_indices(data["dir"], **kwargs)
    g = _create_bars(data, indices)
    return _get_bars(g)


def create_volume_imbalance_bars(data, **kwargs):
    indices = _create_imbalance_bar_indices(data["Volume"] * data["dir"], **kwargs)
    g = _create_bars(data, indices)
    g = _with_dv(g)
    return _get_bars(g)


def create_dollar_volume_imbalance_bars(data, **kwargs):
    indices = _create_imbalance_bar_indices(data["dv"] * data["dir"], **kwargs)
    g = _create_bars(data, indices)
    g = _with_dv(g)
    return _get_bars(g)
