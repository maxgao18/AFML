import numpy as np
import pandas as pd
import datetime as dt

from sklearn.datasets import make_classification

import c


def index_as_string(df):
    df = df.copy()
    df.index = df.index.to_series().apply(str)
    return df


def as_timeseries(data, round_to=None):
    n_samples = len(data)
    index = pd.date_range(
        periods=n_samples, freq=pd.tseries.offsets.Minute(), end=dt.datetime.today()
    )
    X = pd.Series(data, index=index, name="Close")
    if round_to is not None:
        X = X.round(2)
    X = X.to_frame()
    X.index.name = "Date"
    return X


def ar1_process(
    rho: float, mu: float, stddev: float, n_samples: int, initial_value: float = 0
):
    a = np.random.normal(mu, stddev, n_samples)
    a[0] = initial_value
    for i in range(1, n_samples):
        a[i] += rho * a[i - 1]
    return a


def create_price_data(
    start_price: float = 1000.00,
    theta: float = 0.001,
    mu: float = 0.0,
    stddev: float = 0.01,
    n_samples: int = 1000000,
):
    i = np.exp(ar1_process(1 - theta, mu, stddev, n_samples)) * start_price
    return as_timeseries(i, round_to=2)


def explosive_process_data(
    start_price: float = 0,
    delta: float = 0.0001,
    explosive_start_index=None,
    explosive_start_index_percent: float = 0.50,
    noise_stddev: float = 0.01,
    n_samples: int = 10000,
):
    if explosive_start_index is None:
        explosive_start_index = int(n_samples * explosive_start_index_percent)

    if explosive_start_index == 0:
        return as_timeseries(
            ar1_process(1 + delta, 0, noise_stddev, n_samples, start_price)
        )

    noise = np.random.normal(scale=noise_stddev, size=explosive_start_index)
    noise[0] = start_price
    noise = noise.cumsum()
    return as_timeseries(
        np.append(
            noise,
            ar1_process(
                1 + delta,
                0,
                noise_stddev,
                n_samples - explosive_start_index,
                noise[-1] + np.random.normal(scale=noise_stddev),
            ),
        )
    )


def during_market_hours(df, market_open=dt.time(9, 30), market_close=dt.time(16, 0)):
    return df.iloc[df.index.indexer_between_time(market_open, market_close)]


def add_volume_data(df, mu: float = 100.0, var: float = 10.0):
    scale = var / mu
    shape = mu / scale
    i = np.random.gamma(shape, scale, len(df.index))
    df["Volume"] = np.ceil(i)
    return df


def add_dir_data(df, random=False):
    direction = None
    if random:
        direction = np.random.choice([c.Dir.B, c.Dir.S], len(df.index))
    else:
        direction = np.full(len(df.index), c.Dir.U)
        closes = df["Close"]
        previous_closes = df["Close"].shift(1)
        direction[0] = c.Dir.B
        direction[closes > previous_closes] = c.Dir.B
        direction[closes < previous_closes] = c.Dir.S

        for i, e in enumerate(direction):
            if e == c.Dir.U:
                direction[i] = direction[i - 1]

    df["dir"] = direction
    return df
