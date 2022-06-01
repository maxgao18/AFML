import numpy as np
import pandas as pd
import datetime as dt

from sklearn.datasets import make_classification

import c


def _ou_process(
    theta: float = 0.001, mu: float = 0.0, var: float = 0.01, n_samples: int = 1000000
):
    a = np.random.normal(mu, np.sqrt(var), n_samples)
    a[0] = 0
    for i in range(1, n_samples):
        a[i] += (1 - theta) * a[i - 1]
    return a


def create_price_data(
    start_price: float = 1000.00,
    theta: float = 0.001,
    mu: float = 0.0,
    var: float = 0.01,
    n_samples: int = 1000000,
):
    i = np.exp(_ou_process(theta, mu, var, n_samples)) * start_price
    df0 = pd.date_range(
        periods=n_samples, freq=pd.tseries.offsets.Minute(), end=dt.datetime.today()
    )
    X = pd.Series(i, index=df0, name="Close/Last").round(2).to_frame()
    X.index.name = "Date"
    return X


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
