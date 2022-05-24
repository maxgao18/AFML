import pandas as pd
import numpy as np


def _effective_barrier_end(barriers: pd.DataFrame, last_date):
    times = barriers[["pt_dt", "sl_dt"]].copy()
    times["end"] = barriers["vb"].fillna(pd.NaT) + times.index
    times["max"] = last_date
    times = times.min(axis=1)

    return times


def get_concurrent_label_count(data: pd.DataFrame, barriers: pd.DataFrame):
    times = _effective_barrier_end(barriers, data.index[-1])
    count = pd.Series(0, index=data.index)
    for s, e in times.iteritems():
        count[s:e] += 1

    return count


def get_label_uniqueness(barriers: pd.DataFrame, concurrent_label_count: pd.Series):
    times = _effective_barrier_end(barriers, concurrent_label_count.index[-1])
    uniqueness = pd.Series(0, index=barriers.index)
    for s, e in times.iteritems():
        uniqueness.loc[s] = (1.0 / concurrent_label_count.loc[s:e]).mean()

    return uniqueness
