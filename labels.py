import pandas as pd
import numpy as np


def get_concurrent_label_count(data: pd.DataFrame, barriers: pd.DataFrame):
    times = barriers[["pt_dt", "sl_dt"]].copy()
    times["end"] = barriers["vb"].fillna(pd.NaT) + times.index
    times["max"] = data.index[-1]
    times = times.min(axis=1)

    count = pd.Series(0, index=data.index)
    for s, e in times.iteritems():
        count[s:e] += 1

    return count
