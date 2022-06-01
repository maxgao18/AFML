import os

import pandas as pd


def _all_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def fetch(path, files, **kwargs):
    if isinstance(files, str):
        files = [files]

    data = {}
    for f in files:
        name = f.split(".csv")[0]
        data[name] = pd.read_csv(
            os.path.join(path, f), index_col="Date", parse_dates=["Date"], **kwargs
        )

    if len(files) == 1:
        return list(data.values())[0]
    return data


def fetch_all(path, **kwargs):
    return fetch(path, _all_files(path), **kwargs)
