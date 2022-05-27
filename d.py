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
        data[name] = pd.read_csv(os.path.join(path, f), **kwargs)

    if len(files) == 1:
        return data[files[0]]
    return data


def fetch_all(path, **kwargs):
    return fetch(path, _all_files(path), **kwargs)
