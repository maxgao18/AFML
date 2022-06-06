import os
import json

import numpy as np
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


def fetch_book(*args, **kwargs):
    books_data = fetch(*args, **kwargs)
    books_data["Bids"] = books_data["Bids"].apply(json.loads)
    books_data["Asks"] = books_data["Asks"].apply(json.loads)
    books_data["best_bid"] = books_data["Bids"].apply(
        lambda x: max(float(k) for k in x.keys()) if len(x) > 0 else np.nan
    )
    books_data["best_ask"] = books_data["Asks"].apply(
        lambda x: min(float(k) for k in x.keys()) if len(x) > 0 else np.nan
    )
    books_data["spread"] = books_data["best_ask"] - books_data["best_bid"]
    return books_data


def fetch_all(path, **kwargs):
    return fetch(path, _all_files(path), **kwargs)
