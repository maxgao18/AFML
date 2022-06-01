import os
import argparse

import pandas as pd
import numpy as np


def _all_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("dir", type=str, help="Directory")
    parser.add_argument("tkr", type=str, help="Ticker")
    args = parser.parse_args()

    directory = args.dir
    ticker = args.tkr

    files = [f for f in _all_files(directory) if f.startswith(ticker)]
    dfs = [
        pd.read_csv(os.path.join(directory, f), index_col="Date", parse_dates=["Date"])
        for f in files
    ]
    df = pd.concat(dfs)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True, ascending=False)
    df.rename(columns={"Vol.": "Volume", "Price": "Close/Last"}, inplace=True)
    df.drop(columns=["Change %"], inplace=True)

    print(df.dtypes)
    for c in ["Close/Last", "Open", "High", "Low"]:
        df[c] = df[c].astype(str).str.replace(",", "").astype(float)

    for c in ["Volume"]:
        df[c] = (
            pd.to_numeric(
                df[c]
                .replace({"K": "*1e3", "M": "*1e6", "-": "'nan'"}, regex=True)
                .map(pd.eval),
                errors="coerce",
            )
            .round()
            .astype("Int64")
        )

    df.to_csv(os.path.join(directory, f"{ticker}.csv"))
