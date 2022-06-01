import argparse

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort price data")
    parser.add_argument("file", type=str, help="File")
    args = parser.parse_args()

    f = args.file

    df = pd.read_csv(f, index_col="Date", parse_dates=["Date"])
    df.sort_index(inplace=True, ascending=True)
    df.to_csv(f)
