import pandas as pd

from matplotlib.patches import Rectangle

import c


def plot_triple_barrier(
    ax,
    market_data: pd.DataFrame,
    barriers: pd.DataFrame,
    target: float,
    price_target=None,
    stop_loss=None,
    vertical_barrier=None,
):
    has_side = "side" in barriers.columns
    has_pt_sl = "pt_sl" in barriers.columns

    for start_dt in barriers.index:
        side = barriers.at[start_dt, "side"] if has_side else c.Dir.B
        if side == c.Dir.S:
            price_target, stop_loss = stop_loss, price_target

        initial_price = market_data.at[start_dt, "Close"]
        price_target_dt = barriers.at[start_dt, "pt_dt"]
        stop_loss_dt = barriers.at[start_dt, "sl_dt"]

        width = (
            market_data.index[-1] - start_dt
            if vertical_barrier is None
            else vertical_barrier
        )

        if price_target is not None:
            upper_target = initial_price * (1 + target * price_target)
            rect = Rectangle(
                (start_dt, market_data.at[start_dt, "Close"]),
                width,
                upper_target - initial_price,
                linestyle="dashed",
                facecolor="None",
                edgecolor="g" if side == c.Dir.B else "r",
            )
            ax.add_patch(rect)
        if stop_loss is not None:
            lower_target = initial_price * (1 - target * stop_loss)
            rect = Rectangle(
                (start_dt, market_data.at[start_dt, "Close"]),
                width,
                lower_target - initial_price,
                linestyle="dashed",
                facecolor="None",
                edgecolor="r" if side == c.Dir.B else "g",
            )
            ax.add_patch(rect)
        if pd.notnull(price_target_dt):
            marker = (
                "x"
                if has_pt_sl and barriers.at[start_dt, "pt_sl"] != c.TripleBarrier.PT
                else 6
            )
            ax.plot(
                price_target_dt,
                market_data.at[price_target_dt, "Close"],
                color="g",
                marker=marker,
                markersize=10,
            )
        if pd.notnull(stop_loss_dt):
            marker = (
                "x"
                if has_pt_sl and barriers.at[start_dt, "pt_sl"] != c.TripleBarrier.SL
                else 7
            )
            ax.plot(
                stop_loss_dt,
                market_data.at[stop_loss_dt, "Close"],
                color="r",
                marker=marker,
                markersize=10,
            )
