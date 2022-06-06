import pandas as pd
import numpy as np

import c


def apply_tick_rule(prices: pd.Series):
    direction = np.full(prices.shape[0], c.Dir.U)
    previous_prices = prices.shift(1)
    direction[0] = c.Dir.B
    direction[prices > previous_prices] = c.Dir.B
    direction[prices < previous_prices] = c.Dir.S

    for i, e in enumerate(direction):
        if e == c.Dir.U:
            direction[i] = direction[i - 1]

    return direction
