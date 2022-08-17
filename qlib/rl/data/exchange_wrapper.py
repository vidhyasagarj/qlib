# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import cast

import pandas as pd

from qlib.backtest import Exchange, Order

from .pickle_styled import IntradayBacktestData


class QlibIntradayBacktestData(IntradayBacktestData):
    """Backtest data for Qlib simulator"""

    def __init__(
        self,
        order: Order,
        exchange: Exchange,
        ticks_index: pd.DatetimeIndex,
        ticks_for_order: pd.DatetimeIndex,
    ) -> None:
        super(QlibIntradayBacktestData, self).__init__()
        self._order = order
        self._exchange = exchange
        self._start_time = ticks_for_order[0]
        self._end_time = ticks_for_order[-1]
        self.ticks_index = ticks_index
        self.ticks_for_order = ticks_for_order

        self._deal_price = cast(
            pd.Series,
            self._exchange.get_deal_price(
                self._order.stock_id,
                self._start_time,
                self._end_time,
                direction=self._order.direction,
                method=None,
            ),
        )
        self._volume = cast(
            pd.Series,
            self._exchange.get_volume(
                self._order.stock_id,
                self._start_time,
                self._end_time,
                method=None,
            ),
        )

    def __repr__(self) -> str:
        return (
            f"Order: {self._order}, Exchange: {self._exchange}, "
            f"Start time: {self._start_time}, End time: {self._end_time}"
        )

    def __len__(self) -> int:
        return len(self._deal_price)

    def get_deal_price(self) -> pd.Series:
        return self._deal_price

    def get_volume(self) -> pd.Series:
        return self._volume

    def get_time_index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([e[1] for e in list(self._exchange.quote_df.index)])
