from typing import List, Optional

import pandas as pd

from qlib.backtest import Account, CommonInfrastructure, get_exchange
from qlib.rl.from_neutrader.config import ExchangeConfig


def get_common_infra(
    config: ExchangeConfig,
    trade_start_date: pd.Timestamp,
    trade_end_date: pd.Timestamp,
    codes: List[str],
    cash_limit: Optional[float] = None,
) -> CommonInfrastructure:
    # need to specify a range here for acceleration
    if cash_limit is None:
        trade_account = Account(init_cash=int(1e12), benchmark_config={}, pos_type="InfPosition")
    else:
        trade_account = Account(
            init_cash=cash_limit,
            benchmark_config={},
            pos_type="Position",
            position_dict={code: {"amount": 1e12, "price": 1.0} for code in codes},
        )

    exchange = get_exchange(
        codes=codes,
        freq="1min",
        limit_threshold=config.limit_threshold,
        deal_price=config.deal_price,
        open_cost=config.open_cost,
        close_cost=config.close_cost,
        min_cost=config.min_cost if config.trade_unit is not None else 0,
        start_time=trade_start_date,
        end_time=trade_end_date + pd.DateOffset(1),
        trade_unit=config.trade_unit,
        volume_threshold=config.volume_threshold,
    )

    return CommonInfrastructure(trade_account=trade_account, trade_exchange=exchange)
