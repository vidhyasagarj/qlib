import os
import sys
from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path

import pandas as pd

from qlib.backtest.decision import TradeRangeByTime
from qlib.contrib.strategy.rule_strategy import FileOrderStrategy
from qlib.rl.amc4th_migration.utils import read_order_file
from qlib.rl.from_neutrader.config import QlibBacktestConfig
from qlib.rl.from_neutrader.feature import init_qlib
from qlib.rl.from_neutrader.infra import get_common_infra


def single(
    config: QlibBacktestConfig,
    stock: str,
    orders: pd.DataFrame,
    cash_limit: float = None,
) -> None:
    init_qlib(config.qlib, stock)

    trade_start_time = orders['datetime'].min()
    trade_end_time = orders['datetime'].max()
    stocks = orders.instrument.unique().tolist()

    common_infra = get_common_infra(
        config.exchange,
        pd.Timestamp(trade_start_time),
        pd.Timestamp(trade_end_time),
        stocks,
        cash_limit,
    )

    strategy = FileOrderStrategy(
        orders,
        common_infra=common_infra,
        trade_range=TradeRangeByTime(config.start_time, config.end_time),
    )


def main(config: QlibBacktestConfig):
    order_df = read_order_file(config.order_file)

    stock_pool = order_df['instrument'].unique().tolist()
    stock_pool.sort()
    for stock in stock_pool:
        if not stock.startswith("SH600000"):
            continue

        single(
            config=config,
            stock=stock,
            orders=order_df[order_df['instrument'] == stock].copy(),
        )


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument('exp', help='Experiment YAML file')
    args, _ = parser.parse_known_args()

    config_path = Path(args.exp)
    module_name = os.path.splitext(os.path.basename(args.exp))[0]
    sys.path.insert(0, str(config_path.parent))
    module = import_module(module_name)
    sys.path.pop(0)

    qlib_backtest_config: QlibBacktestConfig = module.__dict__["qlib_backtest_config"]

    main(qlib_backtest_config)
