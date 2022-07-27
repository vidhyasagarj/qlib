# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from qlib.strategy.base import BaseStrategy


# TODO: In the future we should merge the dataclass-based config with Qlib's dict-based config.
@dataclass
class ExchangeConfig:
    limit_threshold: Union[float, Tuple[str, str]]
    deal_price: Union[str, Tuple[str, str]]
    volume_threshold: dict
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    min_cost: float = 5.0
    trade_unit: Optional[float] = 100.0
    cash_limit: Optional[Union[Path, float]] = None
    generate_report: bool = False


@dataclass
class QlibConfig:
    provider_uri_day: Path
    provider_uri_1min: Path
    feature_root_dir: Path
    feature_columns_today: List[str]
    feature_columns_yesterday: List[str]


@dataclass
class RuntimeConfig:
    seed: int = 42
    output_dir: Optional[Path] = None
    checkpoint_dir: Optional[Path] = None
    tb_log_dir: Optional[Path] = None
    debug: bool = False
    use_cuda: bool = True
    #
    # def check_path(self, path):
    #     return True


@dataclass
class QlibBacktestConfig:
    order_file: Path
    start_time: str  # e.g., "9:30"
    end_time: str
    qlib: QlibConfig
    exchange: ExchangeConfig
    strategies: Dict[str, BaseStrategy]
    debug_single_stock: Optional[str] = None
    debug_single_day: Optional[str] = None
    concurrency: int = -1
    multiplier: float = 1.
    runtime: RuntimeConfig = RuntimeConfig()
