import os
from pathlib import Path

from qlib.rl.from_neutrader.config import ExchangeConfig, QlibBacktestConfig, QlibConfig, RuntimeConfig


# strategy_dict = {
#     "5min": TWAPStrategyConfig(),
#     "30min": MultiplexStrategyOnTradeStepConfig(
#         strategies=[
#             RLStrategyConfig(
#                 observation=FullHistoryObservationConfig(
#                     max_step=3, total_time=240, data_dim=16, cached_features=None
#                 ),
#                 action=CategoricalWOSplitConfig(values=3),
#                 policy=PPOConfig(
#                     lr=0.0001,
#                     weight_decay=0.0,
#                     discount_factor=1.0,
#                     max_grad_norm=100.0,
#                     reward_normalization=True,
#                     eps_clip=0.3,
#                     value_clip=1.0,
#                     vf_coef=1.0,
#                     gae_lambda=1.0,
#                     network=None,
#                     obs_space=None,
#                     action_space=None,
#                     weight_file=Path(os.path.join(DATA_PATH, "amc-checkpoints/opds_15_225/opds_15_30")),
#                 ),
#                 network=DualAttentionRNNConfig(
#                     input_dims={"data_processed": 16},
#                     hidden_dim=64,
#                     output_dim=32,
#                     rnn_type="<RNNType.GRU: 'gru'>",
#                     rnn_num_layers=1,
#                 ),
#             ),
#             RLStrategyConfig(
#                 observation=FullHistoryObservationConfig(
#                     max_step=6, total_time=240, data_dim=16, cached_features=None
#                 ),
#                 action=CategoricalWOSplitConfig(values=6),
#                 policy=PPOConfig(
#                     lr=0.0001,
#                     weight_decay=0.0,
#                     discount_factor=1.0,
#                     max_grad_norm=100.0,
#                     reward_normalization=True,
#                     eps_clip=0.3,
#                     value_clip=1.0,
#                     vf_coef=1.0,
#                     gae_lambda=1.0,
#                     network=None,
#                     obs_space=None,
#                     action_space=None,
#                     weight_file=Path(os.path.join(DATA_PATH, "amc-checkpoints/opds_15_225/opds_30_60")),
#                 ),
#                 network=DualAttentionRNNConfig(
#                     input_dims={"data_processed": 16},
#                     hidden_dim=64,
#                     output_dim=32,
#                     rnn_type="<RNNType.GRU: 'gru'>",
#                     rnn_num_layers=1,
#                 ),
#             ),
#             RLStrategyConfig(
#                 observation=FullHistoryObservationConfig(
#                     max_step=6, total_time=240, data_dim=16, cached_features=None
#                 ),
#                 action=CategoricalWOSplitConfig(values=6),
#                 policy=PPOConfig(
#                     lr=0.0001,
#                     weight_decay=0.0,
#                     discount_factor=1.0,
#                     max_grad_norm=100.0,
#                     reward_normalization=True,
#                     eps_clip=0.3,
#                     value_clip=1.0,
#                     vf_coef=1.0,
#                     gae_lambda=1.0,
#                     network=None,
#                     obs_space=None,
#                     action_space=None,
#                     weight_file=Path(os.path.join(DATA_PATH, "amc-checkpoints/opds_15_225/opds_60_90")),
#                 ),
#                 network=DualAttentionRNNConfig(
#                     input_dims={"data_processed": 16},
#                     hidden_dim=64,
#                     output_dim=32,
#                     rnn_type="<RNNType.GRU: 'gru'>",
#                     rnn_num_layers=1,
#                 ),
#             ),
#             RLStrategyConfig(
#                 observation=FullHistoryObservationConfig(
#                     max_step=6, total_time=240, data_dim=16, cached_features=None
#                 ),
#                 action=CategoricalWOSplitConfig(values=6),
#                 policy=PPOConfig(
#                     lr=0.0001,
#                     weight_decay=0.0,
#                     discount_factor=1.0,
#                     max_grad_norm=100.0,
#                     reward_normalization=True,
#                     eps_clip=0.3,
#                     value_clip=1.0,
#                     vf_coef=1.0,
#                     gae_lambda=1.0,
#                     network=None,
#                     obs_space=None,
#                     action_space=None,
#                     weight_file=Path(os.path.join(DATA_PATH, "amc-checkpoints/opds_15_225/opds_90_120")),
#                 ),
#                 network=DualAttentionRNNConfig(
#                     input_dims={"data_processed": 16},
#                     hidden_dim=64,
#                     output_dim=32,
#                     rnn_type="<RNNType.GRU: 'gru'>",
#                     rnn_num_layers=1,
#                 ),
#             ),
#             RLStrategyConfig(
#                 observation=FullHistoryObservationConfig(
#                     max_step=6, total_time=240, data_dim=16, cached_features=None
#                 ),
#                 action=CategoricalWOSplitConfig(values=6),
#                 policy=PPOConfig(
#                     lr=0.0001,
#                     weight_decay=0.0,
#                     discount_factor=1.0,
#                     max_grad_norm=100.0,
#                     reward_normalization=True,
#                     eps_clip=0.3,
#                     value_clip=1.0,
#                     vf_coef=1.0,
#                     gae_lambda=1.0,
#                     network=None,
#                     obs_space=None,
#                     action_space=None,
#                     weight_file=Path(os.path.join(DATA_PATH, "amc-checkpoints/opds_15_225/opds_120_150")),
#                 ),
#                 network=DualAttentionRNNConfig(
#                     input_dims={"data_processed": 16},
#                     hidden_dim=64,
#                     output_dim=32,
#                     rnn_type="<RNNType.GRU: 'gru'>",
#                     rnn_num_layers=1,
#                 ),
#             ),
#             RLStrategyConfig(
#                 observation=FullHistoryObservationConfig(
#                     max_step=6, total_time=240, data_dim=16, cached_features=None
#                 ),
#                 action=CategoricalWOSplitConfig(values=6),
#                 policy=PPOConfig(
#                     lr=0.0001,
#                     weight_decay=0.0,
#                     discount_factor=1.0,
#                     max_grad_norm=100.0,
#                     reward_normalization=True,
#                     eps_clip=0.3,
#                     value_clip=1.0,
#                     vf_coef=1.0,
#                     gae_lambda=1.0,
#                     network=None,
#                     obs_space=None,
#                     action_space=None,
#                     weight_file=Path(os.path.join(DATA_PATH, "amc-checkpoints/opds_15_225/opds_150_180")),
#                 ),
#                 network=DualAttentionRNNConfig(
#                     input_dims={"data_processed": 16},
#                     hidden_dim=64,
#                     output_dim=32,
#                     rnn_type="<RNNType.GRU: 'gru'>",
#                     rnn_num_layers=1,
#                 ),
#             ),
#             RLStrategyConfig(
#                 observation=FullHistoryObservationConfig(
#                     max_step=6, total_time=240, data_dim=16, cached_features=None
#                 ),
#                 action=CategoricalWOSplitConfig(values=6),
#                 policy=PPOConfig(
#                     lr=0.0001,
#                     weight_decay=0.0,
#                     discount_factor=1.0,
#                     max_grad_norm=100.0,
#                     reward_normalization=True,
#                     eps_clip=0.3,
#                     value_clip=1.0,
#                     vf_coef=1.0,
#                     gae_lambda=1.0,
#                     network=None,
#                     obs_space=None,
#                     action_space=None,
#                     weight_file=Path(os.path.join(DATA_PATH, "amc-checkpoints/opds_15_225/opds_180_210")),
#                 ),
#                 network=DualAttentionRNNConfig(
#                     input_dims={"data_processed": 16},
#                     hidden_dim=64,
#                     output_dim=32,
#                     rnn_type="<RNNType.GRU: 'gru'>",
#                     rnn_num_layers=1,
#                 ),
#             ),
#             RLStrategyConfig(
#                 observation=FullHistoryObservationConfig(
#                     max_step=3, total_time=240, data_dim=16, cached_features=None
#                 ),
#                 action=CategoricalWOSplitConfig(values=3),
#                 policy=PPOConfig(
#                     lr=0.0001,
#                     weight_decay=0.0,
#                     discount_factor=1.0,
#                     max_grad_norm=100.0,
#                     reward_normalization=True,
#                     eps_clip=0.3,
#                     value_clip=1.0,
#                     vf_coef=1.0,
#                     gae_lambda=1.0,
#                     network=None,
#                     obs_space=None,
#                     action_space=None,
#                     weight_file=Path(os.path.join(DATA_PATH, "amc-checkpoints/opds_15_225/opds_210_225")),
#                 ),
#                 network=DualAttentionRNNConfig(
#                     input_dims={"data_processed": 16},
#                     hidden_dim=64,
#                     output_dim=32,
#                     rnn_type="<RNNType.GRU: 'gru'>",
#                     rnn_num_layers=1,
#                 ),
#             ),
#         ]
#     ),
#     "day": RLStrategyConfig(
#         observation=FullHistoryObservationConfig(max_step=8, total_time=240, data_dim=16, cached_features=None),
#         action=CategoricalWOSplitConfig(values=4),
#         policy=PPOConfig(
#             lr=0.0001,
#             weight_decay=0.0,
#             discount_factor=1.0,
#             max_grad_norm=100.0,
#             reward_normalization=True,
#             eps_clip=0.3,
#             value_clip=1.0,
#             vf_coef=1.0,
#             gae_lambda=1.0,
#             network=None,
#             obs_space=None,
#             action_space=None,
#             weight_file=Path(os.path.join(DATA_PATH, "amc-checkpoints/opds_15_225/opds_15_225_30r_4_80")),
#         ),
#         network=DualAttentionRNNConfig(
#             input_dims={"data_processed": 16},
#             hidden_dim=64,
#             output_dim=32,
#             rnn_type="<RNNType.GRU: 'gru'>",
#             rnn_num_layers=1,
#         ),
#     ),
# }

DATA_PATH = "C:/workspace/NeuTrader/data/"  # TODO: test only

# fmt: off
qlib_backtest_config = QlibBacktestConfig(
    order_file=Path(os.path.join(DATA_PATH, "amc-real-order/orders_v4/csi300_nostop.pkl")),
    start_time="9:45",
    end_time="14:44",
    qlib=QlibConfig(
        provider_uri_day=Path(os.path.join(DATA_PATH, "amc-qlib/huaxia_1d_qlib")),
        provider_uri_1min=Path(os.path.join(DATA_PATH, "amc-qlib/huaxia_1min_qlib")),
        feature_root_dir=Path(os.path.join(DATA_PATH, "amc-qlib-stock")),
        feature_columns_today=[
            "$open", "$high", "$low", "$close", "$vwap", "$bid", "$ask", "$volume", "$bidV", "$bidV1",
            "$bidV3", "$bidV5", "$askV", "$askV1", "$askV3", "$askV5",
        ],
        feature_columns_yesterday=[
            "$open_1", "$high_1", "$low_1", "$close_1", "$vwap_1", "$bid_1", "$ask_1", "$volume_1", "$bidV_1",
            "$bidV1_1", "$bidV3_1", "$bidV5_1", "$askV_1", "$askV1_1", "$askV3_1", "$askV5_1",
        ],
    ),
    exchange=ExchangeConfig(
        limit_threshold=("$ask == 0", "$bid == 0"),
        deal_price=("If($ask == 0, $bid, $ask)", "If($bid == 0, $ask, $bid)"),
        volume_threshold={
            "all": ("cum", "0.2 * DayCumsum($volume, '9:45', '14:44')"),
            "buy": ("current", "$askV1"),
            "sell": ("current", "$bidV1"),
        },
        open_cost=0.0005,
        close_cost=0.0015,
        min_cost=5.0,
        trade_unit=100.0,
        cash_limit=None,
        generate_report=False,
    ),
    strategies={},  # TODO: to be added
    debug_single_stock=None,
    debug_single_day=None,
    concurrency=-1,
    multiplier=1.0,
    runtime=RuntimeConfig(seed=42, output_dir=None, checkpoint_dir=None, tb_log_dir=None, debug=False, use_cuda=True),
)
# fmt: on
