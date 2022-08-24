# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import yaml
from qlib.backtest import Order
from qlib.backtest.decision import OrderDir
from qlib.rl.amc4th_migration.time_index import intraday_timestamps
from qlib.rl.interpreter import ActionInterpreter, StateInterpreter
from qlib.rl.order_execution import SingleAssetOrderExecution
from qlib.rl.reward import Reward
from qlib.rl.trainer import Checkpoint, Trainer, TrainingVessel
from qlib.utils import init_instance_by_config
from tianshou.policy import BasePolicy
from torch import nn
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def _read_orders(order_dir: Path) -> pd.DataFrame:
    if os.path.isfile(order_dir):
        return pd.read_pickle(order_dir)
    else:
        orders = []
        for file in order_dir.iterdir():
            order_data = pd.read_pickle(file)
            orders.append(order_data)
        return pd.concat(orders)


def _get_orders(order_dir: Path, default_start_time: pd.Timedelta, default_end_time: pd.Timedelta) -> List[Order]:
    order_df = _read_orders(order_dir).reset_index()
    order_df = order_df[order_df["instrument"] == "SH603360"]  # TODO: small dataset
    order_df = order_df[order_df["date"] == "2017-11-17"]  # TODO: small dataset
    order_df = order_df[order_df["order_type"] == 0]  # TODO: small dataset

    order_list = []
    for index, row in tqdm(order_df.iterrows(), total=len(order_df), desc=str(path)):
        order = Order(
            stock_id=row["instrument"],
            amount=row["amount"],
            direction=OrderDir(int(row["order_type"])),
            start_time=pd.Timestamp(row["date"]) + default_start_time,
            end_time=pd.Timestamp(row["date"]) + default_end_time
        )
        order_list.append(order)

    return order_list


def train_and_test(
    env_config: dict,
    simulator_config: dict,
    trainer_config: dict,
    data_config: dict,
    state_interpreter: StateInterpreter,
    action_interpreter: ActionInterpreter,
    policy: BasePolicy,
    reward: Reward,
) -> None:
    default_start_time = intraday_timestamps[data_config["source"]["default_start_time"]]
    default_end_time = intraday_timestamps[data_config["source"]["default_end_time"]]
    order_root_path = Path(data_config["source"]["order_dir"])

    def _simulator_factory_simple(order: Order) -> SingleAssetOrderExecution:
        return SingleAssetOrderExecution(
            order=order,
            data_dir=Path(data_config["source"]["data_dir"]),
            ticks_per_step=simulator_config["time_per_step"],
            deal_price_type=data_config["source"]["deal_price_column"],
            vol_threshold=simulator_config["vol_limit"],
        )

    trainer = Trainer(
        max_iters=trainer_config["max_epoch"],
        finite_env_type=env_config["parallel_mode"],
        concurrency=env_config["concurrency"],
        # callbacks=[
        #     Checkpoint(dirpath=Path("C:/Users/huoranli/Downloads/tmp/"), every_n_iters=1),
        # ],
    )

    vessel = TrainingVessel(
        simulator_fn=_simulator_factory_simple,
        state_interpreter=state_interpreter,
        action_interpreter=action_interpreter,
        policy=policy,
        reward=reward,
        train_initial_states=_get_orders(order_root_path / "train", default_start_time, default_end_time),
        val_initial_states=_get_orders(order_root_path / "valid", default_start_time, default_end_time),
        test_initial_states=_get_orders(order_root_path / "test", default_start_time, default_end_time),
        episode_per_iter=trainer_config["episode_per_collect"],
        update_kwargs={
            "batch_size": trainer_config["batch_size"],
            "repeat": trainer_config["repeat_per_collect"],
        },
    )

    trainer.fit(vessel)


def main(config: dict) -> None:
    if "seed" in config["runtime"]:
        seed_everything(config["runtime"]["seed"])

    state_config = config["state_interpreter"]
    state_config["kwargs"]["data_dir"] = Path(config["data"]["source"]["proc_data_dir"])
    state_interpreter: StateInterpreter = init_instance_by_config(state_config)

    action_interpreter: ActionInterpreter = init_instance_by_config(config["action_interpreter"])
    reward: Reward = init_instance_by_config(config["reward"])

    # Create torch network
    config["network"]["kwargs"].update({"obs_space": state_interpreter.observation_space})
    network: nn.Module = init_instance_by_config(config["network"])

    # Create policy
    config["policy"]["kwargs"].update(
        {
            "network": network,
            "obs_space": state_interpreter.observation_space,
            "action_space": action_interpreter.action_space,
        }
    )
    policy: BasePolicy = init_instance_by_config(config["policy"])

    # use_cuda = config["runtime"].get("use_cuda", False)
    # if use_cuda:
    #     policy.cuda()

    train_and_test(
        env_config=config["env"],
        simulator_config=config["simulator"],
        data_config=config["data"],
        trainer_config=config["trainer"],
        action_interpreter=action_interpreter,
        state_interpreter=state_interpreter,
        policy=policy,
        reward=reward,
    )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    path = sys.argv[1]
    config = yaml.safe_load(open(path, "r"))

    main(config)
