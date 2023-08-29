# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import sys
sys.path.insert(0, sys.path[0]+"/../")

from ast import arg
from matplotlib.pyplot import get
import numpy as np
import random

from utils.config import set_np_formatting, set_seed, get_args_scripts, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_sarl import *
from utils.process_marl import process_MultiAgentRL, get_AgentIndex
from utils.process_mtrl import *
from utils.process_metarl import *
import os

from rl_games.common import env_configurations, experiment, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner

import yaml
import argparse

# from utils.rl_games_custom import 
from rl_games.common.algo_observer import IsaacAlgoObserver
from policy_sequencing.transition_value_trainer import TValue_Trainer

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main_rlgames(task, num_envs, use_t_value=False, policy_path=""):
    set_np_formatting()
    args = get_args_scripts(task, num_envs, 'lego', use_rlg_config=True)
    args.use_t_value = use_t_value
    args.checkpoint = policy_path

    if args.checkpoint == "Base":
        args.checkpoint = ""

    if args.algo == "ppo":
        config_name = "cfg/{}/ppo_continuous.yaml".format(args.algo)
    elif args.algo == "lego":
        config_name = "cfg/{}/ppo_continuous.yaml".format(args.algo)
        if args.task in ["BlockAssemblyGrasp", "BlockAssemblyGraspSim", "BlockAssemblyOrient", "BlockAssemblyOrientOnce",
                         "ToolPositioningChainPureRL", "ToolPositioningChainTSTAR", "ToolPositioningChain", "ToolPositioningOrient", "ToolPositioningGrasp"]:
            config_name = "cfg/{}/ppo_continuous_grasp.yaml".format(args.algo)
        if args.task in ["BlockAssemblyInsert", "BlockAssemblyInsertSim"]:
            config_name = "cfg/{}/ppo_continuous_insert.yaml".format(args.algo)

    elif args.algo == "ppo_lstm":
        config_name = "cfg/{}/ppo_continuous_lstm.yaml".format(args.algo)
    else:
        print("We don't support this config in RL-games now")

    args.task_type = "RLgames"
    print('Loading config: ', config_name)

    args.cfg_train = config_name
    cfg, cfg_train, logdir = load_cfg(args, use_rlg_config=True)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    if args.seed is not None:
        cfg_train["seed"] = args.seed
    else:
        cfg_train["seed"] = 22

    cfg["env"]["numEnvs"] = args.num_envs
    cfg["env"]["test"] = args.play

    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))

    agent_index = get_AgentIndex(cfg)
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

    # override
    with open(config_name, 'r') as stream:
        rlgames_cfg = yaml.safe_load(stream)
        rlgames_cfg['params']['config']['name'] = args.task
        rlgames_cfg['params']['config']['num_actors'] = env.num_environments
        rlgames_cfg['params']['seed'] = cfg_train["seed"]
        rlgames_cfg['params']['config']['env_config']['seed'] = cfg_train["seed"]
        rlgames_cfg['params']['config']['vec_env'] = env
        rlgames_cfg['params']['config']['env_info'] = env.get_env_info()

    vargs = vars(args)
    algo_observer = IsaacAlgoObserver()
    
    runner = Runner(algo_observer)
    # runner = Runner()
    runner.load(rlgames_cfg)
    runner.reset()
    runner.run(vargs)

    if not args.use_t_value:
        return os.path.join(runner.nn_dir, "{}.pth".format(args.task))

def transition_value_trainer(task, rollout):
    trainer = TValue_Trainer("./intermediate_state/{}_datasets.hdf5".format(task))
    trainer.init_TValue_function(task, rollout)
    trainer.train_rollout()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, default="BlockAssembly")
    args = parser.parse_args()
    if args.tasks == "BlockAssembly":
        for i in range(10):
            # forward initialization
            search_policy_path = main_rlgames("BlockAssemblySearch", 128)
            orient_policy_path = main_rlgames("BlockAssemblyOrient", 512)
            grasp_sim_policy_path = main_rlgames("BlockAssemblyGraspSim", 512)
            insert_sim_policy_path = main_rlgames("BlockAssemblyInsertSim", 512)

            # backward finetuning
            main_rlgames("BlockAssemblyInsertSim", 512, use_t_value=True, policy_path=insert_sim_policy_path)
            transition_value_trainer("BlockAssemblyInsertSim", rollout=10000)
            main_rlgames("BlockAssemblyGraspSim", 512, use_t_value=True, policy_path=grasp_sim_policy_path)
            transition_value_trainer("BlockAssemblyGraspSim", rollout=10000)
            main_rlgames("BlockAssemblyOrient", 128, use_t_value=True, policy_path=orient_policy_path)
            transition_value_trainer("BlockAssemblyOrient", rollout=10000)
    
    elif args.tasks == "ToolPositioning":
        for i in range(10):
            # forward initialization
            grasp_policy_path = main_rlgames("ToolPositioningGrasp", 512)
            orient_policy_path = main_rlgames("ToolPositioningOrient", 512)

            # backward finetuning
            main_rlgames("ToolPositioningOrient", 512, use_t_value=True, policy_path=orient_policy_path)
            transition_value_trainer("ToolPositioningOrient", rollout=10000)

    else:
        raise Exception(
                "Unrecognized task!")





