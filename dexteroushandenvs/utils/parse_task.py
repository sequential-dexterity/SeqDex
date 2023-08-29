# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from tasks.block_assembly.allegro_hand_block_assembly_grasp import BlockAssemblyGrasp
from tasks.block_assembly.allegro_hand_block_assembly_grasp_sim import BlockAssemblyGraspSim
from tasks.block_assembly.allegro_hand_block_assembly_gui import BlockAssemblyGUI
from tasks.block_assembly.allegro_hand_block_assembly_insert_sim import BlockAssemblyInsertSim
from tasks.block_assembly.allegro_hand_block_assembly_insert import BlockAssemblyInsert
from tasks.block_assembly.allegro_hand_block_assembly_orient_once import BlockAssemblyOrientOnce
from tasks.block_assembly.allegro_hand_block_assembly_orient import BlockAssemblyOrient
from tasks.block_assembly.allegro_hand_block_assembly_search import BlockAssemblySearch

from tasks.tool_positioning.allegro_hand_tool_positioning_chain import ToolPositioningChain
from tasks.tool_positioning.allegro_hand_tool_positioning_chain_pure_rl import ToolPositioningChainPureRL
from tasks.tool_positioning.allegro_hand_tool_positioning_chain_tstar import ToolPositioningChainTSTAR
from tasks.tool_positioning.allegro_hand_tool_positioning_orient import ToolPositioningOrient
from tasks.tool_positioning.allegro_hand_tool_positioning_grasp import ToolPositioningGrasp

from tasks.hand_base.vec_task import VecTaskCPU, VecTaskGPU, VecTaskPython, VecTaskPythonArm
from tasks.hand_base.multi_vec_task import MultiVecTaskPython, SingleVecTaskPythonArm
from tasks.hand_base.multi_task_vec_task import MultiTaskVecTaskPython
from tasks.hand_base.meta_vec_task import MetaVecTaskPython
from tasks.hand_base.vec_task_rlgames import RLgamesVecTaskPython
from tasks.hand_base.imitation_vec_task import ImitationVecTaskPython
from tasks.hand_base.vec_task_lego import LegoVecTaskPython
from tasks.hand_base.multi_vec_task_allegro import MultiVecTaskPythonAllegro

from utils.config import warn_task_name

import json


def parse_task(args, cfg, cfg_train, sim_params, agent_index):

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.rl_device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    if args.task_type == "C++":
        if args.device == "cpu":
            print("C++ CPU")
            task = rlgpu.create_task_cpu(args.task, json.dumps(cfg_task))
            if not task:
                warn_task_name()
            if args.headless:
                task.init(device_id, -1, args.physics_engine, sim_params)
            else:
                task.init(device_id, device_id, args.physics_engine, sim_params)
            env = VecTaskCPU(task, rl_device, False, cfg_train.get("clip_observations", 5.0), cfg_train.get("clip_actions", 1.0))
        else:
            print("C++ GPU")

            task = rlgpu.create_task_gpu(args.task, json.dumps(cfg_task))
            if not task:
                warn_task_name()
            if args.headless:
                task.init(device_id, -1, args.physics_engine, sim_params)
            else:
                task.init(device_id, device_id, args.physics_engine, sim_params)
            env = VecTaskGPU(task, rl_device, cfg_train.get("clip_observations", 5.0), cfg_train.get("clip_actions", 1.0))

    elif args.task_type == "Python":
        print("Task type: Python")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                is_multi_agent=False)
        except NameError as e:
            print(e)
            warn_task_name()
        if args.task == "OneFrankaCabinet" :
            env = VecTaskPythonArm(task, rl_device)
        else :
            env = VecTaskPython(task, rl_device)

    elif args.task_type == "MultiAgent":
        print("Task type: MultiAgent")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                agent_index=agent_index,
                is_multi_agent=True)
        except NameError as e:
            print(e)
            warn_task_name()
        env = MultiVecTaskPythonAllegro(task, rl_device)

    elif args.task_type == "MultiTask":
        print("Task type: MultiTask")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                agent_index=agent_index,
                is_multi_agent=False)
        except NameError as e:
            print(e)
            warn_task_name()
        env = MultiTaskVecTaskPython(task, rl_device)

    elif args.task_type == "Meta":
        print("Task type: Meta")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                agent_index=agent_index,
                is_multi_agent=False)
        except NameError as e:
            print(e)
            warn_task_name()
        env = MetaVecTaskPython(task, rl_device)

    elif args.task_type == "Imitation":
        print("Task type: Imitation")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                agent_index=agent_index,
                is_multi_agent=False)
        except NameError as e:
            print(e)
            warn_task_name()
        env = ImitationVecTaskPython(task, rl_device)

    elif args.task_type == "RLgames":
        print("Task type: RLgames")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                agent_index=agent_index,
                is_multi_agent=False)
        except NameError as e:
            print(e)
            warn_task_name()
        env = RLgamesVecTaskPython(task, rl_device)

    elif args.task_type == "Lego":
        print("Task type: Lego")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                agent_index=agent_index,
                is_multi_agent=False)
        except NameError as e:
            print(e)
            warn_task_name()
        # env = HierarchicalLegoVecTaskPython(task, rl_device)
        env = LegoVecTaskPython(task, rl_device)

    elif args.task_type == "LegoBC":
        print("Task type: Lego")

        try:
            task = eval(args.task)(
                cfg=cfg,
                sim_params=sim_params,
                physics_engine=args.physics_engine,
                device_type=args.device,
                device_id=device_id,
                headless=args.headless,
                agent_index=agent_index,
                is_multi_agent=False)
        except NameError as e:
            print(e)
            warn_task_name()
        # env = HierarchicalLegoVecTaskPython(task, rl_device)
        env = VecTaskPython(task, rl_device)

    return task, env
