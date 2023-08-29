# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from tabnanny import process_tokens
from tracemalloc import start
from gym import spaces

from isaacgym import gymtorch
from isaacgym.torch_utils import to_torch
import torch
import numpy as np
import copy

from tasks.hand_base.base_task import BaseTask

# VecEnv Wrapper for RL training
class MultiVecTaskAllegro():
    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        self.task = task

        self.num_environments = task.num_envs
        self.num_states = task.num_states
        self.num_actions = task.num_actions

        # self.agent_index = self.task.agent_index
        self.num_agents = 2  # used for multi-agent environments


        self.clip_obs = clip_observations
        self.clip_actions = clip_actions
        self.rl_device = task.device

        print("RL device: ", task.device)

        # COMPATIBILITY
        self.obs_space = [spaces.Box(low=-np.Inf, high=np.Inf, shape=(450,)) for _ in range(self.num_agents)]
        self.share_observation_space = [spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_states,)) for _ in
                                        range(self.num_agents)]


        self.act_space = tuple([spaces.Box(low=np.ones(self.num_actions) * -clip_actions,
                                    high=np.ones(self.num_actions) * clip_actions) for _ in
                                range(self.num_agents)])

    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_number_of_agents(self):
        return self.num_agents

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.num_agents}
        return env_info

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations

# Python CPU/GPU Class
class MultiVecTaskPythonAllegro(MultiVecTaskAllegro):

    def get_state(self):
        return torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def step(self, actions):

        a_hand_actions = actions[0]
        for i in range(1, len(actions)):
            a_hand_actions = torch.hstack((a_hand_actions, actions[i]))
        actions = a_hand_actions

        actions_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        self.task.step(actions_tensor)

        hand_obs = []
        obs_buf = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        hand_obs.append(torch.cat([obs_buf[:, :150], obs_buf[:, 300:450], obs_buf[:, 600:750]], dim=1))
        hand_obs.append(torch.cat([obs_buf[:, 150:300], obs_buf[:, 450:600], obs_buf[:, 750:900]], dim=1))
        state_buf = torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs)

        rewards = self.task.rew_buf.unsqueeze(-1).to(self.rl_device)
        dones = self.task.reset_buf.to(self.rl_device)

        sub_agent_obs = []
        agent_state = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(2):
            sub_agent_obs.append(hand_obs[i])
            agent_state.append(state_buf)
            sub_agent_reward.append(rewards)
            sub_agent_done.append(dones)
            sub_agent_info.append(torch.Tensor(0))

        obs_all = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        state_all = torch.transpose(torch.stack(agent_state), 1, 0)
        reward_all = torch.transpose(torch.stack(sub_agent_reward), 1, 0)
        done_all = torch.transpose(torch.stack(sub_agent_done), 1, 0)
        info_all = torch.stack(sub_agent_info)

        return obs_all, state_all, reward_all, done_all, info_all, None

    def reset(self):
        actions = 0.01 * (1 - 2 * torch.rand([self.task.num_envs, self.num_actions * self.num_agents], dtype=torch.float32, device=self.rl_device))

        # step the simulator
        self.task.step(actions)

        hand_obs = []
        obs_buf = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        hand_obs.append(torch.cat([obs_buf[:, :150], obs_buf[:, 300:450], obs_buf[:, 600:750]], dim=1))
        hand_obs.append(torch.cat([obs_buf[:, 150:300], obs_buf[:, 450:600], obs_buf[:, 750:900]], dim=1))
        state_buf = torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs)

        sub_agent_obs = []
        agent_state = []

        for i in range(2):
            sub_agent_obs.append(hand_obs[i])
            agent_state.append(state_buf)

        obs = torch.transpose(torch.stack(sub_agent_obs), 1, 0)
        state_all = torch.transpose(torch.stack(agent_state), 1, 0)

        return obs, state_all, None

class SingleVecTaskPythonArm() :

    def __init__(self, task : BaseTask, rl_device, clip_observations=5.0, clip_actions=1.0) :

        self.task = task
        self.rl_device = rl_device
        self.num_agents = 1
        self.num_observations = task.num_obs
        self.nums_share_observations = task.num_obs
        self.num_environments = task.num_envs
        self.num_actions = task.num_actions
        self.obs_space = [spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.num_observations,)) for _ in range(self.num_agents)]
        self.share_observation_space = [spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.nums_share_observations,)) for _ in range(self.num_agents)]
        self.act_space = [spaces.Box(low=task.franka1_dof_lower_limits, high=task.franka1_dof_lower_limits) for _ in range(self.num_agents)]
    
    def step(self, actions):
        obs, reward, done, info = self.task.step(actions[0])
        obs = obs.unsqueeze(dim=0)
        reward = reward.unsqueeze(dim=0)
        done = done.unsqueeze(dim=0)
        return obs, obs, reward, done, info, None

    def reset(self):

        obs, reward, done, info = self.task.reset()
        obs = obs.unsqueeze(dim=0)
        return obs, obs, None

    def get_number_of_agents(self):
        return self.num_agents

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.num_agents}
        return env_info

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations