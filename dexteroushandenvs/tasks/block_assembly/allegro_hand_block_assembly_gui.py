# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch
import random

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from tasks.hand_base.base_task import BaseTask

from tasks.hand_base.vec_task import VecTask

from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
from PIL import Image as Im
import cv2

from einops import rearrange
import pickle
import time
import pyautogui

class BlockAssemblyGUI(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):

        self.cfg = cfg

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.hand_reset_step = self.cfg["env"]["handResetStep"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)
        self.rotation_axis = "y"
        if self.rotation_axis == "x":
            self.rotation_id = 0
        elif self.rotation_axis == "y":
            self.rotation_id = 1
        else:
            self.rotation_id = 2

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.object_type = self.cfg["env"]["objectType"]
        self.spin_coef = self.cfg["env"].get("spin_coef", 1.0)
        assert self.object_type in ["block", "egg", "pen"]

        self.ignore_z = (self.object_type == "pen")

        self.robot_asset_files_dict = {
            "normal": "urdf/franka_description/robots/franka_panda_allegro.urdf",
        }
        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/box/mobility.urdf",
            "pen": "mjcf/open_ai_assets/hand/pen.xml"
        }

        # can be "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["full_no_vel", "full", "full_state", "full_contact", "partial_contact"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.palm_name = "palm"
        self.contact_sensor_names = ["link_1.0_fsr", "link_2.0_fsr", "link_3.0_tip_fsr",
                                     "link_5.0_fsr", "link_6.0_fsr", "link_7.0_tip_fsr", "link_9.0_fsr",
                                     "link_10.0_fsr", "link_11.0_tip_fsr", "link_14.0_fsr", "link_15.0_fsr",
                                     "link_15.0_tip_fsr"]
        self.fingertip_names = ["link_3.0",
                                "link_7.0",
                                "link_11.0",
                                "link_15.0"]
        # 11, 13, 16, 20, 22, 24, 27, 29, 32, 36, 39, 40
        # self.contact_sensor_names = ["link_1.0", "link_2.0", "link_3.0_tip",
        #                              "link_5.0", "link_6.0", "link_7.0_tip", "link_9.0",
        #                              "link_10.0", "link_11.0_tip", "link_14.0", "link_15.0",
        #                              "link_15.0_tip"]
        self.stack_obs = 3
        # self.num_obs_dict = {
        #     "full_no_vel": 50,
        #     "full": 72,
        #     "full_state": 88,
        #     "full_contact": 90,
        #     "partial_contact": 74 + 128*128*4
        # }

        self.num_obs_dict = {
            "partial_contact": 62
        }
        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 120 + 128*128*4
            num_states = 187

        self.one_frame_num_obs = self.num_obs_dict[self.obs_type]
        self.one_frame_num_states = num_states
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type] * self.stack_obs
        self.cfg["env"]["numStates"] = num_states * self.stack_obs
        self.cfg["env"]["numActions"] = 23

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.enable_camera_sensors = self.cfg["env"]["enable_camera_sensors"]

        super().__init__(cfg=self.cfg, enable_camera_sensors=self.enable_camera_sensors)
        pyautogui.press("tab")  # tab will remove the toolbar on the left side of the screen

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        if self.viewer != None:
            # front
            cam_pos = gymapi.Vec3(0.8, 0.0, 1.25)
            cam_target = gymapi.Vec3(-0.6, 0.0, 0.25)
            # fullfigure
            cam_pos = gymapi.Vec3(0.75, -0.35, 1.25)
            cam_target = gymapi.Vec3(-0.6, 0.35, 0.25)
            
            cam_pos = gymapi.Vec3(0.7, -0.4, 1.3)
            cam_target = gymapi.Vec3(-0.6, 0.4, 0.2)

            # right size spin view
            # cam_pos = gymapi.Vec3(0.25, 0.6, 1.1)
            # cam_target = gymapi.Vec3(0.25, -0.3, 0.5)

            # spin view
            # cam_pos = gymapi.Vec3(0.25, -0.075, 0.825)
            # cam_target = gymapi.Vec3(0.25, 0.45, 0.6)

            # insert view
            # cam_pos = gymapi.Vec3(0.25, -0.075 - 0.4, 0.825)
            # cam_target = gymapi.Vec3(0.25, 0.45 - 0.4, 0.6)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.jacobian_tensor = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "hand"))

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.arm_hand_default_dof_pos = torch.zeros(self.num_arm_hand_dofs, dtype=torch.float, device=self.device)
        self.arm_hand_default_dof_pos[:7] = torch.tensor([0.9467, -0.5708, -2.4997, -2.3102, -0.7739,  2.6366, 2.2207], dtype=torch.float, device=self.device)        

        self.arm_hand_default_dof_pos[7:] = to_torch([0.0, -0.174, 0.785, 0.785,
                                            0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785], dtype=torch.float, device=self.device)

        self.arm_hand_prepare_dof_poses = torch.zeros((self.num_envs, self.num_arm_hand_dofs), dtype=torch.float, device=self.device)
        self.end_effector_rotation = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)

        self.arm_hand_prepare_dof_pos_list = []
        self.end_effector_rot_list = []
        self.arm_hand_insertion_prepare_dof_pos_list = []

        # rot = [0, 0.707, 0, 0.707]
        self.arm_hand_prepare_dof_pos = to_torch([0.0, -0.49826458111314524, -0.01990020486871322, -2.4732269941140346, -0.01307073642274261, 2.00396583422025, 1.5480939705504309,
                                                0.0, -0.174, 0.785, 0.785,
                                            0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785], dtype=torch.float, device=self.device)
        
        self.arm_hand_prepare_dof_pos[7:] = scale(torch.zeros(16, dtype=torch.float, device=self.device), 
                                                self.arm_hand_dof_lower_limits[7:23], self.arm_hand_dof_upper_limits[7:23])
        
        self.arm_hand_prepare_dof_pos_list.append(self.arm_hand_prepare_dof_pos)
        self.end_effector_rot_list.append(to_torch([0, 0.707, 0, 0.707], device=self.device))

        self.arm_hand_insertion_prepare_dof_pos = to_torch([-0.1560, -0.2140, -0.2795, -2.1806, -0.0681,  1.9730,  1.1735], dtype=torch.float, device=self.device)
        self.arm_hand_insertion_prepare_dof_pos_list.append(self.arm_hand_insertion_prepare_dof_pos)
        # self.arm_hand_insertion_prepare_dof_pos = to_torch([-1.0, -0.4954,  0.4536, -2.4975,  0.2445,  2.0486, 1.1839], dtype=torch.float, device=self.device)
        self.arm_hand_insertion_prepare_dof_pos = to_torch([-0.1800, -0.1604, -0.2770, -2.2674, -0.0533,  2.1049,  1.1696], dtype=torch.float, device=self.device)
        self.arm_hand_insertion_prepare_dof_pos_list.append(self.arm_hand_insertion_prepare_dof_pos)

        # face forward
        self.arm_hand_prepare_dof_pos = to_torch([-1.4528e-02,  2.3290e-01,  1.5519e-02, -2.7374e+00,  8.7328e-04, 4.5402e+00,  3.1363e+00,
                                                0.0, -0.174, 0.785, 0.785,
                                            0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785], dtype=torch.float, device=self.device)
        
        self.arm_hand_prepare_dof_pos_list.append(self.arm_hand_prepare_dof_pos)
        self.end_effector_rot_list.append(to_torch([1, 0., 0., 0.], device=self.device))

        # face right, [-0.4227, -0.6155, -0.3687, -0.5537]  0.0276,  0.0870, -0.4854, -2.6056,  1.2111,  1.3671, -1.1870
        # self.arm_hand_prepare_dof_pos = to_torch([1.0260,  0.0671, 0.42, -2.4576, -0.25,  3.7172,  1.82,
        #                                         0.0, -0.174, 0.785, 0.785,
        #                                     0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785], dtype=torch.float, device=self.device)
        self.arm_hand_prepare_dof_pos = to_torch([0.1707,  0.0737, -0.5725, -2.4737,  1.2567,  1.3162, -1.0150,
                                                0.0, -0.174, 0.785, 0.785,
                                            0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785], dtype=torch.float, device=self.device)
        self.arm_hand_prepare_dof_pos_list.append(self.arm_hand_prepare_dof_pos)
        self.end_effector_rot_list.append(to_torch([0.5, 0.5, 0.5, 0.5], device=self.device))

        # face left, [ 0.4175, -0.5494,  0.4410, -0.5739] -1.5712, -1.5254,  1.7900, -2.2848,  3.1094,  3.7490, -2.8722
        # self.arm_hand_prepare_dof_pos = to_torch([1.0260,  0.0671, -2.72, -2.4576, -0.25,  3.7172,  -1.32,
        #                                         0.0, -0.174, 0.785, 0.785,
        #                                     0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785], dtype=torch.float, device=self.device)
        self.arm_hand_prepare_dof_pos = to_torch([-0.4006, -0.1464,  0.7419, -2.3031, -1.2898,  1.3568, 2.2076,
                                                0.0, -0.174, 0.785, 0.785,
                                            0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785], dtype=torch.float, device=self.device)
        self.arm_hand_prepare_dof_pos_list.append(self.arm_hand_prepare_dof_pos)
        self.end_effector_rot_list.append(to_torch([-0.707, 0.707, 0.0, -0.0], device=self.device))

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_arm_hand_dofs]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.all_lego_brick_pos_tensors = []
        self.contact_tensor = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1)
        print("Contact Tensor Dimension", self.contact_tensor.shape)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0
        self.total_steps = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.object_pose_for_open_loop = torch.zeros_like(self.root_state_tensor[self.object_indices, 0:7])

        self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.hand_indices[0], "panda_link7", gymapi.DOMAIN_ENV)
        print("hand_base_rigid_body_index: ", self.hand_base_rigid_body_index)
        
        self.hand_pos_history = torch.zeros((self.num_envs, self.max_episode_length, 3), dtype=torch.float, device=self.device)
        self.segmentation_object_center_point_x = torch.zeros((self.num_envs, 1), dtype=torch.int, device=self.device)
        self.segmentation_object_center_point_y = torch.zeros((self.num_envs, 1), dtype=torch.int, device=self.device)
        self.segmentation_object_point_num = torch.zeros((self.num_envs, 1), dtype=torch.int, device=self.device)

        self.meta_obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.meta_states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.meta_rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.meta_reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.meta_progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)

        self.arm_hand_prepare_dof_poses[:, :] = self.arm_hand_prepare_dof_pos_list[3]
        self.end_effector_rotation[:, :] = self.end_effector_rot_list[3]

        self.saved_searching_ternimal_states = torch.zeros(
            (10000 + 1024, self.root_state_tensor.view(self.num_envs, -1, 13).shape[1], 13), device=self.device, dtype=torch.float)
        self.saved_searching_ternimal_states_index = 0

        self.state_buf_stack_frames = []
        self.obs_buf_stack_frames = []
        for i in range(self.stack_obs):
            self.obs_buf_stack_frames.append(torch.zeros_like(self.obs_buf[:, 0:self.one_frame_num_obs]))
            self.state_buf_stack_frames.append(torch.zeros_like(self.states_buf[:, 0:self.one_frame_num_states]))

        self.spin_stack_obs = 3
        # self.grasping_one_frame_num_obs = 74
        self.spin_one_frame_num_obs = 132
        self.spin_one_frame_num_states = 154

        self.spin_num_obs = self.spin_stack_obs * self.spin_one_frame_num_obs
        self.spin_obs_buf = torch.zeros(
            (self.num_envs, self.spin_num_obs), device=self.device, dtype=torch.float)
        self.spin_num_states = self.spin_stack_obs * self.spin_one_frame_num_states
        self.spin_states_buf = torch.zeros(
            (self.num_envs, self.spin_num_states), device=self.device, dtype=torch.float)

        self.spin_state_buf_stack_frames = []
        self.spin_obs_buf_stack_frames = []
        for i in range(self.spin_stack_obs):
            self.spin_obs_buf_stack_frames.append(torch.zeros_like(self.spin_obs_buf[:, 0:self.spin_one_frame_num_obs]))
            self.spin_state_buf_stack_frames.append(torch.zeros_like(self.spin_states_buf[:, 0:self.spin_one_frame_num_states]))

        self.grasping_stack_obs = 3
        # self.grasping_one_frame_num_obs = 74
        self.grasping_one_frame_num_obs = 132
        self.grasping_one_frame_num_states = 154

        self.grasping_num_obs = self.grasping_stack_obs * self.grasping_one_frame_num_obs
        self.grasping_obs_buf = torch.zeros(
            (self.num_envs, self.grasping_num_obs), device=self.device, dtype=torch.float)
        self.grasping_num_states = self.grasping_stack_obs * self.grasping_one_frame_num_states
        self.grasping_states_buf = torch.zeros(
            (self.num_envs, self.grasping_num_states), device=self.device, dtype=torch.float)
        
        self.grasping_state_buf_stack_frames = []
        self.grasping_obs_buf_stack_frames = []
        for i in range(self.grasping_stack_obs):
            self.grasping_obs_buf_stack_frames.append(torch.zeros_like(self.grasping_obs_buf[:, 0:self.grasping_one_frame_num_obs]))
            self.grasping_state_buf_stack_frames.append(torch.zeros_like(self.grasping_states_buf[:, 0:self.grasping_one_frame_num_states]))

        self.insertion_stack_obs = 1
        # self.insertion_one_frame_num_obs = 156
        self.insertion_one_frame_num_obs = 75
        self.insertion_one_frame_num_states = 168

        self.insertion_num_obs = self.insertion_stack_obs * self.insertion_one_frame_num_obs
        self.insertion_obs_buf = torch.zeros(
            (self.num_envs, self.insertion_num_obs), device=self.device, dtype=torch.float)
        self.insertion_num_states = self.insertion_stack_obs * self.insertion_one_frame_num_states
        self.insertion_states_buf = torch.zeros(
            (self.num_envs, self.insertion_num_states), device=self.device, dtype=torch.float)
        
        self.insertion_state_buf_stack_frames = []
        self.insertion_obs_buf_stack_frames = []
        for i in range(self.insertion_stack_obs):
            self.insertion_obs_buf_stack_frames.append(torch.zeros_like(self.insertion_obs_buf[:, 0:self.insertion_one_frame_num_obs]))
            self.insertion_state_buf_stack_frames.append(torch.zeros_like(self.insertion_states_buf[:, 0:self.insertion_one_frame_num_states]))

        # load policy
        import copy
        from utils.robot_controller.nn_builder import build_network
        from utils.robot_controller.nn_controller import NNController
        from rl_games.algos_torch import torch_ext

        self.search_policy = NNController(num_actors=1, config_path='./utils/robot_controller/grasp_network.yaml', obs_dim=self.num_obs)
        self.search_policy.load(os.getcwd()+'/checkpoint/block_assembly/last_AllegroHandLegoTestSpin_ep_17000_rew_559.8565.pth')

        self.spin_policy = NNController(num_actors=1, config_path='./utils/robot_controller/grasp_network.yaml', obs_dim=self.spin_num_obs)
        self.spin_policy.load(os.getcwd()+'/checkpoint/block_assembly/last_AllegroHandLegoTestPAISim_ep_19000_rew_1530.9819.pth')

        self.grasp_policy = NNController(num_actors=1, config_path='./utils/robot_controller/grasp_network.yaml', obs_dim=self.grasping_num_obs)
        self.grasp_policy.load(os.getcwd()+'/checkpoint/block_assembly/last_AllegroHandLegoTestPAISim_ep_19000_rew_1530.9819.pth')

        self.insert_policy = NNController(num_actors=1, config_path='./utils/robot_controller/insert_network.yaml', obs_dim=self.insertion_num_obs)
        self.insert_policy.load(os.getcwd()+'/checkpoint/block_assembly/last_AllegroHandLegoTestPAInsertSimep206140rew[85.62].pth')

        self.target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.target_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)

        self.last_target_pos = to_torch([0.25, -0.2, 0.618], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
        quat = gymapi.Quat().from_euler_zyx(0, 0, 0)
        self.last_target_rot = to_torch([quat.x, quat.y, quat.z, quat.w], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)

        self.force_reset = False

        self.record_rendering_config = False
        if self.record_rendering_config:
            self.record = {"configurations": [], "poses": []}

        self.success_buf = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        self.target_euler = to_torch([0.0, 3.1415, 1.571], device=self.device).repeat((self.num_envs, 1))

        self.segmentation_target_init_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3].clone()
        self.segmentation_target_init_rot = self.root_state_tensor[self.lego_segmentation_indices, 3:7].clone()
        self.segmentation_target_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3].clone()
        self.segmentation_target_rot = self.root_state_tensor[self.lego_segmentation_indices, 3:7].clone()

        self.record_camera_view_image = False
        if self.record_camera_view_image:
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.out = cv2.VideoWriter('./intermediate_state/tvalue_vis_dig.mp4', fourcc, 30.0, (6144, 2048))# Capture video from camera or file

            # cv2.namedWindow("DEBUG_RGB_VIS", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("DEBUG_RGB_VIS", 512, 512)
            # cv2.namedWindow("DEBUG_HI_RGB_VIS", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("DEBUG_HI_RGB_VIS", 512, 512)

        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_W, "W")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_S, "S")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_A, "A")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_D, "D")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Q, "Q")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_E, "E")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_T, "T")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_Y, "Y")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_G, "G")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_H, "H")

        self.record_inference_video = self.cfg["record_video"]
        if self.record_inference_video:
            self.record_lego_type = self.cfg["record_lego_type"]
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.video_save_path = "./output_video/"
            os.makedirs(self.video_save_path, exist_ok=True)
            self.out = cv2.VideoWriter(self.video_save_path + "{}-{}-{}_{}:{}:{}.mp4".format(time.localtime()[0], time.localtime()[1], time.localtime()[2], time.localtime()[3], time.localtime()[4], time.localtime()[5]), fourcc, 30.0, (2160, 1440))# Capture video from camera or file

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim_params.physx.max_gpu_contact_pairs = int(self.sim_params.physx.max_gpu_contact_pairs)
        # self.sim_params.dt = 1./120.

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')

        arm_hand_asset_file = self.robot_asset_files_dict["normal"]
        # arm_hand_asset_file = "urdf/xarm6/xarm6_allegro_left.urdf"
        #"urdf/xarm6/xarm6_allegro_fsr.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            # arm_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", arm_hand_asset_file)

        object_asset_file = self.asset_files_dict[self.object_type]

        # load arm and hand.
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        # asset_options.use_mesh_materials = True
        # asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        # asset_options.override_com = True
        # asset_options.override_inertia = True
        # asset_options.vhacd_enabled = True
        # asset_options.vhacd_params = gymapi.VhacdParams()
        # asset_options.vhacd_params.resolution = 200000
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        arm_hand_asset = self.gym.load_asset(self.sim, asset_root, arm_hand_asset_file, asset_options)
        self.num_arm_hand_bodies = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        self.num_arm_hand_shapes = self.gym.get_asset_rigid_shape_count(arm_hand_asset)
        self.num_arm_hand_dofs = self.gym.get_asset_dof_count(arm_hand_asset)
        print("Num dofs: ", self.num_arm_hand_dofs)
        print("num_arm_hand_shapes: ", self.num_arm_hand_shapes)
        print("num_arm_hand_bodies: ", self.num_arm_hand_bodies)
        self.num_arm_hand_actuators = self.num_arm_hand_dofs #self.gym.get_asset_actuator_count(shadow_hand_asset)

        # Set up each DOF.
        self.actuated_dof_indices = [i for i in range(7, self.num_arm_hand_dofs)]
        self.finger_dof_indices = [10, 18, 22]

        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        self.arm_hand_dof_default_pos = []
        self.arm_hand_dof_default_vel = []

        robot_lower_qpos = []
        robot_upper_qpos = []

        robot_dof_props = self.gym.get_asset_dof_properties(arm_hand_asset)

        for i in range(23):
            robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if i < 7:
                robot_dof_props['stiffness'][i] = 400
                robot_dof_props['effort'][i] = 200
                robot_dof_props['damping'][i] = 80

            else:
                robot_dof_props['velocity'][i] = 10.0
                robot_dof_props['effort'][i] = 0.7
                robot_dof_props['stiffness'][i] = 20
                robot_dof_props['damping'][i] = 1
            robot_lower_qpos.append(robot_dof_props['lower'][i])
            robot_upper_qpos.append(robot_dof_props['upper'][i])

        self.finger_dof_indices = to_torch(self.finger_dof_indices, dtype=torch.long, device=self.device)
        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.arm_hand_dof_lower_limits = to_torch(robot_lower_qpos, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(robot_upper_qpos, device=self.device)
        self.arm_hand_dof_lower_qvel = to_torch(-robot_dof_props["velocity"], device=self.device)
        self.arm_hand_dof_upper_qvel = to_torch(robot_dof_props["velocity"], device=self.device)

        for i in range(self.num_arm_hand_dofs):
            self.arm_hand_dof_default_vel.append(0.0)

        self.arm_hand_dof_default_pos = to_torch(self.arm_hand_dof_default_pos, device=self.device)
        self.arm_hand_dof_default_vel = to_torch(self.arm_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = True
        object_asset_options.fix_base_link = True
        # object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        # object_asset_options.override_com = True
        # object_asset_options.override_inertia = True
        # object_asset_options.vhacd_enabled = True
        # object_asset_options.vhacd_params = gymapi.VhacdParams()
        # object_asset_options.vhacd_params.resolution = 100000
        # object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        # Put objects in the scene.
        arm_hand_start_pose = gymapi.Transform()
        arm_hand_start_pose.p = gymapi.Vec3(-0.35, 0.0, 0.6)
        arm_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0.0)

        # create table asset
        table_dims = gymapi.Vec3(1.5, 1.5, 0.6)
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset_options.flip_visual_attachments = True
        table_asset_options.collapse_fixed_joints = True
        table_asset_options.disable_gravity = True
        table_asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        # create box asset
        box_assets = []
        box_start_poses = []

        box_thin = 0.01
        # box_xyz = [0.343, 0.279, 0.127]
        box_xyz = [0.60, 0.416, 0.165]

        box_offset = [0.25, 0.19, 0]

        box_asset_options = gymapi.AssetOptions()
        box_asset_options.disable_gravity = False
        box_asset_options.fix_base_link = True
        box_asset_options.flip_visual_attachments = True
        box_asset_options.collapse_fixed_joints = True
        box_asset_options.disable_gravity = True
        box_asset_options.thickness = 0.001

        box_bottom_asset = self.gym.create_box(self.sim, box_xyz[0], box_xyz[1], box_thin, table_asset_options)
        box_left_asset = self.gym.create_box(self.sim, box_xyz[0], box_thin, box_xyz[2], table_asset_options)
        box_right_asset = self.gym.create_box(self.sim, box_xyz[0], box_thin, box_xyz[2], table_asset_options)
        box_former_asset = self.gym.create_box(self.sim, box_thin, box_xyz[1], box_xyz[2], table_asset_options)
        box_after_asset = self.gym.create_box(self.sim, box_thin, box_xyz[1], box_xyz[2], table_asset_options)

        box_bottom_start_pose = gymapi.Transform()
        box_bottom_start_pose.p = gymapi.Vec3(0.0 + box_offset[0], 0.0 + box_offset[1], 0.6 + (box_thin) / 2)
        box_left_start_pose = gymapi.Transform()
        box_left_start_pose.p = gymapi.Vec3(0.0 + box_offset[0], (box_xyz[1] - box_thin) / 2 + box_offset[1], 0.6 + (box_xyz[2]) / 2)
        box_right_start_pose = gymapi.Transform()
        box_right_start_pose.p = gymapi.Vec3(0.0 + box_offset[0], -(box_xyz[1] - box_thin) / 2 + box_offset[1], 0.6 + (box_xyz[2]) / 2)
        box_former_start_pose = gymapi.Transform()
        box_former_start_pose.p = gymapi.Vec3((box_xyz[0] - box_thin) / 2 + box_offset[0], 0.0 + box_offset[1], 0.6 + (box_xyz[2]) / 2)
        box_after_start_pose = gymapi.Transform()
        box_after_start_pose.p = gymapi.Vec3(-(box_xyz[0] - box_thin) / 2 + box_offset[0], 0.0 + box_offset[1], 0.6 + (box_xyz[2]) / 2)

        box_assets.append(box_bottom_asset)
        box_assets.append(box_left_asset)
        box_assets.append(box_right_asset)
        box_assets.append(box_former_asset)
        box_assets.append(box_after_asset)
        box_start_poses.append(box_bottom_start_pose)
        box_start_poses.append(box_left_start_pose)
        box_start_poses.append(box_right_start_pose)
        box_start_poses.append(box_former_start_pose)
        box_start_poses.append(box_after_start_pose)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0, 0.0, -10.78)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 1.571)

        if self.object_type == "pen":
            object_start_pose.p.z = arm_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, -10.12)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.04

        lego_path = "urdf/blender/urdf/"
        all_lego_files_name = os.listdir("../assets/" + lego_path)
        all_lego_files_name = ['1x2.urdf', '1x2_curve.urdf', '1x3_curve_soft.urdf', '1x3_curve.urdf', '1x1.urdf', '1x3.urdf', '1x4.urdf', '2x2_curve_soft.urdf']
        # all_lego_files_name = ['1x2.urdf', '1x2_curve.urdf', '1x3_curve_soft.urdf', '1x3_curve.urdf', '1x1.urdf', '1x3.urdf', '1x2.urdf', '1x2_curve.urdf']

        self.lego_assets = []
        lego_start_poses = []
        self.segmentation_id = 1
        self.num_object_bodies = 0
        self.num_object_shapes = 0
        for n in range(12):
            # if n > 0:
            #     all_lego_files_name = ['1x2.urdf', '1x2_curve.urdf', '1x1.urdf', '1x3_curve_soft.urdf', '1x1.urdf', '1x2.urdf', '1x2.urdf', '1x2_curve.urdf']
            for i, lego_file_name in enumerate(all_lego_files_name):
                lego_asset_options = gymapi.AssetOptions()
                lego_asset_options.disable_gravity = False
                # lego_asset_options.fix_base_link = Trueyg
                # lego_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
                # lego_asset_options.override_com = True
                # lego_asset_options.override_inertia = True
                # lego_asset_options.vhacd_enabled = True
                # lego_asset_options.vhacd_params = gymapi.VhacdParams()
                # lego_asset_options.vhacd_params.resolution = 100000
                lego_asset_options.thickness = 0.00001
                lego_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                # lego_asset_options.density = 1000
                lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)

                lego_start_pose = gymapi.Transform()
                # if n > 0:
                #     lego_start_pose.p = gymapi.Vec3(-0.15 + 0.1 * int(i % 4) + 0.1, -0.25 + 0.1 * int(i % 24 / 4), 0.62 + 0.15 * int(i / 24) + n * 0.2 + 0.2)
                # else:

                if i == 4:
                    i = 0
                elif i == 0:
                    i = 4

                if n % 2 == 0:
                    lego_start_pos = [-0.17 + 0.17 * int(i % 3) + 0.25, -0.13 + 0.13 * int(i / 3) + 0.19, 0.7 + n * 0.06]
                else:
                    lego_start_pos = [0.17 - 0.17 * int(i % 3) + 0.25, 0.13 - 0.13 * int(i / 3) + 0.19, 0.7 + n * 0.06]
                
                # if n % 2 == 0:
                #     lego_start_pos = [-0.2 + 0.2 * int(i % 3) + 0.25, -0.19 + 0.19 * int(i / 3) + 0.19, 0.7 + n * 0.06]
                # else:
                #     lego_start_pos = [0.2 - 0.2 * int(i % 3) + 0.25, 0.19 - 0.19 * int(i / 3) + 0.19, 0.7 + n * 0.06]
                

                # if int(i % 3) == 1 and int(i / 3) == 1:
                #     lego_start_pos[0] += 10.1
                # if lego_start_pos[0] <= 0.2:
                #     lego_start_pos[0] += 0.3

                lego_start_pose.p = gymapi.Vec3(lego_start_pos[0], lego_start_pos[1], lego_start_pos[2])
                
                lego_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0, 0.785)
                # Assets visualization
                # lego_start_pose.p = gymapi.Vec3(-0.15 + 0.2 * int(i % 18) + 0.1, 0, 0.62 + 0.2 * int(i / 18) + n * 0.8 + 5.0)
                # lego_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0, 0)
                
                self.num_object_bodies += self.gym.get_asset_rigid_body_count(lego_asset) 
                self.num_object_shapes += self.gym.get_asset_rigid_shape_count(lego_asset) 
                print("num_object_shapes: ", self.num_object_shapes)
                print("num_object_bodies: ", self.num_object_bodies)

                self.lego_assets.append(lego_asset)
                lego_start_poses.append(lego_start_pose)

        all_lego_files_name = ['1x2.urdf', '1x2_curve.urdf', '1x3_curve_soft.urdf', '1x3_curve.urdf', '1x1.urdf', '1x3.urdf', '1x4.urdf', '2x2_curve_soft.urdf']
        # all_lego_files_name = ['1x2.urdf', '1x2_curve.urdf', '1x3_curve_soft.urdf', '1x3_curve.urdf', '1x1.urdf', '1x3.urdf', '1x2.urdf', '1x2_curve.urdf']

        lego_asset_options = gymapi.AssetOptions()
        lego_asset_options.disable_gravity = False
        # lego_asset_options.fix_base_link = True
        lego_asset_options.thickness = 0.00001
        lego_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # lego_asset_options.density = 2000
        flat_lego_begin = len(self.lego_assets)        
        ran_list = [0 ,0 ,0, 1, 2, 2]
        lego_list = [0, 5, 6]
        bianchang = [0.03, 0.045, 0.06]        
        for j in range(10):
            random.shuffle(ran_list)
            lego_center = [0.254 - bianchang[ran_list[0]] + 0.25, 0.175 + 0.19 - 0.039 * j, 0.63]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0] , lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[0]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            self.lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)            
            lego_center = [lego_center[0] - (bianchang[ran_list[0]] + bianchang[ran_list[1]] + 0.006), lego_center[1], lego_center[2]]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0], lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[1]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            self.lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)            
            lego_center = [lego_center[0] - (bianchang[ran_list[1]] + bianchang[ran_list[2]] + 0.006), lego_center[1], lego_center[2]]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0], lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[2]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            self.lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)            
            lego_center = [lego_center[0] - (bianchang[ran_list[2]] + bianchang[ran_list[3]] + 0.006), lego_center[1], lego_center[2]]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0], lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[3]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            self.lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)            
            lego_center = [lego_center[0] - (bianchang[ran_list[3]] + bianchang[ran_list[4]] + 0.006), lego_center[1], lego_center[2]]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0], lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[4]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            self.lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)            
            lego_center = [lego_center[0] - (bianchang[ran_list[4]] + bianchang[ran_list[5]] + 0.006), lego_center[1], lego_center[2]]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0], lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[5]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            self.lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)        
        
            self.num_object_bodies += self.gym.get_asset_rigid_body_count(lego_asset)
            self.num_object_shapes += self.gym.get_asset_rigid_shape_count(lego_asset)
            print("num_object_shapes: ", self.num_object_shapes)
            print("num_object_bodies: ", self.num_object_bodies)

        flat_lego_end = len(self.lego_assets)

        lego_path = "urdf/blender/urdf/"
        all_lego_files_name = os.listdir("../assets/" + lego_path)
        all_lego_files_name = ['1x2.urdf', '1x2_curve.urdf', '1x3_curve_soft.urdf', '1x3_curve.urdf', '1x1.urdf', '1x3.urdf', '1x4.urdf', '2x2_curve_soft.urdf']

        insert_lego_assets = []
        insert_lego_start_poses = []
        for n in range(1):
            for i, lego_file_name in enumerate(all_lego_files_name):
                insert_lego_asset_options = gymapi.AssetOptions()
                insert_lego_asset_options.disable_gravity = False
                # lego_asset_options.fix_base_link = True
                insert_lego_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
                insert_lego_asset_options.override_com = True
                insert_lego_asset_options.override_inertia = True
                insert_lego_asset_options.vhacd_enabled = True
                insert_lego_asset_options.vhacd_params = gymapi.VhacdParams()
                insert_lego_asset_options.vhacd_params.resolution = 500000
                # lego_asset_options.vhacd_params.convex_hull_downsampling = 16
                # lego_asset_options.vhacd_params.mode = 1
                insert_lego_asset_options.vhacd_params.max_convex_hulls = 512
                insert_lego_asset_options.vhacd_params.convex_hull_approximation = False

                insert_lego_asset_options.thickness = 0.00001
                insert_lego_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

                insert_lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, insert_lego_asset_options)

                self.num_object_bodies += self.gym.get_asset_rigid_body_count(insert_lego_asset)
                self.num_object_shapes += self.gym.get_asset_rigid_shape_count(insert_lego_asset)
                print("num_object_shapes: ", self.num_object_shapes)
                print("num_object_bodies: ", self.num_object_bodies)

                insert_lego_start_pose = gymapi.Transform()
                # if n > 0:
                #     lego_start_pose.p = gymapi.Vec3(-0.15 + 0.1 * int(i % 4) + 0.1, -0.25 + 0.1 * int(i % 24 / 4), 0.62 + 0.15 * int(i / 24) + n * 0.2 + 0.2)
                # else:
                insert_lego_start_pose.p = gymapi.Vec3(-1.03 - 0.13 * int(i % 3) + 0.1, 0.23 - 0.23 * int(i / 3), 0.22 + n * 0.07)

                insert_lego_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0.0, 0.785)

                insert_lego_assets.append(insert_lego_asset)
                insert_lego_start_poses.append(insert_lego_start_pose)

        inserted_lego_assets = []
        inserted_lego_start_poses = []
        self.inserted_lego_type = []
        
        for n in range(3):
            for i, lego_file_name in enumerate(all_lego_files_name):
                inserted_lego_asset_options = gymapi.AssetOptions()
                inserted_lego_asset_options.disable_gravity = False
                inserted_lego_asset_options.fix_base_link = True
                inserted_lego_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
                inserted_lego_asset_options.override_com = True
                inserted_lego_asset_options.override_inertia = True
                inserted_lego_asset_options.vhacd_enabled = True
                inserted_lego_asset_options.vhacd_params = gymapi.VhacdParams()
                inserted_lego_asset_options.vhacd_params.resolution = 500000
                # lego_asset_options.vhacd_params.convex_hull_downsampling = 16
                # lego_asset_options.vhacd_params.mode = 1
                inserted_lego_asset_options.vhacd_params.max_convex_hulls = 512
                inserted_lego_asset_options.vhacd_params.convex_hull_approximation = False

                inserted_lego_asset_options.thickness = 0.00001
                inserted_lego_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

                inserted_lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, inserted_lego_asset_options)

                self.num_object_bodies += self.gym.get_asset_rigid_body_count(inserted_lego_asset)
                self.num_object_shapes += self.gym.get_asset_rigid_shape_count(inserted_lego_asset)
                print("num_object_shapes: ", self.num_object_shapes)
                print("num_object_bodies: ", self.num_object_bodies)

                inserted_lego_start_pose = gymapi.Transform()
                # if n > 0:
                #     lego_start_pose.p = gymapi.Vec3(-0.15 + 0.1 * int(i % 4) + 0.1, -0.25 + 0.1 * int(i % 24 / 4), 0.62 + 0.15 * int(i / 24) + n * 0.2 + 0.2)
                # else:
                inserted_lego_start_pose.p = gymapi.Vec3(1.73 - 0.13 * int(i % 3) + 0.1, 0.23 - 0.23 * int(i / 3), 0.22 + n * 0.07)

                inserted_lego_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0.0, 0.785)

                inserted_lego_assets.append(inserted_lego_asset)
                inserted_lego_start_poses.append(inserted_lego_start_pose)

        interface_lego_assets = []
        interface_lego_start_poses = []
        self.interface_lego_type = []
        
        for n in range(1):
            for i, lego_file_name in enumerate(all_lego_files_name):
                interface_lego_asset_options = gymapi.AssetOptions()
                interface_lego_asset_options.disable_gravity = False
                interface_lego_asset_options.fix_base_link = True
                interface_lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, interface_lego_asset_options)

                self.num_object_bodies += self.gym.get_asset_rigid_body_count(interface_lego_asset)
                self.num_object_shapes += self.gym.get_asset_rigid_shape_count(interface_lego_asset)
                print("num_object_shapes: ", self.num_object_shapes)
                print("num_object_bodies: ", self.num_object_bodies)

                interface_lego_start_pose = gymapi.Transform()
                # if n > 0:
                #     lego_start_pose.p = gymapi.Vec3(-0.15 + 0.1 * int(i % 4) + 0.1, -0.25 + 0.1 * int(i % 24 / 4), 0.62 + 0.15 * int(i / 24) + n * 0.2 + 0.2)
                # else:
                interface_lego_start_pose.p = gymapi.Vec3(2.73 - 0.13 * int(i % 3) + 0.1, 0.23 - 0.23 * int(i / 3), 0.22 + n * 0.07)

                interface_lego_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0.0, 0.785)

                interface_lego_assets.append(interface_lego_asset)
                interface_lego_start_poses.append(interface_lego_start_pose)


        # create grasp table and box asset
        extra_lego_asset_options = gymapi.AssetOptions()
        extra_lego_asset_options.disable_gravity = False
        extra_lego_asset_options.fix_base_link = True
        extra_lego_asset_options.density = 500
        extra_lego_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        extra_lego_asset_options.override_com = True
        extra_lego_asset_options.override_inertia = True
        extra_lego_asset_options.vhacd_enabled = True
        extra_lego_asset_options.vhacd_params = gymapi.VhacdParams()
        extra_lego_asset_options.vhacd_params.resolution = 5000000
        extra_lego_asset_options.vhacd_params.convex_hull_approximation = False
        extra_lego_asset_options.vhacd_params.max_convex_hulls = 512
        extra_lego_asset_options.thickness = 0.00001
        extra_lego_asset = self.gym.load_asset(self.sim, asset_root, "urdf/blender/assets_for_insertion/urdf/8x8x1_real.urdf", extra_lego_asset_options)
        # extra_lego_asset = self.gym.load_asset(self.sim, asset_root, "urdf/blender/assets_for_insertion/urdf/4x4x1_real.urdf", extra_lego_asset_options)

        self.num_object_bodies += self.gym.get_asset_rigid_body_count(extra_lego_asset)
        self.num_object_shapes += self.gym.get_asset_rigid_shape_count(extra_lego_asset)
        print("num_extra_lego_asset_shapes: ", self.num_object_shapes)
        print("num_extra_lego_asset_bodies: ", self.num_object_bodies)

        extra_lego_start_pose = gymapi.Transform()
        extra_lego_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0.0, 0.0)
        # Assets visualization
        extra_lego_start_pose.p = gymapi.Vec3(0.25, -0.19, 0.618)

        # compute aggregate size
        max_agg_bodies = self.num_arm_hand_bodies + 2 + 5 + 5 + self.num_object_bodies + 10
        max_agg_shapes = self.num_arm_hand_shapes + 2 + 5 + 5 + self.num_object_shapes + 100

        self.arm_hands = []
        self.envs = []

        self.object_init_state = []
        self.lego_init_states = []
        self.hand_start_states = []
        self.extra_lego_init_states = []
        self.insert_lego_init_states = []
        self.inserted_lego_init_states = []
        self.interface_lego_init_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.predict_object_indices = []
        self.table_indices = []
        self.lego_indices = []
        self.lego_segmentation_indices = []
        self.extra_object_indices = []
        self.insert_lego_indices = []
        self.inserted_lego_indices = []
        self.interface_lego_indices = []
        self.no_collison_hand_indices = []

        self.segmentation_id_list = []

        arm_hand_rb_count = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        self.object_rb_handles = list(range(arm_hand_rb_count, arm_hand_rb_count + object_rb_count))

        self.cameras = []
        self.camera_tensors = []
        self.camera_seg_tensors = []
        self.camera_view_matrixs = []
        self.camera_proj_matrixs = []

        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 1024
        self.camera_props.height = 1024
        self.camera_props.horizontal_fov = 65

        self.camera_props.enable_tensors = True

        self.high_res_camera_tensors = []

        self.high_res_camera_props = gymapi.CameraProperties()
        self.high_res_camera_props.width = 2160
        self.high_res_camera_props.height = 1440
        self.high_res_camera_props.enable_tensors = True

        self.attach_camera_tensors = []
        self.attach_camera_seg_tensors = []

        self.attach_camera_props = gymapi.CameraProperties()
        self.attach_camera_props.width = 2048
        self.attach_camera_props.height = 2048
        self.attach_camera_props.enable_tensors = True

        self.camera_attach_pose = gymapi.Transform()
        self.camera_attach_pose.r = gymapi.Quat().from_euler_zyx(0, -3.141 + 0.5, 1.571)
        self.camera_attach_pose.p = gymapi.Vec3(0.03, 0.107-0.098, 0.067+0.107)

        self.camera_offset_quat = gymapi.Quat().from_euler_zyx(0, - 3.141 + 0.5, 1.571)
        self.camera_offset_quat = to_torch([self.camera_offset_quat.x, self.camera_offset_quat.y, self.camera_offset_quat.z, self.camera_offset_quat.w], device=self.device)
        self.camera_offset_pos = to_torch([0.03, 0.107 - 0.098, 0.067 + 0.107], device=self.device)

        self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
        self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)

        self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            arm_hand_actor = self.gym.create_actor(env_ptr, arm_hand_asset, arm_hand_start_pose, "hand", i, -1, 0)

            self.hand_start_states.append([arm_hand_start_pose.p.x,
                                           arm_hand_start_pose.p.y,
                                           arm_hand_start_pose.p.z,
                                           arm_hand_start_pose.r.x,
                                           arm_hand_start_pose.r.y,
                                           arm_hand_start_pose.r.z,
                                           arm_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, arm_hand_actor, robot_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, arm_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # arm_hand_start_pose.p.z -= 10
            # no_collision_arm_hand_actor = self.gym.create_actor(env_ptr, arm_hand_asset, arm_hand_start_pose, "no_collision_hand", i + self.num_envs, 0, 0)

            # self.gym.set_actor_dof_properties(env_ptr, no_collision_arm_hand_actor, robot_dof_props)
            # hand_idx = self.gym.get_actor_index(env_ptr, no_collision_arm_hand_actor, gymapi.DOMAIN_SIM)
            # self.no_collison_hand_indices.append(hand_idx)

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            # self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.gym.set_rigid_body_color(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.9, 0.8)
            )
            self.table_indices.append(table_idx)
            
            # table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            # for object_shape_prop in table_shape_props:
            #     object_shape_prop.friction = 1
            # self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            # add box
            for box_i, box_asset in enumerate(box_assets):
                box_handle = self.gym.create_actor(env_ptr, box_asset, box_start_poses[box_i], "box_{}".format(box_i), i, 0, 0)
                # self.lego_init_state.append([lego_init_state.p.x, lego_init_state.p.y, object_start_pose.p.z,
                #                             lego_init_state.r.x, lego_init_state.r.y, object_start_pose.r.z, object_start_pose.r.w,
                #                             0, 0, 0, 0, 0, 0])
                # object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
                # self.object_indices.append(object_idx)
                lego_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, box_handle)
                # for lego_body_prop in lego_body_props:
                #     # print(lego_body_prop.mass)
                #     lego_body_prop.mass *= 50
                # self.gym.set_actor_rigid_body_properties(env_ptr, box_handle, lego_body_props)

                self.gym.set_rigid_body_color(env_ptr, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))

            # add lego
            self.color_map = [[0.8, 0.64, 0.2], [0.13, 0.54, 0.13], [0, 0.4, 0.8], [1, 0.54, 0], [0.69, 0.13, 0.13], [0.69, 0.13, 0.13], [0, 0.4, 0.8], [0.13, 0.54, 0.13]]
            self.color_map_bottom = [[0, 0.4, 0.8], [0.13, 0.54, 0.13], [0, 0.4, 0.8], [1, 0.54, 0], [0.69, 0.13, 0.13], [0.69, 0.13, 0.13], [0, 0.4, 0.8], [0.13, 0.54, 0.13]]

            lego_idx = []
            lego_seg_idx = 1
            for lego_i, lego_asset in enumerate(self.lego_assets):
                lego_handle = self.gym.create_actor(env_ptr, lego_asset, lego_start_poses[lego_i], "lego_{}".format(lego_i), i, 0, lego_seg_idx)
                lego_seg_idx += 1
                # lego_handle = self.gym.create_actor(env_ptr, lego_asset, lego_start_poses[lego_i], "lego_{}".format(lego_i), i + self.num_envs + lego_i, -1, 0)
                self.lego_init_states.append([lego_start_poses[lego_i].p.x, lego_start_poses[lego_i].p.y, lego_start_poses[lego_i].p.z,
                                            lego_start_poses[lego_i].r.x, lego_start_poses[lego_i].r.y, lego_start_poses[lego_i].r.z, lego_start_poses[lego_i].r.w,
                                            0, 0, 0, 0, 0, 0])
                idx = self.gym.get_actor_index(env_ptr, lego_handle, gymapi.DOMAIN_SIM)
                if lego_i == self.segmentation_id:
                    self.lego_segmentation_indices.append(idx)

                lego_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, lego_handle)
                for lego_body_prop in lego_body_props:
                    # print(lego_body_prop.mass)
                    if flat_lego_end > lego_i > flat_lego_begin:
                        lego_body_prop.mass *= 1
                    lego_body_prop.mass *= 1
                self.gym.set_actor_rigid_body_properties(env_ptr, lego_handle, lego_body_props)

                lego_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, lego_handle)
                for object_shape_prop in lego_shape_props:
                    object_shape_prop.friction = 1
                    if flat_lego_end > lego_i > flat_lego_begin:
                        object_shape_prop.friction = 1
                    object_shape_prop.restitution = 0.0
                self.gym.set_actor_rigid_shape_properties(env_ptr, lego_handle, lego_shape_props)

                lego_idx.append(idx)
                color = self.color_map[lego_i % 8]
                if flat_lego_end > lego_i > flat_lego_begin:
                    color = self.color_map_bottom[random.randint(0, 7)]

                # if lego_i != self.segmentation_id:
                # color = [0.9, 0.9, 0.9]
                # self.gym.set_rigid_body_color(env_ptr, lego_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(color[0], color[1], color[2]))
                # color = self.color_map[lego_i % 8]
                self.gym.set_rigid_body_color(env_ptr, lego_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(color[0], color[1], color[2]))
            
            self.lego_indices.append(lego_idx)

            insert_lego_idx = []
            for insert_lego_i, insert_lego_asset in enumerate(insert_lego_assets):
                insert_lego_handle = self.gym.create_actor(env_ptr, insert_lego_asset, insert_lego_start_poses[insert_lego_i], "insert_lego_{}".format(insert_lego_i), i, 0, insert_lego_i + lego_i)
                self.insert_lego_init_states.append([insert_lego_start_poses[insert_lego_i].p.x, insert_lego_start_poses[insert_lego_i].p.y, insert_lego_start_poses[insert_lego_i].p.z,
                                            insert_lego_start_poses[insert_lego_i].r.x, insert_lego_start_poses[insert_lego_i].r.y, insert_lego_start_poses[insert_lego_i].r.z, insert_lego_start_poses[insert_lego_i].r.w,
                                            0, 0, 0, 0, 0, 0])
                
                insert_lego_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, insert_lego_handle)
                for insert_lego_body_prop in insert_lego_body_props:
                    # print(lego_body_prop.mass)
                    insert_lego_body_prop.mass *= 1
                self.gym.set_actor_rigid_body_properties(env_ptr, insert_lego_handle, insert_lego_body_props)

                insert_lego_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, insert_lego_handle)
                for insert_object_shape_prop in insert_lego_shape_props:
                    insert_object_shape_prop.friction = 1
                    insert_object_shape_prop.restitution = 0.00
                self.gym.set_actor_rigid_shape_properties(env_ptr, insert_lego_handle, insert_lego_shape_props)

                idx = self.gym.get_actor_index(env_ptr, insert_lego_handle, gymapi.DOMAIN_SIM)
                insert_lego_idx.append(idx)
                color = self.color_map[insert_lego_i % 8]
                self.gym.set_rigid_body_color(env_ptr, insert_lego_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(color[0], color[1], color[2]))
            self.insert_lego_indices.append(insert_lego_idx)

            inserted_lego_idx = []
            for inserted_lego_i, inserted_lego_asset in enumerate(inserted_lego_assets):
                inserted_lego_handle = self.gym.create_actor(env_ptr, inserted_lego_asset, inserted_lego_start_poses[inserted_lego_i], "inserted_lego_{}".format(inserted_lego_i), i, 0, inserted_lego_i + insert_lego_i + lego_i)
                self.inserted_lego_init_states.append([inserted_lego_start_poses[inserted_lego_i].p.x, inserted_lego_start_poses[inserted_lego_i].p.y, inserted_lego_start_poses[inserted_lego_i].p.z,
                                            inserted_lego_start_poses[inserted_lego_i].r.x, inserted_lego_start_poses[inserted_lego_i].r.y, inserted_lego_start_poses[inserted_lego_i].r.z, inserted_lego_start_poses[inserted_lego_i].r.w,
                                            0, 0, 0, 0, 0, 0])
                idx = self.gym.get_actor_index(env_ptr, inserted_lego_handle, gymapi.DOMAIN_SIM)
                inserted_lego_idx.append(idx)
                color = self.color_map[inserted_lego_i % 8]
                self.gym.set_rigid_body_color(env_ptr, inserted_lego_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(color[0], color[1], color[2]))
            self.inserted_lego_indices.append(inserted_lego_idx)

            interface_lego_idx = []
            for interface_lego_i, interface_lego_asset in enumerate(interface_lego_assets):
                interface_lego_handle = self.gym.create_actor(env_ptr, interface_lego_asset, interface_lego_start_poses[interface_lego_i], "interface_lego_{}".format(inserted_lego_i), i + self.num_envs, 0, interface_lego_i + inserted_lego_i + insert_lego_i + lego_i)
                self.interface_lego_init_states.append([interface_lego_start_poses[interface_lego_i].p.x, interface_lego_start_poses[interface_lego_i].p.y, interface_lego_start_poses[interface_lego_i].p.z,
                                            interface_lego_start_poses[interface_lego_i].r.x, interface_lego_start_poses[interface_lego_i].r.y, interface_lego_start_poses[interface_lego_i].r.z, interface_lego_start_poses[interface_lego_i].r.w,
                                            0, 0, 0, 0, 0, 0])
                idx = self.gym.get_actor_index(env_ptr, interface_lego_handle, gymapi.DOMAIN_SIM)
                interface_lego_idx.append(idx)
                color = self.color_map[interface_lego_i % 8]
                self.gym.set_rigid_body_color(env_ptr, interface_lego_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(color[0], color[1], color[2]))
            self.interface_lego_indices.append(interface_lego_idx)

            extra_lego_handle = self.gym.create_actor(env_ptr, extra_lego_asset, extra_lego_start_pose, "extra_lego", i, 0, 0)
            self.extra_lego_init_states.append([extra_lego_start_pose.p.x, extra_lego_start_pose.p.y, extra_lego_start_pose.p.z,
                                        extra_lego_start_pose.r.x, extra_lego_start_pose.r.y, extra_lego_start_pose.r.z, extra_lego_start_pose.r.w,
                                        0, 0, 0, 0, 0, 0])
            self.gym.get_actor_index(env_ptr, extra_lego_handle, gymapi.DOMAIN_SIM)
            extra_object_idx = self.gym.get_actor_index(env_ptr, extra_lego_handle, gymapi.DOMAIN_SIM)
            extra_lego_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, extra_lego_handle)
            for object_shape_prop in extra_lego_shape_props:
                object_shape_prop.friction = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, extra_lego_handle, extra_lego_shape_props)

            self.gym.set_rigid_body_color(env_ptr, extra_lego_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))
            self.extra_object_indices.append(extra_object_idx)

            # add grasp table and box
            # realsense_asset_options = gymapi.AssetOptions()
            # realsense_asset_options.fix_base_link = True
            # realsense_pose = gymapi.Transform()
            # realsense_pose.p = gymapi.Vec3(0.25, 0.35, 1.05)
            # realsense_pose.r = gymapi.Quat().from_euler_zyx(-1.5, 0, 3.14151)
            # realsense_asset = self.gym.load_asset(self.sim, asset_root, "urdf/franka_description/robots/realsense_d435.urdf", realsense_asset_options)

            # realsense_lego_handle = self.gym.create_actor(env_ptr, realsense_asset, realsense_pose, "camera", i, 0, 0)

            self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.75, 0.75, 0.75), 
                                            gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(1, 1, 1))
            self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.75, 0.75, 0.75), 
                                            gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(1, 1, 1))
            self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.75, 0.75, 0.75), 
                                            gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(1, 1, 1))
            self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.75, 0.75, 0.75), 
                                            gymapi.Vec3(0.5, 0.5, 0.5), gymapi.Vec3(1, 1, 1))

            # cam_pos = gymapi.Vec3(0.25, -0.075, 0.825)
            # cam_target = gymapi.Vec3(0.25, 0.45, 0.6)

            if self.enable_camera_sensors:
                camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(0.25, -0.075, 0.825), gymapi.Vec3(0.25, 0.45, 0.6))
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                camera_seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION)
                torch_cam_seg_tensor = gymtorch.wrap_tensor(camera_seg_tensor)

                cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)))).to(self.device)
                cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle), device=self.device)
            
            self.mount_rigid_body_index = self.gym.find_actor_rigid_body_index(env_ptr, arm_hand_actor, "panda_link7", gymapi.DOMAIN_ENV)
            print("mount_rigid_body_index: ", self.mount_rigid_body_index)

            if self.enable_camera_sensors:
                high_res_camera_handle = self.gym.create_camera_sensor(env_ptr, self.high_res_camera_props)
                self.gym.set_camera_location(high_res_camera_handle, env_ptr, gymapi.Vec3(0.7, -0.4, 1.3), gymapi.Vec3(-0.6, 0.4, 0.2))

                # self.gym.attach_camera_to_body(high_res_camera_handle, env_ptr, self.mount_rigid_body_index, self.camera_attach_pose, gymapi.FOLLOW_TRANSFORM)
                high_res_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, high_res_camera_handle, gymapi.IMAGE_COLOR)
                high_res_torch_cam_tensor = gymtorch.wrap_tensor(high_res_camera_tensor)
                self.high_res_camera_tensors.append(high_res_torch_cam_tensor)

            attach_camera_handle = self.gym.create_camera_sensor(env_ptr, self.attach_camera_props)
            self.gym.attach_camera_to_body(attach_camera_handle, env_ptr, self.mount_rigid_body_index, self.camera_attach_pose, gymapi.FOLLOW_TRANSFORM)
            attach_camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, attach_camera_handle, gymapi.IMAGE_COLOR)
            attach_torch_cam_tensor = gymtorch.wrap_tensor(attach_camera_tensor)
            self.attach_camera_tensors.append(attach_torch_cam_tensor)

            attach_camera_seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, attach_camera_handle, gymapi.IMAGE_SEGMENTATION)
            attach_torch_cam_seg_tensor = gymtorch.wrap_tensor(attach_camera_seg_tensor)
            self.attach_camera_seg_tensors.append(attach_torch_cam_seg_tensor)

            # Set up object...
            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98)
                )
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98)
                )

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            if self.enable_camera_sensors:
                origin = self.gym.get_env_origin(env_ptr)
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z
                self.camera_tensors.append(torch_cam_tensor)
                self.camera_seg_tensors.append(torch_cam_seg_tensor)
                self.camera_view_matrixs.append(cam_vinv)
                self.camera_proj_matrixs.append(cam_proj)
                self.cameras.append(camera_handle)

            self.envs.append(env_ptr)
            self.arm_hands.append(arm_hand_actor)

        self.camera_rgbd_image_tensors = torch.stack(self.camera_tensors, dim=0).view(self.num_envs, -1)
        self.camera_seg_image_tensors = ((torch.stack(self.camera_seg_tensors, dim=0) == self.segmentation_id) * 255).view(self.num_envs, -1)
        self.emergence_reward = torch.zeros_like(self.rew_buf, device=self.device, dtype=torch.float)
        self.emergence_pixel = torch.zeros_like(self.rew_buf, device=self.device, dtype=torch.float)
        self.last_emergence_pixel = torch.zeros_like(self.rew_buf, device=self.device, dtype=torch.float)

        self.heap_movement_penalty= torch.zeros_like(self.rew_buf, device=self.device, dtype=torch.float)

        # Acquire specific links.
        # sensor_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, sensor_name) for sensor_name in
        #                   self.contact_sensor_names]
        sensor_handles = [0, 1, 2, 3, 4, 5, 6]
        self.sensor_handle_indices = to_torch(sensor_handles, dtype=torch.int64)

        self.fingertip_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, name) for name in self.fingertip_names]
        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.02
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.lego_init_states = to_torch(self.lego_init_states, device=self.device).view(self.num_envs, len(self.lego_assets), 13)
        self.insert_lego_init_states = to_torch(self.insert_lego_init_states, device=self.device).view(self.num_envs, -1, 13)
        self.inserted_lego_init_states = to_torch(self.inserted_lego_init_states, device=self.device).view(self.num_envs, -1, 13)
        self.interface_lego_init_states = to_torch(self.interface_lego_init_states, device=self.device).view(self.num_envs, -1, 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.lego_indices = to_torch(self.lego_indices, dtype=torch.long, device=self.device)
        self.lego_segmentation_indices = to_torch(self.lego_segmentation_indices, dtype=torch.long, device=self.device)
        self.extra_object_indices = to_torch(self.extra_object_indices, dtype=torch.long, device=self.device)
        self.insert_lego_indices = to_torch(self.insert_lego_indices, dtype=torch.long, device=self.device)
        self.inserted_lego_indices = to_torch(self.inserted_lego_indices, dtype=torch.long, device=self.device)
        self.interface_lego_indices = to_torch(self.interface_lego_indices, dtype=torch.long, device=self.device)
        self.no_collison_hand_indices = to_torch(self.no_collison_hand_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            torch.tensor(self.spin_coef).to(self.device), self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes, self.hand_reset_step, self.contacts, self.palm_contacts_z, self.segmentation_object_point_num.squeeze(-1),
            self.max_episode_length, self.object_pos, self.object_rot, self.object_angvel, self.goal_pos, self.goal_rot, self.segmentation_target_pos, self.hand_base_pos, self.emergence_reward, self.arm_hand_ff_pos, self.arm_hand_rf_pos, self.arm_hand_mf_pos, self.arm_hand_th_pos, self.heap_movement_penalty,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty, self.rotation_id,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen"), 
        )

        self.meta_rew_buf += self.rew_buf[:].clone()

        self.extras['emergence_reward'] = self.emergence_reward
        self.extras['heap_movement_penalty'] = self.heap_movement_penalty
        self.extras['meta_reward'] = self.meta_rew_buf

        self.total_steps += 1
        # print("Total epoch = {}".format(int(self.total_steps/8)))

        if self.print_success_stat:
            print("Total steps = {}".format(self.total_steps))
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        if self.record_camera_view_image:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
        
            camera_rgba_image = self.camera_rgb_visulization(self.camera_tensors, env_id=0, is_depth_image=False)
            hi_camera_rgba_image = self.camera_rgb_visulization(self.high_res_camera_tensors, env_id=0, is_depth_image=False)

            cv2.imshow("DEBUG_RGB_VIS", camera_rgba_image)
            cv2.imshow("DEBUG_HI_RGB_VIS", hi_camera_rgba_image)
            cv2.waitKey(1)

            self.gym.end_access_image_tensors(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.hand_base_pose = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:7]
        self.hand_base_pos = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3]
        self.hand_base_rot = self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7]
        self.hand_base_linvel = self.rigid_body_states[:, self.hand_base_rigid_body_index, 7:10]
        self.hand_base_angvel = self.rigid_body_states[:, self.hand_base_rigid_body_index, 10:13]

        self.hand_pos_history[:, self.progress_buf[0] - 1, :] = self.hand_base_pos.clone()

        self.segmentation_target_state = self.root_state_tensor[self.lego_segmentation_indices, 0:13]
        self.segmentation_target_pose = self.root_state_tensor[self.lego_segmentation_indices, 0:7]
        self.segmentation_target_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3]
        self.segmentation_target_rot = self.root_state_tensor[self.lego_segmentation_indices, 3:7]
        self.segmentation_target_linvel = self.root_state_tensor[self.lego_segmentation_indices, 7:10]
        self.segmentation_target_angvel = self.root_state_tensor[self.lego_segmentation_indices, 10:13]

        self.extra_target_pose = self.root_state_tensor[self.extra_object_indices, 0:7]
        self.extra_target_pos = self.root_state_tensor[self.extra_object_indices, 0:3]
        self.extra_target_rot = self.root_state_tensor[self.extra_object_indices, 3:7]
        self.extra_target_linvel = self.root_state_tensor[self.extra_object_indices, 7:10]
        self.extra_target_angvel = self.root_state_tensor[self.extra_object_indices, 10:13]

        self.arm_hand_ff_pos = self.rigid_body_states[:, self.fingertip_handles[0], 0:3]
        self.arm_hand_ff_rot = self.rigid_body_states[:, self.fingertip_handles[0], 3:7]
        self.arm_hand_ff_linvel = self.rigid_body_states[:, self.fingertip_handles[0], 7:10]
        self.arm_hand_ff_angvel = self.rigid_body_states[:, self.fingertip_handles[0], 10:13]

        # self.arm_hand_ff_pos = self.arm_hand_ff_pos + quat_apply(self.arm_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.arm_hand_mf_pos = self.rigid_body_states[:, self.fingertip_handles[1], 0:3]
        self.arm_hand_mf_rot = self.rigid_body_states[:, self.fingertip_handles[1], 3:7]
        self.arm_hand_mf_linvel = self.rigid_body_states[:, self.fingertip_handles[1], 7:10]
        self.arm_hand_mf_angvel = self.rigid_body_states[:, self.fingertip_handles[1], 10:13]

        self.arm_hand_rf_pos = self.rigid_body_states[:, self.fingertip_handles[2], 0:3]
        self.arm_hand_rf_rot = self.rigid_body_states[:, self.fingertip_handles[2], 3:7]
        self.arm_hand_rf_linvel = self.rigid_body_states[:, self.fingertip_handles[2], 7:10]
        self.arm_hand_rf_angvel = self.rigid_body_states[:, self.fingertip_handles[2], 10:13]
        # self.arm_hand_lf_rot = self.rigid_body_states[:, 20, 3:7]
        # self.arm_hand_lf_pos = self.arm_hand_lf_pos + quat_apply(self.arm_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.arm_hand_th_pos = self.rigid_body_states[:, self.fingertip_handles[3], 0:3]
        self.arm_hand_th_rot = self.rigid_body_states[:, self.fingertip_handles[3], 3:7]
        self.arm_hand_th_linvel = self.rigid_body_states[:, self.fingertip_handles[3], 7:10]
        self.arm_hand_th_angvel = self.rigid_body_states[:, self.fingertip_handles[3], 10:13]

        self.arm_hand_ff_state = self.rigid_body_states[:, self.fingertip_handles[0], 0:13]
        self.arm_hand_mf_state = self.rigid_body_states[:, self.fingertip_handles[1], 0:13]
        self.arm_hand_rf_state = self.rigid_body_states[:, self.fingertip_handles[2], 0:13]
        self.arm_hand_th_state = self.rigid_body_states[:, self.fingertip_handles[3], 0:13]

        self.arm_hand_ff_pos = self.arm_hand_ff_pos + quat_apply(self.arm_hand_ff_rot[:], to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.arm_hand_mf_pos = self.arm_hand_mf_pos + quat_apply(self.arm_hand_mf_rot[:], to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.arm_hand_rf_pos = self.arm_hand_rf_pos + quat_apply(self.arm_hand_rf_rot[:], to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.arm_hand_th_pos = self.arm_hand_th_pos + quat_apply(self.arm_hand_th_rot[:], to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.04)

        contacts = self.contact_tensor.reshape(self.num_envs, -1, 3)  # 39+27
        palm_contacts = contacts[:, 8, :]
        contacts = contacts[:, self.sensor_handle_indices, :] # 12
        contacts = torch.norm(contacts, dim=-1)
        self.contacts = torch.where(contacts >= 0.1, 1.0, 0.0)

        self.palm_contacts_z = palm_contacts[:, 2]
        # self.palm_contacts = torch.where(palm_contacts_z >= 100, 1.0, 0.0)

        # if self.palm_contacts_z[0] > 100.0:
        #     self.gym.set_rigid_body_color(
        #                 self.envs[0], self.hand_indices[0], 8, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.3, 0.3))
        # else:
        #     self.gym.set_rigid_body_color(
        #                 self.envs[0], self.hand_indices[0], 8, gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))

        # for i in range(len(self.contacts[0])):
        #     if self.contacts[0][i] == 1.0:
        #         self.gym.set_rigid_body_color(
        #                     self.envs[0], self.hand_indices[0], self.sensor_handle_indices[i], gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.3, 0.3))
        #     else:
        #         self.gym.set_rigid_body_color(
        #                     self.envs[0], self.hand_indices[0], self.sensor_handle_indices[i], gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))
        #
        self.seg_idx = 5

        attach_camera_seg_image = self.camera_segmentation_visulization(self.attach_camera_tensors, self.attach_camera_seg_tensors, segmentation_id=self.seg_idx + 1, env_id=0, is_depth_image=False)
        self.segmentation_object_point_list = torch.nonzero(torch.where(self.attach_camera_seg_tensors[0] == self.seg_idx + 1, self.attach_camera_seg_tensors[0], torch.zeros_like(self.attach_camera_seg_tensors[0])))
        self.segmentation_object_point_list = self.segmentation_object_point_list.float()            
        # self.segmentation_object_point_num = self.segmentation_object_point_list.shape[0]

        # print(self.segmentation_object_point_num)
        axis1 = quat_apply(self.root_state_tensor[self.lego_indices.view(-1)[self.seg_idx], 3:7], self.z_unit_tensor)
        axis2 = self.z_unit_tensor
        dot1 = torch.bmm(axis1.view(self.num_envs, 1, 3), axis2.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
        self.lego_z_align_reward = (torch.sign(dot1) * dot1 ** 2)

        if self.record_camera_view_image:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            camera_rgba_image = self.camera_rgb_visulization(self.camera_tensors, env_id=0, is_depth_image=False)
            attach_camera_rgba_image = self.camera_rgb_visulization(self.attach_camera_tensors, env_id=0, is_depth_image=False)
        
            attach_camera_seg_image[self.attach_camera_seg_tensors[0].cpu() == self.seg_idx + 1] = 255
            attach_camera_seg_image_mask = attach_camera_seg_image.copy()
            attach_camera_seg_image_mask[self.attach_camera_seg_tensors[0].cpu() != self.seg_idx + 1] = 128
            concat_frame = cv2.hconcat([attach_camera_seg_image_mask, attach_camera_seg_image, attach_camera_rgba_image])
            self.out.write(concat_frame)        
            cv2.imshow('frame',concat_frame)
            cv2.waitKey(1)
            self.gym.end_access_image_tensors(self.sim)

        if self.record_inference_video:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            hi_camera_rgba_image = self.camera_rgb_visulization(self.high_res_camera_tensors, env_id=0, is_depth_image=False)
            self.out.write(hi_camera_rgba_image)  
            cv2.imshow('Frame', hi_camera_rgba_image)
            cv2.waitKey(1)      
            self.gym.end_access_image_tensors(self.sim)

        self.arm_hand_finger_dist = (torch.norm(self.segmentation_target_pos - self.arm_hand_ff_pos, p=2, dim=-1) + torch.norm(self.segmentation_target_pos - self.arm_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(self.segmentation_target_pos - self.arm_hand_rf_pos, p=2, dim=-1) + torch.norm(self.segmentation_target_pos - self.arm_hand_th_pos, p=2, dim=-1))

        # visual test
        self.robot_base_pos = self.root_state_tensor[self.hand_indices, 0:3]
        self.robot_base_rot = self.root_state_tensor[self.hand_indices, 3:7]

        self.q_robot_base_inv, self.p_robot_base_inv = tf_inverse(self.robot_base_rot, self.robot_base_pos)
        self.hand_base_view_hand_rot, self.hand_base_view_hand_pos = tf_combine(self.q_robot_base_inv, self.p_robot_base_inv, self.hand_base_rot, self.hand_base_pos)

        # object 6d pose randomization
        self.mount_pos = self.rigid_body_states[:, self.mount_rigid_body_index, 0:3]
        self.mount_rot = self.rigid_body_states[:, self.mount_rigid_body_index, 3:7]

        self.q_camera, self.p_camera = tf_combine(self.mount_rot, self.mount_pos, self.camera_offset_quat.repeat(self.num_envs, 1), self.camera_offset_pos.repeat(self.num_envs, 1))
        self.q_camera_inv, self.p_camera_inv = tf_inverse(self.q_camera, self.p_camera)

        self.camera_view_segmentation_target_rot, self.camera_view_segmentation_target_pos = tf_combine(self.q_camera_inv, self.p_camera_inv, self.segmentation_target_rot, self.segmentation_target_pos)

        pose_rand_floats = torch_rand_float(-1, 1, (self.num_envs, 7), device=self.device)
        self.camera_view_segmentation_target_pos_noise = self.camera_view_segmentation_target_pos + pose_rand_floats[:, 0:3] * 0.01
        self.camera_view_segmentation_target_rot_noise = self.camera_view_segmentation_target_rot + pose_rand_floats[:, 3:7] * 0.01

        self.hand_object_dist = torch.norm(self.segmentation_target_pos - self.hand_base_pos, p=2, dim=-1)

        self.camera_view_segmentation_target_init_rot, self.camera_view_segmentation_target_init_pos = tf_combine(self.q_camera_inv, self.p_camera_inv, self.segmentation_target_init_rot, self.segmentation_target_init_pos)
        axis1 = quat_apply(self.root_state_tensor[self.lego_segmentation_indices, 3:7], self.z_unit_tensor)
        axis2 = self.z_unit_tensor
        dot1 = torch.bmm(axis1.view(self.num_envs, 1, 3), axis2.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
        self.lego_z_align_reward = (torch.sign(dot1) * dot1 ** 2)

        if self.progress_buf[0] <= 241:
            self.compute_digging_observations(False)
        elif 361 >= self.progress_buf[0] > 241:
            self.compute_spin_observations()
        elif 536 >= self.progress_buf[0] > 361:
            self.compute_grasping_observations()
        elif self.progress_buf[0] >= 536:
            self.compute_insertion_observations()
        else:
            print("Unknown observations type!")

    def compute_digging_observations(self, full_contact=True):
        self.obs_buf[:, 0:16] = unscale(self.arm_hand_dof_pos[:, 7:23],
                                                            self.arm_hand_dof_lower_limits[7:23],
                                                            self.arm_hand_dof_upper_limits[7:23])

        self.obs_buf[:, 30:46] = self.actions[:, 7:23] - unscale(self.arm_hand_dof_pos[:, 7:23],
                                                            self.arm_hand_dof_lower_limits[7:23],
                                                            self.arm_hand_dof_upper_limits[7:23])
        
        self.obs_buf[:, 46:62] = self.actions[:, 7:23]

        for i in range(len(self.obs_buf_stack_frames) - 1):
            self.obs_buf[:, (i+1) * self.one_frame_num_obs:(i+2) * self.one_frame_num_obs] = self.obs_buf_stack_frames[i]
            self.obs_buf_stack_frames[i] = self.obs_buf[:, (i) * self.one_frame_num_obs:(i+1) * self.one_frame_num_obs].clone()

    def compute_spin_observations(self, full_contact=True):        
        # self.spin_obs_buf[:, 0:16] = unscale(self.arm_hand_dof_pos[:, 7:23],
        #                                                     self.arm_hand_dof_lower_limits[7:23],
        #                                                     self.arm_hand_dof_upper_limits[7:23])


        # self.spin_obs_buf[:, 16:19] = self.camera_view_segmentation_target_pos
        # self.spin_obs_buf[:, 19:23] = self.camera_view_segmentation_target_rot
        # self.spin_obs_buf[:, 16:19] = torch.tensor([0.2048, 0.0404, 0.0361], device=self.device)

        # self.spin_obs_buf[:, 23:26] = self.camera_view_segmentation_target_init_pos
        # self.spin_obs_buf[:, 26:30] = self.camera_view_segmentation_target_init_rot
        # self.spin_obs_buf[:, 23:26] = torch.tensor([0.2048, 0.0404, 0.0361], device=self.device)

        # self.spin_obs_buf[:, 30:46] = self.actions[:, 7:23] - unscale(self.arm_hand_dof_pos[:, 7:23],
        #                                                     self.arm_hand_dof_lower_limits[7:23],
        #                                                     self.arm_hand_dof_upper_limits[7:23])
        
        # self.spin_obs_buf[:, 46:62] = self.actions[:, 7:23]

        self.spin_obs_buf[:, 0:16] = unscale(self.arm_hand_dof_pos[:, 7:23],
                                                            self.arm_hand_dof_lower_limits[7:23],
                                                            self.arm_hand_dof_upper_limits[7:23])
    
        self.spin_obs_buf[:, 16:19] = self.hand_base_view_hand_pos
        self.spin_obs_buf[:, 19:23] = self.hand_base_view_hand_rot

        self.spin_obs_buf[:, 23:26] = self.camera_view_segmentation_target_pos
        self.spin_obs_buf[:, 26:30] = self.camera_view_segmentation_target_rot

        self.spin_obs_buf[:, 30:46] = self.vel_obs_scale * self.arm_hand_dof_vel[:, 7:23]

        self.spin_obs_buf[:, 46:59] = self.arm_hand_ff_state
        self.spin_obs_buf[:, 59:72] = self.arm_hand_rf_state
        self.spin_obs_buf[:, 72:85] = self.arm_hand_mf_state
        self.spin_obs_buf[:, 85:98] = self.arm_hand_th_state

        self.spin_obs_buf[:, 98:111] = self.segmentation_target_state

        self.spin_obs_buf[:, 111:114] = self.hand_base_pos
        self.spin_obs_buf[:, 114:118] = self.hand_base_rot

        self.spin_obs_buf[:, 118:121] = self.segmentation_target_init_pos
        self.spin_obs_buf[:, 121:125] = self.segmentation_target_init_rot

        self.spin_obs_buf[:, 125:128] = self.segmentation_target_pos - self.segmentation_target_init_pos
        self.spin_obs_buf[:, 128:131] = self.hand_base_pos - self.segmentation_target_pos

        self.spin_obs_buf[:, 131:132] = (self.progress_buf.unsqueeze(-1) - 241) / 150

        for i in range(len(self.spin_obs_buf_stack_frames) - 1):
            self.spin_obs_buf[:, (i+1) * self.spin_one_frame_num_obs:(i+2) * self.spin_one_frame_num_obs] = self.spin_obs_buf_stack_frames[i]
            self.spin_obs_buf_stack_frames[i] = self.spin_obs_buf[:, (i) * self.spin_one_frame_num_obs:(i+1) * self.spin_one_frame_num_obs].clone()

    def compute_grasping_observations(self, full_contact=True):        
        self.grasping_obs_buf[:, 0:16] = unscale(self.arm_hand_dof_pos[:, 7:23],
                                                            self.arm_hand_dof_lower_limits[7:23],
                                                            self.arm_hand_dof_upper_limits[7:23])
    
        self.grasping_obs_buf[:, 16:19] = self.hand_base_view_hand_pos
        self.grasping_obs_buf[:, 19:23] = self.hand_base_view_hand_rot

        self.grasping_obs_buf[:, 23:26] = self.camera_view_segmentation_target_pos
        self.grasping_obs_buf[:, 26:30] = self.camera_view_segmentation_target_rot

        self.grasping_obs_buf[:, 30:46] = self.vel_obs_scale * self.arm_hand_dof_vel[:, 7:23]

        self.grasping_obs_buf[:, 46:59] = self.arm_hand_ff_state
        self.grasping_obs_buf[:, 59:72] = self.arm_hand_rf_state
        self.grasping_obs_buf[:, 72:85] = self.arm_hand_mf_state
        self.grasping_obs_buf[:, 85:98] = self.arm_hand_th_state

        self.grasping_obs_buf[:, 98:111] = self.segmentation_target_state

        self.grasping_obs_buf[:, 111:114] = self.hand_base_pos
        self.grasping_obs_buf[:, 114:118] = self.hand_base_rot

        self.grasping_obs_buf[:, 118:121] = self.segmentation_target_init_pos
        self.grasping_obs_buf[:, 121:125] = self.segmentation_target_init_rot

        self.grasping_obs_buf[:, 125:128] = self.segmentation_target_pos - self.segmentation_target_init_pos
        self.grasping_obs_buf[:, 128:131] = self.hand_base_pos - self.segmentation_target_pos

        self.grasping_obs_buf[:, 131:132] = (self.progress_buf.unsqueeze(-1) + 20 - 361) / 150

        for i in range(len(self.grasping_obs_buf_stack_frames) - 1):
            self.grasping_obs_buf[:, (i+1) * self.grasping_one_frame_num_obs:(i+2) * self.grasping_one_frame_num_obs] = self.grasping_obs_buf_stack_frames[i]
            self.grasping_obs_buf_stack_frames[i] = self.grasping_obs_buf[:, (i) * self.grasping_one_frame_num_obs:(i+1) * self.grasping_one_frame_num_obs].clone()

    def compute_insertion_observations(self, full_contact=True):
        # full obs
        # self.insertion_obs_buf[:, 0:23] = unscale(self.arm_hand_dof_pos[:,0:23],
        #                                                     self.arm_hand_dof_lower_limits[0:23],
        #                                                     self.arm_hand_dof_upper_limits[0:23])
        # # self.obs_buf[:, 16:23] = self.goal_pose
        # self.insertion_obs_buf[:, 23:46] = self.actions

        # self.insertion_obs_buf[:, 46:53] = self.hand_base_pose

        # self.insertion_obs_buf[:, 53:56] = self.segmentation_target_pos
        # self.insertion_obs_buf[:, 56:60] = self.segmentation_target_rot

        # self.insertion_obs_buf[:, 60:61] = (self.progress_buf.unsqueeze(-1) - 536) / 125

        # self.insertion_obs_buf[:, 61:64] = self.target_pos
        # self.insertion_obs_buf[:, 64:68] = self.target_rot

        # self.insertion_obs_buf[:, 68:71] = self.segmentation_target_pos - self.target_pos
        # self.insertion_obs_buf[:, 71:75] = quat_mul(self.segmentation_target_rot, quat_conjugate(self.target_rot))

        # for i in range(len(self.insertion_obs_buf_stack_frames) - 1):
        #     self.insertion_obs_buf[:, (i+1) * self.insertion_one_frame_num_obs:(i+2) * self.insertion_one_frame_num_obs] = self.insertion_obs_buf_stack_frames[i]
        #     self.insertion_obs_buf_stack_frames[i] = self.insertion_obs_buf[:, (i) * self.insertion_one_frame_num_obs:(i+1) * self.insertion_one_frame_num_obs].clone()

        self.insertion_obs_buf[:, 0:16] = unscale(self.arm_hand_dof_pos[:,7:23],
                                                            self.arm_hand_dof_lower_limits[7:23],
                                                            self.arm_hand_dof_upper_limits[7:23])
        self.insertion_obs_buf[:, 23:46] = self.actions
        # self.obs_buf[:, 16:23] = self.goal_pose
        self.insertion_obs_buf[:, 46:49] = self.hand_base_pos - self.target_pos
        self.insertion_obs_buf[:, 49:53] = quat_mul(self.hand_base_rot, quat_conjugate(self.target_rot))

        self.insertion_obs_buf[:, 53:56] = self.hand_base_pos - self.segmentation_target_pos
        self.insertion_obs_buf[:, 56:60] = quat_mul(self.hand_base_rot, quat_conjugate(self.segmentation_target_rot))

        self.insertion_obs_buf[:, 61:64] = self.target_pos
        self.insertion_obs_buf[:, 64:68] = self.target_rot

        self.insertion_obs_buf[:, 68:71] = self.segmentation_target_pos - self.target_pos
        self.insertion_obs_buf[:, 71:75] = quat_mul(self.segmentation_target_rot, quat_conjugate(self.target_rot))

        for i in range(len(self.insertion_obs_buf_stack_frames) - 1):
            self.insertion_obs_buf[:, (i+1) * self.insertion_one_frame_num_obs:(i+2) * self.insertion_one_frame_num_obs] = self.insertion_obs_buf_stack_frames[i]
            self.insertion_obs_buf_stack_frames[i] = self.insertion_obs_buf[:, (i) * self.insertion_one_frame_num_obs:(i+1) * self.insertion_one_frame_num_obs].clone()

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats_x = torch_rand_float(-1, 1, (len(env_ids), 4), device=self.device)
        rand_floats_y = torch_rand_float(-1, 1, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats_x[:, 0], rand_floats_y[:, 1],
                                     self.x_unit_tensor[env_ids],
                                     self.y_unit_tensor[env_ids])

        if apply_reset:
            self.object_pose_for_open_loop[env_ids] = self.goal_states[env_ids, 0:7]

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        # if not apply_reset:
        self.goal_states[env_ids, 3:7] = new_rot
        # self.goal_states[env_ids, 3:7] = self.goal_init_state[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            self.object_pose_for_open_loop[env_ids] = self.goal_states[env_ids, 0:7].clone()
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def put_valid_lego(self, lego_index, pos, rotx, rotz, rand_floats):    
        self.root_state_tensor[lego_index, 0] = pos[0]
        self.root_state_tensor[lego_index, 1] = pos[1]
        self.root_state_tensor[lego_index, 2] = pos[2] + rand_floats[:, 2] * 0.1
        quat = gymapi.Quat().from_euler_zyx(0.0, 0.0, 0)
        self.root_state_tensor[lego_index, 7:13] = 0

    def replay_rollout(self, root_state_tensor, dof_state, i=0):
        # if i >= 595:
        #     root_state_tensor[self.insert_lego_indices[0][0], 3:7] = root_state_tensor[self.inserted_lego_indices[0][1], 3:7].clone()
        #     root_state_tensor[self.inserted_lego_indices[0][0], 3:7] = root_state_tensor[self.inserted_lego_indices[0][1], 3:7].clone()
        #     root_state_tensor[self.insert_lego_indices[0][0], 0:3] = root_state_tensor[self.inserted_lego_indices[0][1], 0:3].clone()
        #     root_state_tensor[self.insert_lego_indices[0][0], 1] -= 0.06
        #     root_state_tensor[self.inserted_lego_indices[0][0], 0:3] = root_state_tensor[self.inserted_lego_indices[0][1], 0:3].clone()
        #     root_state_tensor[self.inserted_lego_indices[0][0], 1] -= 0.06

        # dof_state.view(self.num_envs, -1, 2)[:, :self.num_arm_hand_dofs][0, 10:14, 0] += 0.15

        self.gym.set_actor_root_state_tensor(self.sim,
                                        gymtorch.unwrap_tensor(root_state_tensor))

        self.gym.set_dof_state_tensor(self.sim,
                                            gymtorch.unwrap_tensor(dof_state))


        if self.record_camera_view_image:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
        
            camera_rgba_image = self.camera_rgb_visulization(self.camera_tensors, env_id=0, is_depth_image=False)
            hi_camera_rgba_image = self.camera_rgb_visulization(self.high_res_camera_tensors, env_id=0, is_depth_image=False)
            attach_camera_rgba_image = self.camera_rgb_visulization(self.attach_camera_tensors, env_id=0, is_depth_image=False)
            
            #
            self.seg_idx = 0

            attach_camera_seg_image = self.camera_segmentation_visulization(self.attach_camera_tensors, self.attach_camera_seg_tensors, segmentation_id=self.seg_idx + 1, env_id=0, is_depth_image=False)
            self.segmentation_object_point_list = torch.nonzero(torch.where(self.attach_camera_seg_tensors[0] == self.seg_idx + 1, self.attach_camera_seg_tensors[0], torch.zeros_like(self.attach_camera_seg_tensors[0])))
            self.segmentation_object_point_list = self.segmentation_object_point_list.float()            
            self.segmentation_object_point_num = self.segmentation_object_point_list.shape[0]

            # print(self.segmentation_object_point_num)
            axis1 = quat_apply(self.root_state_tensor[self.lego_indices.view(-1)[self.seg_idx], 3:7], self.z_unit_tensor)
            axis2 = self.z_unit_tensor
            dot1 = torch.bmm(axis1.view(self.num_envs, 1, 3), axis2.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
            self.lego_z_align_reward = (torch.sign(dot1) * dot1 ** 2)
            # print()

            # cv2.putText(attach_camera_rgba_image, "{}".format(self.segmentation_object_point_num), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            # cv2.putText(attach_camera_rgba_image, "{}".format(self.lego_z_align_reward[0]), (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            attach_camera_seg_image[self.attach_camera_seg_tensors[0].cpu() == self.seg_idx + 1] = 255

            attach_camera_seg_image_mask = attach_camera_seg_image.copy()
            attach_camera_seg_image_mask[self.attach_camera_seg_tensors[0].cpu() != self.seg_idx + 1] = 128

            concat_frame = cv2.hconcat([attach_camera_seg_image_mask, attach_camera_seg_image, attach_camera_rgba_image])
            self.out.write(concat_frame)        
                    
            cv2.imshow('frame',concat_frame)
            cv2.waitKey(1)
            self.gym.end_access_image_tensors(self.sim)

        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

    # default robot pose: [0.00, 0.782, -1.087, 3.487, 2.109, -1.415]
    def reset_idx(self, env_ids, goal_env_ids, first_lego=True):
        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_hand_dofs * 2 + 5), device=self.device)
        lego_init_rand_floats = torch_rand_float(-1.0, 1.0, (self.num_envs * (156), 3), device=self.device)
        lego_init_rand_floats[(156-80):, :] = 0

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        self.root_state_tensor[self.object_indices[env_ids], 3:7] = self.object_init_state[env_ids, 3:7].clone()
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])
        self.object_pose_for_open_loop[env_ids] = self.root_state_tensor[self.object_indices[env_ids], 0:7].clone()

        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 0:7] = self.lego_init_states[env_ids].view(-1, 13)[:, 0:7].clone()
        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 7:13] = torch.zeros_like(self.root_state_tensor[self.lego_indices[env_ids].view(-1), 7:13])
        
        # lego_init_rand_floats[0:72, 1] += 50
        # lego_init_rand_floats[24:48, 1] += 50

        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 0:1] = self.lego_init_states[env_ids].view(-1, 13)[:, 0:1].clone() + lego_init_rand_floats[:, 0:1] * 0.02
        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 1:2] = self.lego_init_states[env_ids].view(-1, 13)[:, 1:2].clone() + lego_init_rand_floats[:, 1:2] * 0.02
        
        self.set_valid_lego = True

        if self.set_valid_lego:         
            # shoe
            # self.put_valid_lego(self.lego_indices[0][0], [0.28, 0.19, 1.1], rotx=1.571, rotz=-1.571, rand_floats=rand_floats)
            # self.put_valid_lego(self.lego_indices[0][1], [0.22, 0.19, 1.1], rotx=1.571, rotz=-1.571, rand_floats=rand_floats)
            # self.put_valid_lego(self.lego_indices[0][2], [0.22, 0.19, 1.1], rotx=1.571, rotz=-1.571, rand_floats=rand_floats)
            # self.put_valid_lego(self.lego_indices[0][3], [0.25, 0.24, 1.1], rotx=1.571, rotz=-1.571, rand_floats=rand_floats)
            # self.put_valid_lego(self.lego_indices[0][4], [0.25, 0.14, 1.1], rotx=1.571, rotz=-1.571, rand_floats=rand_floats)
            # self.put_valid_lego(self.lego_indices[0][5], [0.28, 0.24, 1.1], rotx=1.571, rotz=-1.571, rand_floats=rand_floats)
            # self.put_valid_lego(self.lego_indices[0][6], [0.28, 0.14, 1.1], rotx=1.571, rotz=-1.571, rand_floats=rand_floats)
            self.put_valid_lego(self.lego_indices[0][7], [0.20, 0.24, 1.1], rotx=1.571, rotz=-1.571, rand_floats=rand_floats)

        self.root_state_tensor[self.insert_lego_indices[env_ids].view(-1), 0:7] = self.insert_lego_init_states[env_ids].view(-1, 13)[:, 0:7].clone()
        self.root_state_tensor[self.insert_lego_indices[env_ids].view(-1), 7:13] = torch.zeros_like(self.root_state_tensor[self.insert_lego_indices[env_ids].view(-1), 7:13])

        quat = gymapi.Quat().from_euler_zyx(0.0, 0.0, 0.0)
        self.root_state_tensor[self.extra_object_indices[env_ids], 0] = 0.25
        self.root_state_tensor[self.extra_object_indices[env_ids], 1] = -0.2
        self.root_state_tensor[self.extra_object_indices[env_ids], 2] = 0.618
        self.root_state_tensor[self.extra_object_indices[env_ids], 3] = quat.x
        self.root_state_tensor[self.extra_object_indices[env_ids], 4] = quat.y
        self.root_state_tensor[self.extra_object_indices[env_ids], 5] = quat.z
        self.root_state_tensor[self.extra_object_indices[env_ids], 6] = quat.w
        self.root_state_tensor[self.extra_object_indices[env_ids], 7:13] = 0

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids],
                                                 self.extra_object_indices[env_ids],
                                                 self.insert_lego_indices[env_ids].view(-1),
                                                 self.inserted_lego_indices[env_ids].view(-1),
                                                 self.lego_indices[env_ids].view(-1)]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                    * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        self.arm_hand_prepare_dof_poses[:, :] = self.arm_hand_prepare_dof_pos_list[0]
        self.end_effector_rotation[:, :] = self.end_effector_rot_list[0]

        self.arm_hand_dof_pos[env_ids, 0:23] = self.arm_hand_prepare_dof_poses
        self.arm_hand_dof_vel[env_ids, :] = self.arm_hand_dof_default_vel #+ \
        self.prev_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_prepare_dof_poses
        self.cur_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_prepare_dof_poses

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(self.dof_state),
        #                                       gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        self.meta_rew_buf[env_ids] = 0

        for i in range(self.grasping_stack_obs):
            self.grasping_obs_buf_stack_frames[i] = torch.zeros_like(self.grasping_obs_buf_stack_frames[i])
            self.grasping_state_buf_stack_frames[i] = torch.zeros_like(self.grasping_state_buf_stack_frames[i])

        self.force_reset = False
        if first_lego:
            self.post_reset(env_ids, hand_indices)

    def searching_post_reset(self, env_ids, hand_indices, fall_lego=True):
        self.arm_hand_prepare_dof_poses[:, :] = self.arm_hand_prepare_dof_pos_list[0]
        self.end_effector_rotation[:, :] = self.end_effector_rot_list[0]

        if fall_lego:
            for i in range(100):
                self.gym.refresh_dof_state_tensor(self.sim)
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self.gym.refresh_jacobian_tensors(self.sim)

                if self.record_camera_view_image:
                    self.gym.render_all_camera_sensors(self.sim)
                    self.gym.start_access_image_tensors(self.sim)
                
                    camera_rgba_image = self.camera_rgb_visulization(self.camera_tensors, env_id=0, is_depth_image=False)
                    hi_camera_rgba_image = self.camera_rgb_visulization(self.high_res_camera_tensors, env_id=0, is_depth_image=False)

                    cv2.imshow("DEBUG_RGB_VIS", camera_rgba_image)
                    cv2.imshow("DEBUG_HI_RGB_VIS", hi_camera_rgba_image)
                    cv2.waitKey(1)

                    self.gym.end_access_image_tensors(self.sim)

                self.render()
                self.gym.simulate(self.sim)

        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_jacobian_tensors(self.sim)
        # self.gym.render_all_camera_sensors(self.sim)
        # self.gym.start_access_image_tensors(self.sim)

        # self.lego_segmentation_indices[0] = self.lego_indices[0][0]

        # rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_hand_dofs * 2 + 5), device=self.device)
        # quat = gymapi.Quat().from_euler_zyx(rand_floats[0, 0] * 1.57, 0.0, rand_floats[0, 2] * 3.14)
        # self.root_state_tensor[self.lego_segmentation_indices[env_ids], 0] = 0.25 + rand_floats[env_ids, 0] * 0.0
        # self.root_state_tensor[self.lego_segmentation_indices[env_ids], 1] = 0.19 + rand_floats[env_ids, 0] * 0.0
        # self.root_state_tensor[self.lego_segmentation_indices[env_ids], 2] = 0.9
        # self.root_state_tensor[self.lego_segmentation_indices[env_ids], 3] = quat.x
        # self.root_state_tensor[self.lego_segmentation_indices[env_ids], 4] = quat.y
        # self.root_state_tensor[self.lego_segmentation_indices[env_ids], 5] = quat.z
        # self.root_state_tensor[self.lego_segmentation_indices[env_ids], 6] = quat.w
        
        # object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
        #                                          self.goal_object_indices[env_ids],
        #                                          self.lego_indices[env_ids].view(-1)]).to(torch.int32))
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.root_state_tensor),
        #                                              gymtorch.unwrap_tensor(object_indices), len(object_indices))
        # for i in range(20):
        #     self.render()
        #     self.gym.simulate(self.sim)

        self.render_for_camera()
        self.gym.fetch_results(self.sim, True)

        if self.enable_camera_sensors:
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)

            if self.record_camera_view_image:
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)

                camera_rgba_image = self.camera_rgb_visulization(self.camera_tensors, env_id=0, is_depth_image=False)
                hi_camera_rgba_image = self.camera_rgb_visulization(self.high_res_camera_tensors, env_id=0, is_depth_image=False)

                cv2.imshow("DEBUG_RGB_VIS", camera_rgba_image)
                cv2.imshow("DEBUG_HI_RGB_VIS", hi_camera_rgba_image)
                cv2.waitKey(1)

                self.gym.end_access_image_tensors(self.sim)

            # self.compute_emergence_reward(self.camera_tensors, self.camera_seg_tensors, segmentation_id=self.segmentation_id)
            # for i in range(self.num_envs):
            #     torch_seg_tensor = self.camera_tensors[i]
            #     self.last_emergence_pixel[i] = torch_seg_tensor[torch_seg_tensor == self.segmentation_id].shape[0]

            self.last_all_lego_brick_pos = self.root_state_tensor[self.lego_indices[:], 0:3].clone()
            
            self.hand_pos_history = torch.zeros_like(self.hand_pos_history)
            self.hand_pos_history_0 = torch.mean(self.hand_pos_history[:, 0*self.hand_reset_step:1*self.hand_reset_step, :], dim=1, keepdim=False)
            self.hand_pos_history_1 = torch.mean(self.hand_pos_history[:, 1*self.hand_reset_step:2*self.hand_reset_step, :], dim=1, keepdim=False)
            self.hand_pos_history_2 = torch.mean(self.hand_pos_history[:, 2*self.hand_reset_step:3*self.hand_reset_step, :], dim=1, keepdim=False)
            self.hand_pos_history_3 = torch.mean(self.hand_pos_history[:, 3*self.hand_reset_step:4*self.hand_reset_step, :], dim=1, keepdim=False)
            self.hand_pos_history_4 = torch.mean(self.hand_pos_history[:, 4*self.hand_reset_step:5*self.hand_reset_step, :], dim=1, keepdim=False)
            self.hand_pos_history_5 = torch.mean(self.hand_pos_history[:, 5*self.hand_reset_step:6*self.hand_reset_step, :], dim=1, keepdim=False)
            self.hand_pos_history_6 = torch.mean(self.hand_pos_history[:, 6*self.hand_reset_step:7*self.hand_reset_step, :], dim=1, keepdim=False)
            self.hand_pos_history_7 = torch.mean(self.hand_pos_history[:, 7*self.hand_reset_step:8*self.hand_reset_step, :], dim=1, keepdim=False)
            # self.camera_rgbd_image_tensors = torch.stack(self.camera_tensors, dim=0).view(self.num_envs, -1)
            # self.camera_seg_image_tensors = ((torch.stack(self.camera_seg_tensors, dim=0) == self.segmentation_id) * 255).view(self.num_envs, -1)
            # cv2.namedWindow("DEBUG_RGB_VIS", 0)
            # cv2.namedWindow("DEBUG_SEG_VIS", 0)

            # cv2.imshow("DEBUG_RGB_VIS", camera_rgba_image)
            # cv2.imshow("DEBUG_SEG_VIS", camera_seg_image)
            # cv2.waitKey(1)

            self.all_lego_brick_pos = self.root_state_tensor[self.lego_indices[:].view(-1), 0:3].clone().view(self.num_envs, -1, 3)
            self.init_heap_movement_penalty = torch.where(abs(self.all_lego_brick_pos[:self.num_envs, :, 0] - 1) > 0.25,
                                                torch.where(abs(self.all_lego_brick_pos[:self.num_envs, :, 1]) > 0.35, torch.ones_like(self.all_lego_brick_pos[:self.num_envs, :, 0]), torch.zeros_like(self.all_lego_brick_pos[:self.num_envs, :, 0])), torch.zeros_like(self.all_lego_brick_pos[:self.num_envs, :, 0]))
            
            self.init_heap_movement_penalty = torch.sum(self.init_heap_movement_penalty, dim=1, keepdim=False)

        if fall_lego:
            self.arm_hand_dof_pos[env_ids, 0:23] = self.arm_hand_prepare_dof_poses
            self.prev_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_prepare_dof_poses
            self.cur_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_prepare_dof_poses

            # remove lego outsize the box
            for i in self.lego_indices[env_ids].view(-1):
                if self.root_state_tensor[i, 0:1] < -0.05 or self.root_state_tensor[i, 0:1] > 0.5 or self.root_state_tensor[i, 1:2] < 0.0 or self.root_state_tensor[i, 1:2] > 0.38:
                    self.root_state_tensor[i, 0:1] += 100
        
            self.gym.set_actor_root_state_tensor(self.sim,
                                            gymtorch.unwrap_tensor(self.root_state_tensor))
            
            self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.prev_targets),
                                                            gymtorch.unwrap_tensor(hand_indices), len(env_ids))

            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.segmentation_target_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3]
        self.segmentation_target_rot = self.root_state_tensor[self.lego_segmentation_indices, 3:7]

        print("searching_post_reset finish")

    def run_searching_policy(self):
        self.visualize_fingertip(color=[1, 0.4, 0.4])
        self.set_dof_effort(self.envs[0], self.hand_indices[0], 5, 70)
        # self.set_dof_effort(self.envs[0], self.hand_indices[0], 0.7, 20)

        self.progress_buf = torch.ones_like(self.progress_buf) * 101

        for i in range(self.stack_obs):
            self.obs_buf_stack_frames[i] = torch.zeros_like(self.obs_buf_stack_frames[i])
            self.state_buf_stack_frames[i] = torch.zeros_like(self.state_buf_stack_frames[i])

        self.dig_pos = []
        self.success_count = 0
        
        while self.progress_buf[0] < 241:
            search_action = self.search_policy.predict(observation=torch.clamp(self.obs_buf, -5, 5).to(self.device), deterministic=True)
            self.step(search_action)

            # self.dig_pos.append(self.arm_hand_dof_pos.cpu().numpy())
            if self.lego_z_align_reward > 0.8:
                if self.success_count == 10:
                    break
                else:
                    self.success_count += 1
        print("searching finished")

        # with open("/home/hp-3070/视频/lego_demo/qpos/dig_pos_{}.pkl".format(self.segmentation_id), "wb") as f:
        #     pickle.dump(self.dig_pos, f)

        # exit()
        # self.set_dof_effort(self.envs[0], self.hand_indices[0], 0.7, 20)

    def run_spin_policy(self):
        self.visualize_fingertip(color=[0.7, 1., 0.4])

        self.progress_buf = torch.ones_like(self.progress_buf) * 241
        self.actions = torch.zeros((self.num_envs, 23), device=self.device, dtype=torch.float)

        for i in range(self.spin_stack_obs):
            self.spin_obs_buf_stack_frames[i] = torch.zeros_like(self.spin_obs_buf_stack_frames[i])
            self.spin_state_buf_stack_frames[i] = torch.zeros_like(self.spin_state_buf_stack_frames[i])

        # record spin qpos
        self.spin_qpos = []
        self.success_count = 0
        # self.set_dof_effort(self.envs[0], self.hand_indices[0], 0.7, 20)

        while self.progress_buf[0] < 361:
            spin_action = self.spin_policy.predict(observation=torch.clamp(self.spin_obs_buf, -5, 5).to(self.device), deterministic=True)
            self.step(spin_action)

            # self.spin_qpos.append(self.arm_hand_dof_pos.cpu().numpy())

            if self.lego_z_align_reward > 0.8:
                if self.success_count == 30:
                    break
                else:
                    self.success_count += 1

            if abs(self.segmentation_target_pos[0, 2] - self.segmentation_target_init_pos[0, 2]) > 0.1:
                    self.progress_buf[0] = 361


        # self.set_dof_effort(self.envs[0], self.hand_indices[0], 0.7, 20)

        # with open("/home/hp-3070/视频/lego_demo/qpos/spin_pos_{}.pkl".format(self.segmentation_id), "wb") as f:
        #     pickle.dump(self.spin_qpos, f)

        # exit()

    def grasping_post_reset(self, env_ids, hand_indices, policy=None):
        # grasping post reset
        self.visualize_fingertip(color=[0.8, 0.8, 0.8])
        self.segmentation_target_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3]

        # self.arm_hand_dof_pos[env_ids, 0:23] = self.arm_hand_prepare_dof_poses
        # self.prev_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_prepare_dof_poses
        # self.cur_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_prepare_dof_poses

        # self.gym.set_dof_position_target_tensor_indexed(self.sim,
        #                                                 gymtorch.unwrap_tensor(self.prev_targets),
        #                                                 gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(self.dof_state),
        #                                       gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        # save_index = self.lego_segmentation_indices.clone()
        # self.lego_segmentation_indices[0]=1+8*8
        # self.change_inserted_lego_mesh(env_ids=env_ids, lego_type=1)

        for i in range(2):
            self.render()
            self.gym.simulate(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        
        # self.lego_segmentation_indices = save_index
        self.segmentation_target_init_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3].clone()
        self.segmentation_target_init_rot = self.root_state_tensor[self.lego_segmentation_indices, 3:7].clone()

        # pre-grasp
        self.cur_targets_clone = self.cur_targets[:, :].clone()
        self.set_collision_filter(self.envs[0], self.hand_indices[0], 1)
        self.set_collision_filter(self.envs[0], self.lego_segmentation_indices[0], 1)

        pre_grasp_frame = 120
        if policy == "insert":
            pre_grasp_frame = 120

        for i in range(pre_grasp_frame):
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)

            if self.record_camera_view_image:
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                camera_rgba_image = self.camera_rgb_visulization(self.camera_tensors, env_id=0, is_depth_image=False)
                hi_camera_rgba_image = self.camera_rgb_visulization(self.high_res_camera_tensors, env_id=0, is_depth_image=False)
                cv2.imshow("DEBUG_RGB_VIS", camera_rgba_image)
                cv2.imshow("DEBUG_HI_RGB_VIS", hi_camera_rgba_image)
                cv2.waitKey(1)
                self.gym.end_access_image_tensors(self.sim)

            if self.record_inference_video:
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                hi_camera_rgba_image = self.camera_rgb_visulization(self.high_res_camera_tensors, env_id=0, is_depth_image=False)
                self.out.write(hi_camera_rgba_image)    
                cv2.imshow('Frame', hi_camera_rgba_image)
                cv2.waitKey(1)
                self.gym.end_access_image_tensors(self.sim)

            self.segmentation_target_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3]

            pos_err = (self.segmentation_target_pos - self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3])
            
            if policy == "spin":
                pos_err[:, 2] += 0.25
                if i < 60:
                    pos_err[:, 2] += 0.2

                pos_err[:, 0] -= 0.18
            elif policy == "dig":
                pos_err[:, 2] += 0.24
                if i < 60:
                    pos_err[:, 2] += 0.2

                pos_err[:, 0] -= 0.20
            else:
                # pos_err[:, 1]  -= 0.02
                pos_err[:, 2] += 0.24

                if i < 60:
                    pos_err[:, 2] += 0.2

                pos_err[:, 0] -= 0.18

            target_rot = quat_from_euler_xyz(self.target_euler[:, 0], self.target_euler[:, 1], self.target_euler[:, 2])
            rot_err = orientation_error(target_rot, self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7].clone())

            dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
            delta = control_ik(self.jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :7], self.device, dpose, self.num_envs)
            targets = self.arm_hand_dof_pos[:, 0:7] + delta[:, :7]

            self.arm_hand_dof_pos[env_ids, 0:7] = targets[env_ids, :7]
            self.prev_targets[env_ids, :7] = targets[env_ids, :7]
            self.cur_targets[env_ids, :7] = targets[env_ids, :7]
            self.arm_hand_dof_vel[env_ids, :] = self.arm_hand_dof_default_vel 

            self.prev_targets[env_ids, 7:self.num_arm_hand_dofs] = (scale(torch.tensor([0, 0, -1, 0.5, 1, 0, -1, 0.5, 0, 0, -1, 0.5, 0, 0, -1, 0.5], dtype=torch.float, device=self.device), 
                                            self.arm_hand_dof_lower_limits[7:23], self.arm_hand_dof_upper_limits[7:23]) - self.arm_hand_dof_pos[:, 7:23]) * (i / 60) + self.arm_hand_dof_pos[:, 7:23]
            self.cur_targets[env_ids, 7:self.num_arm_hand_dofs] = (scale(torch.tensor([0, 0, -1, 0.5, 1, 0, -1, 0.5, 0, 0, -1, 0.5, 0, 0, -1, 0.5], dtype=torch.float, device=self.device), 
                                            self.arm_hand_dof_lower_limits[7:23], self.arm_hand_dof_upper_limits[7:23]) - self.arm_hand_dof_pos[:, 7:23]) * (i / 60) + self.arm_hand_dof_pos[:, 7:23]

            if 41 >= i > 40:
                self.set_collision_filter(self.envs[0], self.hand_indices[0], 0)
                self.set_collision_filter(self.envs[0], self.lego_segmentation_indices[0], 0)

            #     self.prev_targets[env_ids, 7:self.num_arm_hand_dofs] -= 0.02
            #     self.cur_targets[env_ids, 7:self.num_arm_hand_dofs] -= 0.02
            # else:
            #     self.prev_targets[env_ids, 7:self.num_arm_hand_dofs] -= 0.02
            #     self.cur_targets[env_ids, 7:self.num_arm_hand_dofs] -= 0.02

            # if policy == "insert":
            #     self.arm_hand_prepare_dof_poses[:, :] = self.arm_hand_prepare_dof_pos_list[0]
            #     self.end_effector_rotation[:, :] = self.end_effector_rot_list[0]

            #     self.arm_hand_dof_pos[env_ids, 0:23] = self.arm_hand_prepare_dof_poses[0]
            #     self.prev_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_prepare_dof_poses[0]

            self.cur_targets[:, :] = self.prev_targets[:, :].clone()

            # self.cur_targets[:, :] = self.cur_targets_clone - 0.01

            self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.cur_targets),
                                                            gymtorch.unwrap_tensor(hand_indices), len(env_ids))

            self.render()
            self.gym.simulate(self.sim)

        # remove lego outsize the box
        for i in self.lego_indices[env_ids].view(-1):
            if self.root_state_tensor[i, 0:1] < -0.05 or self.root_state_tensor[i, 0:1] > 0.5 or self.root_state_tensor[i, 1:2] < 0.0 or self.root_state_tensor[i, 1:2] > 0.4:
                self.root_state_tensor[i, 0:1] += 100

        self.gym.set_actor_root_state_tensor(self.sim,
                                        gymtorch.unwrap_tensor(self.root_state_tensor))

        self.segmentation_target_init_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3].clone()
        self.segmentation_target_init_rot = self.root_state_tensor[self.lego_segmentation_indices, 3:7].clone()

        print("grasping_post_reset finish")

    def run_grasping_policy(self):
        self.visualize_fingertip(color=[0.6, 0.3, 1])
        self.progress_buf = torch.ones_like(self.progress_buf) * 361
        self.actions = torch.zeros((self.num_envs, 23), device=self.device, dtype=torch.float)

        sim_params = self.gym.get_sim_params(self.sim)
        sim_params.physx.max_depenetration_velocity = 1000.0
        self.gym.set_sim_params(self.sim, sim_params)
        # for i in range(self.grasping_stack_obs):
        #     self.grasping_obs_buf_stack_frames[i] = torch.zeros_like(self.grasping_obs_buf_stack_frames[i])
        #     self.grasping_state_buf_stack_frames[i] = torch.zeros_like(self.grasping_state_buf_stack_frames[i])

        self.grasp_qpos = []
        self.set_dof_effort(self.envs[0], self.hand_indices[0], 5, 50)

        while self.progress_buf[0] < 536:
            grasp_action = self.grasp_policy.predict(observation=torch.clamp(self.grasping_obs_buf, -5, 5).to(self.device), deterministic=True)
            self.step(grasp_action)

            # if self.segmentation_target_pos[0, 2] > 1.05:
            if self.arm_hand_finger_dist[0] > 0.6:
                if self.progress_buf[0] >= 436:
                    self.progress_buf[0] = 536

            # print("progress:", self.progress_buf[0])
            if self.segmentation_target_pos[0, 2] > 0.9 or abs(self.segmentation_target_pos[0, 0] - 0.25) > 0.25 or abs(self.segmentation_target_pos[0, 1] - 0.19) > 0.19:
                if self.progress_buf[0] <= 436:
                    self.progress_buf[0] = 436

            if self.progress_buf[0] == 461:
                for i in self.lego_indices[0].view(-1):
                    if self.root_state_tensor[i, 2] > 0.9:
                        self.root_state_tensor[i, 0:1] += 100

        # self.grasp_qpos.append(self.arm_hand_dof_pos.cpu().numpy())
        sim_params = self.gym.get_sim_params(self.sim)
        sim_params.physx.max_depenetration_velocity = 1.0
        self.gym.set_sim_params(self.sim, sim_params)

        # with open("/home/hp-3070/视频/lego_demo/qpos/grasp_pos_{}.pkl".format(self.segmentation_id), "wb") as f:
        #     pickle.dump(self.grasp_qpos, f)

        # exit()

    def run_insertion_policy(self):
        sim_params = self.gym.get_sim_params(self.sim)
        sim_params.physx.max_depenetration_velocity = 1000.0
        self.gym.set_sim_params(self.sim, sim_params)

        self.visualize_fingertip(color=[0.3, 1, 1])

        self.progress_buf = torch.ones_like(self.progress_buf) * 536
        # self.actions = torch.zeros_like(self.actions)
        # for i in range(self.insertion_stack_obs):
        #     self.insertion_obs_buf_stack_frames[i] = torch.zeros_like(self.insertion_obs_buf_stack_frames[i])
        #     self.insertion_state_buf_stack_frames[i] = torch.zeros_like(self.insertion_state_buf_stack_frames[i])

        self.insert_qpos = []
        self.insert_success_count = 0
        env_ids = [0]

        while self.progress_buf[0] < 636:

            insert_action = self.insert_policy.predict(observation=torch.clamp(self.insertion_obs_buf, -5, 5).to(self.device), deterministic=True)
            self.step(insert_action)

            # self.insert_qpos.append(self.arm_hand_dof_pos.cpu().numpy())
            if self.arm_hand_finger_dist[0] > 0.6:
                    self.progress_buf[0] = 636

            if self.is_insertion_success([0]):
                if self.insert_success_count >= 10:
                    self.progress_buf[0] = 636
                else:
                    self.insert_success_count += 1

                print(self.insert_success_count)
                # hand_indices = self.hand_indices[env_ids].to(torch.int32)
                # self.cur_targets_clone = self.cur_targets[:, :].clone()
                # self.set_collision_filter(self.envs[0], self.hand_indices[0], 1)
                # self.set_collision_filter(self.envs[0], self.lego_segmentation_indices[0], 1)
                # for i in range(50):
                #     self.gym.refresh_dof_state_tensor(self.sim)
                #     self.gym.refresh_actor_root_state_tensor(self.sim)
                #     self.gym.refresh_rigid_body_state_tensor(self.sim)
                #     self.gym.refresh_jacobian_tensors(self.sim)

                #     if self.record_camera_view_image:
                #         self.gym.render_all_camera_sensors(self.sim)
                #         self.gym.start_access_image_tensors(self.sim)

                #         camera_rgba_image = self.camera_rgb_visulization(self.camera_tensors, env_id=0, is_depth_image=False)
                #         hi_camera_rgba_image = self.camera_rgb_visulization(self.high_res_camera_tensors, env_id=0, is_depth_image=False)

                #         cv2.imshow("DEBUG_RGB_VIS", camera_rgba_image)
                #         cv2.imshow("DEBUG_HI_RGB_VIS", hi_camera_rgba_image)
                #         cv2.waitKey(1)

                #         self.gym.end_access_image_tensors(self.sim)

                #     # self.segmentation_target_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3]

                #     # pos_err = (self.segmentation_target_pos - self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3])
                #     # pos_err[:, 2] += 0.24 + 0.2
                #     # pos_err[:, 0] -= 0.18

                #     # target_rot = quat_from_euler_xyz(self.target_euler[:, 0], self.target_euler[:, 1], self.target_euler[:, 2])
                #     # rot_err = orientation_error(target_rot, self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7].clone())

                #     # dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
                #     # delta = control_ik(self.jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :7], self.device, dpose, self.num_envs)
                #     # targets = self.arm_hand_dof_pos[:, 0:7] + delta[:, :7]

                #     # self.arm_hand_dof_pos[env_ids, 0:7] = targets[env_ids, :7]
                #     # self.prev_targets[env_ids, :7] = targets[env_ids, :7]
                #     # self.cur_targets[env_ids, :7] = targets[env_ids, :7]
                #     # self.arm_hand_dof_vel[env_ids, :] = self.arm_hand_dof_default_vel 

                #     # self.prev_targets[env_ids, 7:self.num_arm_hand_dofs] = (scale(torch.tensor([0, 0, -1, 0.5, 1, 0, -1, 0.5, 0, 0, -1, 0.5, 0, 0, -1, 0.5], dtype=torch.float, device=self.device), 
                #     #                                 self.arm_hand_dof_lower_limits[7:23], self.arm_hand_dof_upper_limits[7:23]) - self.arm_hand_dof_pos[:, 7:23]) * (i / 60) + self.arm_hand_dof_pos[:, 7:23]
                #     # self.cur_targets[env_ids, 7:self.num_arm_hand_dofs] = (scale(torch.tensor([0, 0, -1, 0.5, 1, 0, -1, 0.5, 0, 0, -1, 0.5, 0, 0, -1, 0.5], dtype=torch.float, device=self.device), 
                #     #                                 self.arm_hand_dof_lower_limits[7:23], self.arm_hand_dof_upper_limits[7:23]) - self.arm_hand_dof_pos[:, 7:23]) * (i / 60) + self.arm_hand_dof_pos[:, 7:23]

                #     self.cur_targets[:, :] = self.arm_hand_prepare_dof_poses
                #     self.prev_targets[:, :] = self.arm_hand_prepare_dof_poses


                #     self.gym.set_dof_position_target_tensor_indexed(self.sim,
                #                                                     gymtorch.unwrap_tensor(self.cur_targets),
                #                                                     gymtorch.unwrap_tensor(hand_indices), len(env_ids))

                #     self.render()
                #     self.gym.simulate(self.sim)

                # break
        self.set_collision_filter(self.envs[0], self.hand_indices[0], 0)
        self.set_collision_filter(self.envs[0], self.lego_segmentation_indices[0], 0)
        # with open("/home/hp-3070/视频/lego_demo/qpos/insert_pos_{}.pkl".format(self.segmentation_id - 108), "wb") as f:
        #     pickle.dump(self.insert_qpos, f)

        # exit()

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.extra_object_indices[env_ids],
                                                 self.insert_lego_indices[env_ids].view(-1),
                                                 self.lego_indices[env_ids].view(-1)]).to(torch.int32))
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_state_tensor[self.lego_segmentation_indices, 7:13] = 0
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

    def set_target_lego_index(self, lego_type):
        # judge grasp which one
        # ['2x2.urdf', '1x4.urdf', '2x4.urdf', '1x3_curve_soft.urdf', '2x2_curve.urdf', '1x2.urdf', '2x3_curve.urdf', '2x2_curve_soft.urdf']
        # [[0, 0, 0], [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 1], [1, 0, 0]]
        self.last_lego_type_z = 0.6
        self.last_lego_type_z_idx = 0
        for i in range(9):
            self.lego_type_z = self.root_state_tensor.clone()[self.lego_indices.view(-1)[lego_type + i * 8]][2]
            self.lego_type_y = self.root_state_tensor.clone()[self.lego_indices.view(-1)[lego_type + i * 8]][1]
            self.lego_type_x = self.root_state_tensor.clone()[self.lego_indices.view(-1)[lego_type + i * 8]][0]
            
            if 1.92 > self.lego_type_z.squeeze(0) > self.last_lego_type_z:
                if 0.4 > self.lego_type_y.squeeze(0) > 0.00:
                    if 0.5 > self.lego_type_x.squeeze(0) > 0.0:
                        self.last_lego_type_z = self.lego_type_z
                        self.last_lego_type_z_idx = lego_type + i * 8

        print("now target lego z: {}".format(self.last_lego_type_z))
        print("now target lego index: {}".format(self.last_lego_type_z_idx))

        if self.last_lego_type_z == 0.6:
            print("Can't find valid target lego")
            exit()

        self.segmentation_id = self.last_lego_type_z_idx
        self.lego_segmentation_indices[0] = self.lego_indices.view(-1)[self.last_lego_type_z_idx]
        # self.lego_segmentation_indices[0] = self.lego_indices[0][lego_type]

        print("set_ind:", self.lego_segmentation_indices)

        color = self.color_map[lego_type % 8]
        self.gym.set_rigid_body_color(self.envs[0], self.lego_segmentation_indices[0], 0, gymapi.MESH_VISUAL, gymapi.Vec3(color[0], color[1], color[2]))

    def set_insert_target_pose(self, offset_pos, offset_rot):
        self.target_pos = to_torch([offset_pos[0]+self.last_target_pos[0, 0], offset_pos[1]+self.last_target_pos[0, 1], offset_pos[2]+self.last_target_pos[0, 2]], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
        quat = gymapi.Quat().from_euler_zyx(0, 0, offset_rot * 1.57)
        self.target_rot = to_torch([quat.x, quat.y, quat.z, quat.w], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
        # self.target_rot = self.last_target_rot

        # self.last_target_pos = self.target_pos.clone()
        # self.last_target_rot = self.target_rot.clone()

    def is_grasping_success(self, env_ids):
        if self.segmentation_target_pos[env_ids, 1] < 0:
            if self.arm_hand_finger_dist[env_ids] < 0.6:
                print("grasping_success")
                return True
            
        print("grasping_fail")
        return False

    def change_insert_lego_mesh(self, env_ids, lego_type):
        lego_type = lego_type % 8

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.extra_object_indices[env_ids],
                                                 self.insert_lego_indices[env_ids].view(-1),
                                                 self.inserted_lego_indices[env_ids].view(-1),
                                                 self.lego_indices[env_ids].view(-1)]).to(torch.int32))
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.saved_insert_lego_states = self.root_state_tensor[self.lego_segmentation_indices].clone()

        self.pre_exchange_lego_segmentation_indices = self.lego_segmentation_indices.clone()

        self.segmentation_id = lego_type + len(self.lego_assets)
        self.lego_segmentation_indices[0] = self.insert_lego_indices.view(-1)[lego_type]

        # self.saved_pre_lego_states = self.root_state_tensor[self.lego_segmentation_indices].clone()

        self.root_state_tensor[self.pre_exchange_lego_segmentation_indices, 0] = 1.7
        self.root_state_tensor[self.pre_exchange_lego_segmentation_indices, 2] = 0.3

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        # for _ in range(2):
        #     self.render()
        #     self.gym.simulate(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.segmentation_id = lego_type + len(self.lego_assets)
        self.lego_segmentation_indices[0] = self.insert_lego_indices.view(-1)[lego_type]

        self.root_state_tensor[self.lego_segmentation_indices] = self.saved_insert_lego_states

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        # for _ in range(2):
        #     self.render()
        #     self.gym.simulate(self.sim)

    def change_inserted_lego_mesh(self, env_ids, lego_type):        
        lego_type = lego_type % 8
        if lego_type in self.inserted_lego_type:
            lego_type += 8

        self.inserted_lego_type.append(lego_type)


        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.extra_object_indices[env_ids],
                                                 self.insert_lego_indices[env_ids].view(-1),
                                                 self.inserted_lego_indices[env_ids].view(-1),
                                                 self.lego_indices[env_ids].view(-1)]).to(torch.int32))
        
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.saved_insert_lego_states = self.root_state_tensor[self.lego_segmentation_indices].clone()

        self.pre_exchange_lego_segmentation_indices = self.lego_segmentation_indices.clone()

        self.root_state_tensor[self.pre_exchange_lego_segmentation_indices, 0] = 1.7
        self.root_state_tensor[self.pre_exchange_lego_segmentation_indices, 2] = 0.3

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.lego_segmentation_indices[0] = self.inserted_lego_indices.view(-1)[lego_type]

        self.root_state_tensor[self.lego_segmentation_indices] = self.saved_insert_lego_states

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

    def set_lego_target(self, offset_pos, offset_rot):        
        self.target_pos = to_torch([offset_pos[0]+self.last_target_pos[0, 0], offset_pos[1]+self.last_target_pos[0, 1], offset_pos[2]+self.last_target_pos[0, 2]], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
        quat = gymapi.Quat().from_euler_zyx(0, 0, offset_rot * 1.57)
        self.target_rot = to_torch([quat.x, quat.y, quat.z, quat.w], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)

    def set_lego_target_interface(self, env_ids, lego_type, offset_pos, offset_rot):        
        lego_type = lego_type % 8
        # if lego_type in self.interface_lego_type:
        #     lego_type += 8

        # self.interface_lego_type.append(lego_type)

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.extra_object_indices[env_ids],
                                                 self.insert_lego_indices[env_ids].view(-1),
                                                 self.inserted_lego_indices[env_ids].view(-1),
                                                 self.interface_lego_indices[env_ids].view(-1),
                                                 self.lego_indices[env_ids].view(-1)]).to(torch.int32))

        self.target_pos = to_torch([offset_pos[0]+self.last_target_pos[0, 0], offset_pos[1]+self.last_target_pos[0, 1], offset_pos[2]+self.last_target_pos[0, 2]], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
        quat = gymapi.Quat().from_euler_zyx(0, 0, offset_rot * 1.57)
        self.target_rot = to_torch([quat.x, quat.y, quat.z, quat.w], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
        is_exit = False
        is_change_lego_type = False
        last_time_step = 0

        cam_pos = gymapi.Vec3(0.55, -0.3, 0.9)
        cam_target = gymapi.Vec3(-0.3, -0.1, 0.4)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        for i in range(100000):
            for _ in range(self.control_freq_inv):
                self.render()
                self.gym.simulate(self.sim)

            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)

            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "W":
                    self.target_pos[:, 0] += 0.015 / 2
                if evt.action == "S":
                    self.target_pos[:, 0] -= 0.015 / 2
                if evt.action == "D":
                    self.target_pos[:, 1] += 0.015 / 2
                if evt.action == "A":
                    self.target_pos[:, 1] -= 0.015 / 2
                if evt.action == "Q":
                    self.target_pos[:, 2] += (0.0375 / 2)
                if evt.action == "E":
                    self.target_pos[:, 2] -= (0.0375 / 2)
                if evt.action == "T":
                    offset_rot += 1
                    quat = gymapi.Quat().from_euler_zyx(0, 0, offset_rot * (1.57 / 2))
                    self.target_rot = to_torch([quat.x, quat.y, quat.z, quat.w], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
                if evt.action == "Y":
                    offset_rot -= 1
                    quat = gymapi.Quat().from_euler_zyx(0, 0, offset_rot * (1.57 / 2))
                    self.target_rot = to_torch([quat.x, quat.y, quat.z, quat.w], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)

                if evt.action == "G":
                    is_exit = True

                if evt.action == "H":
                    is_change_lego_type = True

            self.root_state_tensor[self.interface_lego_indices[0, lego_type], 0:3] = self.target_pos
            self.root_state_tensor[self.interface_lego_indices[0, lego_type], 3:7] = self.target_rot

            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.root_state_tensor),
                                                        gymtorch.unwrap_tensor(object_indices), len(object_indices))
            hand_indices = self.hand_indices[env_ids].to(torch.int32)
            self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.prev_targets),
                                                            gymtorch.unwrap_tensor(hand_indices), len(env_ids))

            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(hand_indices), len(env_ids))

            if is_change_lego_type:
                self.root_state_tensor[self.interface_lego_indices[0, lego_type], :] = self.interface_lego_init_states[0, lego_type, :].clone()
                if i - last_time_step >= 10:
                    lego_type += 1
                lego_type %= 8

                self.root_state_tensor[self.interface_lego_indices[0, lego_type], 0:3] = self.target_pos
                self.root_state_tensor[self.interface_lego_indices[0, lego_type], 3:7] = self.target_rot

                self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.root_state_tensor),
                                                            gymtorch.unwrap_tensor(object_indices), len(object_indices))
                
                is_change_lego_type = False
                last_time_step = i

            if is_exit:
                self.root_state_tensor[self.interface_lego_indices[0, lego_type], :] = self.interface_lego_init_states[0, lego_type, :].clone()
        
                self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                            gymtorch.unwrap_tensor(self.root_state_tensor),
                                                            gymtorch.unwrap_tensor(object_indices), len(object_indices))
                cam_pos = gymapi.Vec3(0.7, -0.4, 1.3)
                cam_target = gymapi.Vec3(-0.6, 0.4, 0.2)
                self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

                self.lego_type = lego_type
                print(self.lego_type)
                break

    def is_insertion_success(self, env_ids, lego_type=0):
        quat_diff = quat_mul(self.segmentation_target_rot, quat_conjugate(self.target_rot))
        rot_dist1 = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        self.symmetry_extra_target_rot = quat_mul(self.target_rot, to_torch([0.0, 0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1))
        quat_diff2 = quat_mul(self.segmentation_target_rot, quat_conjugate(self.symmetry_extra_target_rot))
        rot_dist2 = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff2[:, 0:3], p=2, dim=-1), max=1.0))
        rot_dist = torch.min(rot_dist1, rot_dist2)

        if torch.norm(self.target_pos - self.segmentation_target_pos, p=2, dim=-1) < 0.02:
            if rot_dist < 0.2:
                # if self.segmentation_target_pos[0, 2] <= 0.66:
                # if rot_dist < 10.2:
                    print("insertion_success")
                    return True
        print("insertion_fail")
        return False

    def is_spin_success(self, env_ids):
        axis1 = quat_apply(self.root_state_tensor[self.lego_segmentation_indices, 3:7], self.z_unit_tensor)
        axis2 = self.z_unit_tensor
        dot1 = torch.bmm(axis1.view(self.num_envs, 1, 3), axis2.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
        self.lego_z_align_reward = (torch.sign(dot1) * dot1 ** 2)

        print("lego_z_align_reward: ", self.lego_z_align_reward)
        if self.lego_z_align_reward > 0.6:
            print("spin_success")
            return True
        print("spin_fail")
        return False

    def is_search_success(self, env_ids, lego_type=0):
        pos_err = (self.segmentation_target_pos - self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3])

        pos_err_sum = abs(pos_err[0]).sum()
        print("pos_err_sum: ", pos_err_sum)

        if pos_err_sum < 0.03 + 0.24 + 0.18: 
            print("search_success")
            return True
        print("search_fail")
        return False

    def retri_grasp_rollout(self, env_ids, hand_indices, lego_type, offset_pos, offset_rot, fall_lego):
        self.searching_post_reset(env_ids, hand_indices, fall_lego=fall_lego)
        self.lego_type = lego_type
        
        if fall_lego and not self.record_inference_video:
            self.set_lego_target_interface(env_ids, lego_type, offset_pos=offset_pos, offset_rot=offset_rot)
            lego_type = self.lego_type
        elif self.record_inference_video:
            self.set_lego_target(offset_pos=offset_pos, offset_rot=offset_rot)
            self.lego_type = lego_type

        self.set_target_lego_index(lego_type=lego_type)

        # self.set_insert_target_pose(offset_pos=offset_pos, offset_rot=offset_rot)

        # if not self.is_spin_success(env_ids=env_ids):
        #     self.grasping_post_reset(env_ids, hand_indices, policy="dig")
        #     self.run_searching_policy()

        if not self.is_spin_success(env_ids=env_ids):
            # self.grasping_post_reset(env_ids, hand_indices, policy="spin")
            # self.run_spin_policy()
            self.grasping_post_reset(env_ids, hand_indices, policy="insert")

        else:
            self.grasping_post_reset(env_ids, hand_indices, policy="insert")

        self.run_grasping_policy()
        # self.insert_init_align()
        # self.visualize_fingertip(color=[0.3, 1, 1])

    def assemble_one_lego(self, env_ids, hand_indices, lego_type, offset_pos, offset_rot, fall_lego, first_lego=False):
        if not first_lego:
            self.reset_idx(env_ids, [], first_lego=False)

        self.retri_grasp_rollout(env_ids=env_ids, hand_indices=hand_indices, lego_type=lego_type, offset_pos=offset_pos, offset_rot=offset_rot, fall_lego=fall_lego)
        lego_type = self.lego_type

        while not self.is_insertion_success(env_ids=env_ids, lego_type=lego_type):
            while not self.is_grasping_success(env_ids=env_ids):
                self.retri_grasp_rollout(env_ids=env_ids, hand_indices=hand_indices, lego_type=lego_type, offset_pos=offset_pos, offset_rot=offset_rot, fall_lego=False)
                # self.grasping_post_reset(env_ids, hand_indices, policy="insert")
                # self.run_grasping_policy()  
            else:
                self.change_insert_lego_mesh(env_ids=env_ids, lego_type=lego_type)
                self.run_insertion_policy()
                # if not self.is_insertion_success(env_ids=env_ids) and lego_type==2:
                #     self.force_reset = True
                # if not self.is_insertion_success(env_ids=env_ids):
                #     self.change_insert_lego_mesh(env_ids=env_ids, lego_type=lego_type)
        else:
            self.change_inserted_lego_mesh(env_ids=env_ids, lego_type=lego_type)

    def post_reset(self, env_ids, hand_indices):
        # step physics and render each frame
        # ['1x2.urdf', '1x2_curve.urdf', '1x3_curve_soft.urdf', '1x3_curve.urdf', '1x1.urdf', '1x3.urdf', '1x4.urdf', '2x2_curve_soft.urdf']

        # 1 X 2
        # target_rot:0 -> 0, -0.015
        # target_rot:1 -> -0.015, 0
        # target_rot:2 -> 0, 0.015
        # target_rot:3 -> 0.015, 0

        # 1 X 3
        # target_rot:0 -> 0.015, -0.015
        # target_rot:1 -> -0.015, -0.015
        # target_rot:2 -> -0.015, 0.015
        # target_rot:3 -> 0.015, 0.015

        if self.record_inference_video:
            self.assemble_one_lego(env_ids=env_ids, hand_indices=hand_indices, lego_type=self.record_lego_type, offset_pos=[-0.0, -0.015, 0.0375], offset_rot=0, fall_lego=True, first_lego=True)
            exit()

        self.assemble_one_lego(env_ids=env_ids, hand_indices=hand_indices, lego_type=0, offset_pos=[-0.0, -0.015, 0.15], offset_rot=0, fall_lego=True, first_lego=True)
        self.assemble_one_lego(env_ids=env_ids, hand_indices=hand_indices, lego_type=0, offset_pos=[-0.0, -0.015, 0.15], offset_rot=0, fall_lego=True)
        self.assemble_one_lego(env_ids=env_ids, hand_indices=hand_indices, lego_type=0, offset_pos=[-0.0, -0.015, 0.15], offset_rot=0, fall_lego=True)
        self.assemble_one_lego(env_ids=env_ids, hand_indices=hand_indices, lego_type=0, offset_pos=[-0.0, -0.015, 0.15], offset_rot=0, fall_lego=True)
        self.assemble_one_lego(env_ids=env_ids, hand_indices=hand_indices, lego_type=0, offset_pos=[-0.0, -0.015, 0.15], offset_rot=0, fall_lego=True)

    def pre_physics_step(self, actions):
        if self.total_steps != 0:
            self.reset_buf[:] = 0

        if self.force_reset:
            self.reset_buf[:] = 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        ##############################################
        ########           digging               ########
        ##############################################
        if 241 >= self.progress_buf[0] > 101:

            self.act_moving_average = 0.2

            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 7:23],
                                                                    self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                    self.arm_hand_dof_upper_limits[self.actuated_dof_indices])
            
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]

            pos_err = self.segmentation_target_pos - self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3]
            pos_err[:, 2] += 0.24
            pos_err[:, 0] -= 0.20
            # pos_err += self.actions[:, 0:3] * 0.02
            # pos_err[:, 2][insert_pre_0_ids] = 0.2 + 0.22 + (self.segmentation_target_init_pos - self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3])[:, 2][insert_pre_0_ids]

            target_rot = quat_from_euler_xyz(self.target_euler[:, 0], self.target_euler[:, 1], self.target_euler[:, 2])
            rot_err = orientation_error(target_rot, self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7].clone())

            dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
            delta = control_ik(self.jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :7], self.device, dpose, self.num_envs)
            self.cur_targets[:, :7] = self.arm_hand_dof_pos[:, 0:7] + delta[:, :7]

            # self.cur_targets[:, :7][insert_pre_1_ids] = self.arm_hand_insertion_prepare_dof_pos_list[0][:7]
            # self.cur_targets[:, :7][insert_pre_2_ids] = self.arm_hand_insertion_prepare_dof_pos_list[1][:7]
            # self.cur_targets[:, 7:23][insert_pre_1_ids] = self.prev_targets[:, 7:23][insert_pre_1_ids]

        ##############################################
        ########           spin               ########
        ##############################################
        if 361 >= self.progress_buf[0] > 241:

            self.act_moving_average = 1.0

            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 7:23],
                                                                    self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                    self.arm_hand_dof_upper_limits[self.actuated_dof_indices])
            
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]

            # insert_pre_0_ids = [self.progress_buf > 80 + 241]
            # insert_pre_1_ids = [self.progress_buf > 120 + 241]
            insert_pre_0_ids = [self.progress_buf > 120 + 241]

            # pos_err = self.segmentation_target_init_pos - self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3]
            # pos_err[:, 2] += 0.25
            # pos_err[:, 0] -= 0.18
            # # pos_err += self.actions[:, 0:3] * 0.02
            # pos_err[:, 2][insert_pre_0_ids] = 0.2 + 0.22 + (self.segmentation_target_init_pos - self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3])[:, 2][insert_pre_0_ids]
            # pos_err[:, 0][insert_pre_0_ids] = 0
            # pos_err[:, 1][insert_pre_0_ids] = 0

            pos_err = self.actions[:, 0:3] * 0.64
            rot_err = self.actions[:, 3:6] * 0.2

            # target_rot = quat_from_euler_xyz(self.target_euler[:, 0], self.target_euler[:, 1], self.target_euler[:, 2])
            # rot_err = orientation_error(target_rot, self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7].clone())

            dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
            delta = control_ik(self.jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :7], self.device, dpose, self.num_envs)
            self.cur_targets[:, :7] = self.arm_hand_dof_pos[:, 0:7] + delta[:, :7]

            self.cur_targets[:, 7:23][insert_pre_0_ids] = self.prev_targets[:, 7:23][insert_pre_0_ids]

        ##############################################
        ########           grasping           ########
        ##############################################
        if 536 >= self.progress_buf[0] > 361:

            self.act_moving_average = 1.0

            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 7:23],
                                                                    self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                    self.arm_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]

            # print(self.fingertip_handles[0:3])
            # self.cur_targets[:, self.finger_dof_indices] += 0.1

            insert_pre_0_ids = [self.progress_buf > 75 + 361]
            insert_pre_1_ids = [self.progress_buf > 100 + 361]
            insert_pre_2_ids = [self.progress_buf > 125 + 361]

            pos_err = self.actions[:, 0:3] * 0.64
            rot_err = self.actions[:, 3:6] * 0.2
            # pos_err[:, 2][insert_pre_0_ids] = 0.2 + 0.22 + (self.segmentation_target_init_pos - self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3])[:, 2][insert_pre_0_ids]
            # pos_err[:, 0][insert_pre_0_ids] = 0
            # pos_err[:, 1][insert_pre_0_ids] = 0

            # target_rot = quat_from_euler_xyz(self.target_euler[:, 0], self.target_euler[:, 1], self.target_euler[:, 2])
            # rot_err = orientation_error(target_rot, self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7].clone()) * 5

            dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
            delta = control_ik(self.jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :7], self.device, dpose, self.num_envs)
            self.cur_targets[:, :7] = self.arm_hand_dof_pos[:, 0:7] + delta[:, :7]

            self.cur_targets[:, :7][insert_pre_0_ids] = self.arm_hand_insertion_prepare_dof_pos_list[0][:7]
            self.cur_targets[:, :7][insert_pre_1_ids] = self.arm_hand_insertion_prepare_dof_pos_list[1][:7]
            self.cur_targets[:, 7:23][insert_pre_0_ids] = self.prev_targets[:, 7:23][insert_pre_0_ids]
            
            # if self.progress_buf[0] > 125 + 361:
            #     pos_err = self.target_pos - self.root_state_tensor[self.lego_segmentation_indices, 0:3]
            #     pos_err[:, 2] += 0.1
            #     pos_err[:, 0] -= 0.0
            #     target_rot = quat_from_euler_xyz(self.target_euler[:, 0], self.target_euler[:, 1], self.target_euler[:, 2])
            #     rot_err = orientation_error(target_rot, self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7].clone())

            #     dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
            #     delta = control_ik(self.jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :7], self.device, dpose, self.num_envs)
            #     self.cur_targets[:, :7][insert_pre_2_ids] = (self.arm_hand_dof_pos[:, 0:7] + delta[:, :7])[insert_pre_2_ids]

        ##############################################
        ########          insertion           ########
        ##############################################
        if 636 >= self.progress_buf[0] > 536:

            self.act_moving_average = 1.0

            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 7:23],
                                                                    self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                    self.arm_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]

            pos_err = self.actions[:, 0:3] * 0.64
            # rot_err = self.actions[:, 3:6] * 0.2
            # rot_err = self.actions[:, 3:6] * 0.05
            target_rot = quat_from_euler_xyz(self.target_euler[:, 0], self.target_euler[:, 1], self.target_euler[:, 2])
            rot_err = orientation_error(target_rot, self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7].clone())

            dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
            delta = control_ik(self.jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :7], self.device, dpose, self.num_envs)
            self.cur_targets[:, :7] = self.arm_hand_dof_pos[:, 0:7] + delta[:, :7]

            quat_diff = quat_mul(self.segmentation_target_rot, quat_conjugate(self.target_rot))
            rot_dist1 = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

            # self.symmetry_extra_target_rot = quat_mul(self.target_rot, to_torch([0.0, 0.0, 1.0, 0.0], device=self.device).repeat(self.num_envs, 1))
            # quat_diff2 = quat_mul(self.segmentation_target_rot, quat_conjugate(self.symmetry_extra_target_rot))
            # rot_dist2 = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff2[:, 0:3], p=2, dim=-1), max=1.0))
            # rot_dist = torch.min(rot_dist1, rot_dist2)

            # if rot_dist < 0.1:
            #     self.cur_targets[:, 7:23] = self.prev_targets[:, 7:23].clone()
        
        self.cur_targets[:, :] = tensor_clamp(self.cur_targets[:, :],
                                                    self.arm_hand_dof_lower_limits[:],
                                                    self.arm_hand_dof_upper_limits[:])
        
        self.prev_targets[:, :] = self.cur_targets[:, :]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        # self.arm_hand_dof_pos[:, 0:7] = self.cur_targets[:, :7]
        # self.gym.set_dof_state_tensor(self.sim,
        #                                       gymtorch.unwrap_tensor(self.dof_state))

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device) * self.object_rb_masses * self.force_scale

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)
        # self.gym.clear_lines(self.viewer)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.add_debug_lines(self.envs[0], self.segmentation_target_pos[0], self.segmentation_target_rot[0])
        # self.add_debug_lines(self.envs[0], self.target_pos[0], self.target_rot[0])

        if self.viewer and self.debug_viz:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            high_res_image = self.camera_rgb_visulization(camera_tensors=self.high_res_camera_tensors, env_id=0)

            # cv2.namedWindow("DEBUG_INSERTION_VIS", 0)
            # cv2.imshow("DEBUG_INSERTION_VIS", high_res_image)
            # cv2.waitKey(1) 

            self.gym.end_access_image_tensors(self.sim)

            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # for i in range(self.num_envs):
                # self.add_debug_lines(self.envs[i], self.segmentation_target_pos[i], self.segmentation_target_rot[i])
                # self.add_debug_lines(self.envs[i], self.target_pos[i], self.target_rot[i])
                # self.add_debug_lines(self.envs[i], self.mount_base_pos[i], self.mount_base_rot[i])

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])

    def visualize_fingertip(self, color=[0, 0, 0]):
        fingertip_handle_indices = to_torch(self.fingertip_handles, device=self.device, dtype=torch.long).view(-1)

        # for i in range(4):
        #     self.gym.set_rigid_body_color(
        #                 self.envs[0], self.hand_indices[0], fingertip_handle_indices[i], gymapi.MESH_VISUAL, gymapi.Vec3(color[0], color[1], color[2]))

    def camera_rgb_visulization(self, camera_tensors, env_id=0, is_depth_image=False):
        torch_rgba_tensor = camera_tensors[env_id].clone()
        camera_image = torch_rgba_tensor.cpu().numpy()
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
        
        return camera_image

    def camera_segmentation_visulization(self, camera_tensors, camera_seg_tensors, segmentation_id=0, env_id=0, is_depth_image=False):
        torch_rgba_tensor = camera_tensors[env_id].clone()
        torch_seg_tensor = camera_seg_tensors[env_id].clone()

        torch_rgba_tensor[torch_seg_tensor != segmentation_id] = 0

        camera_image = torch_rgba_tensor.cpu().numpy()
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)

        return camera_image

    def compute_emergence_reward(self, camera_tensors, camera_seg_tensors, segmentation_id=0):
        for i in range(self.num_envs):
            torch_seg_tensor = camera_seg_tensors[i]
            self.emergence_pixel[i] = torch_seg_tensor[torch_seg_tensor == segmentation_id].shape[0]

        self.emergence_reward = (self.emergence_pixel - self.last_emergence_pixel) * 2
        self.last_emergence_pixel = self.emergence_pixel.clone()

    def compute_heap_movement_penalty(self, all_lego_brick_pos):
        self.heap_movement_penalty = torch.where(abs(all_lego_brick_pos[:self.num_envs, :, 0] - 1) > 0.25,
                                            torch.where(abs(all_lego_brick_pos[:self.num_envs, :, 1]) > 0.35, torch.ones_like(all_lego_brick_pos[:self.num_envs, :, 0]), torch.zeros_like(all_lego_brick_pos[:self.num_envs, :, 0])), torch.zeros_like(all_lego_brick_pos[:self.num_envs, :, 0]))
        
        self.heap_movement_penalty = torch.sum(self.heap_movement_penalty, dim=1, keepdim=False)
        # self.heap_movement_penalty = torch.where(self.emergence_reward < 0.05, torch.mean(torch.norm(all_lego_brick_pos - last_all_lego_brick_pos, p=2, dim=-1), dim=-1, keepdim=False), torch.zeros_like(self.heap_movement_penalty))
        
        self.last_all_lego_brick_pos = self.all_lego_brick_pos.clone()

    def set_collision_filter(self, envs, indices, filter):
        arm_shape_props = self.gym.get_actor_rigid_shape_properties(envs, indices)
        for object_shape_prop in arm_shape_props:
            object_shape_prop.filter = filter
        self.gym.set_actor_rigid_shape_properties(envs, indices, arm_shape_props)

    def set_dof_effort(self, envs, indices, effort, stiffness):
        arm_dof_props = self.gym.get_actor_dof_properties(envs, indices)
        for i, object_shape_prop in enumerate(arm_dof_props):
            if i >= 7:
                object_shape_prop["effort"] = effort
                object_shape_prop["stiffness"] = stiffness
        self.gym.set_actor_dof_properties(envs, indices, arm_dof_props)

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_hand_reward(
    spin_coef, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes, max_hand_reset_length: int, arm_contacts, palm_contacts_z, segmengtation_object_point_num,
    max_episode_length: float, object_pos, object_rot, object_angvel, target_pos, target_rot, segmentation_target_pos, hand_base_pos, emergence_reward, arm_hand_ff_pos, arm_hand_rf_pos, arm_hand_mf_pos, arm_hand_th_pos, heap_movement_penalty,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, rotation_id: int, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    # Distance from the hand to the object
    # goal_dist = torch.norm(hand_base_pos - segmentation_target_pos, p=2, dim=-1)
    # dist_rew = goal_dist * dist_reward_scale

    arm_hand_finger_dist = (torch.norm(segmentation_target_pos - arm_hand_ff_pos, p=2, dim=-1) + torch.norm(segmentation_target_pos - arm_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(segmentation_target_pos - arm_hand_rf_pos, p=2, dim=-1) + torch.norm(segmentation_target_pos - arm_hand_th_pos, p=2, dim=-1))
    dist_rew = arm_hand_finger_dist * (-0.02)

    action_penalty = torch.sum(actions ** 2, dim=-1)

    arm_contacts_penalty = torch.sum(arm_contacts, dim=-1)
    palm_contacts_penalty = torch.clamp(palm_contacts_z / 100, 0, None)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    emergence_reward = torch.where(progress_buf % max_hand_reset_length == 0, emergence_reward, torch.zeros_like(emergence_reward))
    # emergence_reward = torch.where(arm_hand_finger_dist < 2, emergence_reward / 10, torch.zeros_like(emergence_reward))
    
    success_bonus = torch.zeros_like(emergence_reward)
    success_bonus = torch.where(segmengtation_object_point_num > 40, torch.ones_like(success_bonus) * 20, torch.ones_like(success_bonus) * 0)
    success_bonus = torch.where(progress_buf % max_hand_reset_length == 0, success_bonus, torch.zeros_like(success_bonus))

    # object_up_reward = (segmentation_target_pos[:, 2]-0.5) * 5
    # dist_rew = torch.where(progress_buf % max_hand_reset_length == 0, dist_rew, torch.zeros_like(dist_rew))
    # heap_movement_penalty = torch.where(progress_buf % max_hand_reset_length == 0, torch.clamp(heap_movement_penalty - init_heap_movement_penalty, min=0, max=15), torch.zeros_like(heap_movement_penalty))

    reward = emergence_reward - heap_movement_penalty + dist_rew - arm_contacts_penalty - palm_contacts_penalty + success_bonus

    # if reward[0] != 0:
        # print(palm_contacts_penalty[0])
        # print(success_bonus[0])
        # print(emergence_reward[0])
        # print(heap_movement_penalty[0])
        # print(arm_contacts_penalty[0])
    # print(object_up_reward[0])
    # # print(spin_reward[0])
    # print("----finish----")

    # Fall penalty: distance to the goal is larger than a threshold
    # Check env termination conditions, including maximum success number
    resets = torch.where(arm_hand_finger_dist <= -1, torch.ones_like(reset_buf), reset_buf)
    # resets = torch.where(goal_resets == 1, torch.ones_like(resets), resets)
    # resets = torch.where(goal_resets == 1, torch.ones_like(resets), resets)

    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(timed_out, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, reset_goal_buf, progress_buf, successes, cons_successes

def generate_test_rand():
    exclude_center = (0.25, 0.19)
    exclude_length = 0.3
    exclude_width = 0.1# Define the number of points to generate
    num_points = 48# Define the range of x and y coordinates for the points
    x_range = (0, 1)
    y_range = (0, 1)# Calculate the number of points along each axis
    num_points_axis = int(np.sqrt(num_points))# Generate a grid of points with the specified number of points
    x_coords = np.linspace(*x_range, num_points_axis + 1)
    y_coords = np.linspace(*y_range, num_points_axis + 1)
    xx, yy = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((xx.ravel(), yy.ravel()))# Remove points within the excluded rectangle
    mask = ((exclude_center[0] - exclude_length/2 <= points[:, 0]) &
            (points[:, 0] <= exclude_center[0] + exclude_length/2) &
            (exclude_center[1] - exclude_width/2 <= points[:, 1]) &
            (points[:, 1] <= exclude_center[1] + exclude_width/2))
    points = points[~mask]# If there are not enough points, generate more points
    while len(points) < num_points:
        x_coords = np.linspace(*x_range, num_points_axis * 2)
        y_coords = np.linspace(*y_range, num_points_axis * 2)
        xx, yy = np.meshgrid(x_coords, y_coords)
        points = np.column_stack((xx.ravel(), yy.ravel()))    
        mask = ((exclude_center[0] - exclude_length/2 <= points[:, 0]) &
                (points[:, 0] <= exclude_center[0] + exclude_length/2) &
                (exclude_center[1] - exclude_width/2 <= points[:, 1]) &
                (points[:, 1] <= exclude_center[1] + exclude_width/2))
        points = points[~mask]# Convert the list of points to a PyTorch tensor
    tensor_points = torch.tensor(points)# Resize the tensor to the desired size
    tensor_points = tensor_points[:num_points].view(num_points, 2)
    return



@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot

def orientation_error(desired, current):
	cc = quat_conjugate(current)
	q_r = quat_mul(desired, cc)
	return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def control_ik(j_eef, device, dpose, num_envs):
	# Set controller parameters
	# IK params
    damping = 0.05
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u