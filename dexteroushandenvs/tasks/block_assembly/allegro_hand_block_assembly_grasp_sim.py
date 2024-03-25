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
# from utils.cnn_module import FeatureTunk
# from utils.inhand6d.model import PoseNet
import pyquaternion
import pytorch3d.transforms as transform
import time

class TValue(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TValue, self).__init__()
      #  self.feature_tunk = FeatureTunk(pretrained=False, input_dim=input_dim, output_dim=output_dim)

    def forward(self, inputs):
        # 1 * 8 * 8 feat
        inputs = inputs / 255.0
        outputs = self.feature_tunk(inputs)

        return outputs

class ContactSLAMer(nn.Module):
    def __init__(self, tactile, output_pose_size) :
        super(ContactSLAMer, self).__init__()
        self.linear1 = nn.Linear(tactile, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, output_pose_size)

    def forward(self, contact):
        x = F.relu(self.linear1(contact))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        output_pose = self.output_layer(x)
        return output_pose, x

class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_grad_enabled(self.prev)


class BlockAssemblyGraspSim(BaseTask):

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
            "large":  "urdf/xarm6/xarm6_allegro_left_fsr_large.urdf"
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

        self.stack_obs = 3

        self.num_obs_dict = {
            # "partial_contact": 62,
            "partial_contact": 132,
            "student_partial_contact": 30
        }
        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 188
            # num_states = 154

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

        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.25, 0.6, 1.1)
            cam_target = gymapi.Vec3(0.25, -0.3, 0.5)
            # cam_pos = gymapi.Vec3(26, 2.3, 1.2)
            # cam_target = gymapi.Vec3(0.25, 15, -10.5)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

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
        self.arm_hand_default_dof_pos[:7] = torch.tensor([0.0621,  0.3445,  0.3000, -2.3935, -0.2406,  2.7193,  2.1295], dtype=torch.float, device=self.device)        

        self.arm_hand_default_dof_pos[7:] = to_torch([0.0, -0.174, 0.785, 0.785,
                                            0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785], dtype=torch.float, device=self.device)
        
        self.arm_hand_default_dof_pos[7:] = scale(torch.ones(16, dtype=torch.float, device=self.device), 
                                                self.arm_hand_dof_lower_limits[7:23], self.arm_hand_dof_upper_limits[7:23])


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

        # self.arm_hand_insertion_prepare_dof_pos = to_torch([-0.5124, -0.4954,  0.4536, -2.4975,  0.2445,  2.0486, 1.1839], dtype=torch.float, device=self.device)
        self.arm_hand_insertion_prepare_dof_pos = to_torch([-0.1560, -0.2140, -0.2795, -2.1806, -0.0681,  1.9730,  1.1735], dtype=torch.float, device=self.device)
        self.arm_hand_insertion_prepare_dof_pos_list.append(self.arm_hand_insertion_prepare_dof_pos)
        # self.arm_hand_insertion_prepare_dof_pos = to_torch([-1.0, -0.4954,  0.4536, -2.4975,  0.2445,  2.0486, 1.1839], dtype=torch.float, device=self.device)
        self.arm_hand_insertion_prepare_dof_pos = to_torch([-0.1800, -0.1604, -0.2770, -2.2674, -0.0533,  2.1049,  1.1696], dtype=torch.float, device=self.device)
        # self.arm_hand_insertion_prepare_dof_pos = to_torch([-0.1680, -0.0482, -0.2769, -2.4786, -0.0267,  2.4125,  1.1328], dtype=torch.float, device=self.device)
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
        self.arm_hand_insertion_prepare_dof_poses = self.arm_hand_insertion_prepare_dof_pos_list[0]
        self.end_effector_rotation[:, :] = self.end_effector_rot_list[3]
        self.allegro_dof_low_level_action = torch.zeros((self.num_envs, 16), dtype=torch.float, device=self.device)

        self.state_buf_stack_frames = []
        self.obs_buf_stack_frames = []
        self.student_obs_buf_stack_frames = []
        for i in range(self.stack_obs):
            self.obs_buf_stack_frames.append(torch.zeros_like(self.obs_buf[:, 0:self.one_frame_num_obs]))
            self.state_buf_stack_frames.append(torch.zeros_like(self.states_buf[:, 0:self.one_frame_num_states]))
            self.student_obs_buf_stack_frames.append(torch.zeros_like(self.obs_buf[:, 0:self.one_frame_num_obs]))

        self.multi_object_index = torch.zeros((self.num_envs, 12), device=self.device, dtype=torch.float)
        for i in range(self.num_envs):
            self.multi_object_index[i, i % 12] = 1

        self.saved_grasp_object_ternimal_states = torch.zeros(
            (10000 + 1024, 1, 13), device=self.device, dtype=torch.float)
        self.saved_grasp_hand_ternimal_states = torch.zeros(
            (10000 + 1024, self.num_arm_hand_dofs, 2), device=self.device, dtype=torch.float)
        self.saved_grasp_ternimal_states_index = 0

        self.saved_grasp_object_ternimal_states_list = []
        self.saved_grasp_hand_ternimal_states_list = []
        self.saved_grasp_ternimal_states_index_list = []
        self.can_save = []
        self.last_saved = []

        for i in range(8):
            self.saved_grasp_object_ternimal_states_list.append(self.saved_grasp_object_ternimal_states)
            self.saved_grasp_hand_ternimal_states_list.append(self.saved_grasp_hand_ternimal_states)
            self.saved_grasp_ternimal_states_index_list.append(0)
            self.can_save.append(0)

        self.last_saved =  self.saved_grasp_ternimal_states_index_list.copy()

        # tvalue
        with open("intermediate_state/saved_searching_ternimal_states_good_mo_tvalue.pkl", "rb") as f:
            self.saved_searching_ternimal_states_list = pickle.load(f)

        self.saved_searching_ternimal_state = torch.zeros_like(self.root_state_tensor.view(self.num_envs, -1, 13), device=self.device, dtype=torch.float)

        from policy_sequencing.terminal_value_function import GraspInsertTValue
        self.is_test_tvalue = False
        self.t_value = GraspInsertTValue(input_dim=4, output_dim=2).to(self.device)
        for param in self.t_value.parameters():
            param.requires_grad_(True)
        # self.t_value_obs_buf = torch.zeros_like(self.obs_buf[:, :4])
    
        self.t_value_optimizer = optim.Adam(self.t_value.parameters(), lr=0.0003)
        self.t_value_save_path = "./intermediate_state/grasping_insertion_t_value/"
        os.makedirs(self.t_value_save_path, exist_ok=True)
        # self.t_value.load_state_dict(torch.load("./intermediate_state/t_value/2023-1-19_1:39:15/model_5.pt", map_location='cuda:0'))
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss()

        if self.is_test_tvalue:
            self.t_value.load_state_dict(torch.load("./intermediate_state/grasping_insertion_t_value/tstar/grasp_insert_tvalue_10000_0.834.pt", map_location='cuda:0'))
            self.t_value.to(self.device)
            self.t_value.eval()
        else:
            self.t_value.train()

        self.multi_object_index = torch.zeros((self.num_envs, 12), device=self.device, dtype=torch.float)
        for i in range(self.num_envs):
            self.multi_object_index[i, i % 12] = 1

        self.test_robot_controller = False
        if self.test_robot_controller:
            # origin
            from utils.sequence_controller.nn_controller import SeqNNController
            self.seq_policy = SeqNNController(num_actors=self.num_envs, obs_dim=30)
            self.seq_policy.load("./runs/AllegroHandLegoTestOrientGrasp_27-23-32-45/nn/AllegroHandLegoTestOrientGrasp.pth", None)
            self.switch_policy = False 
            self.seq_policy.select_policy("grasp")

        self.apply_teleoper_perturbation = False
        self.perturb_steps = torch.zeros_like(self.progress_buf, dtype=torch.float32)
        self.perturb_direction = torch_rand_float(-1, 1, (self.num_envs, 6), device=self.device).squeeze(-1)
        self.segmentation_target_init_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3].clone()
        self.segmentation_target_init_rot = self.root_state_tensor[self.lego_segmentation_indices, 3:7].clone()
        self.segmentation_target_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3].clone()
        self.segmentation_target_rot = self.root_state_tensor[self.lego_segmentation_indices, 3:7].clone()
        self.base_pos = self.rigid_body_states[:, 0, 0:3]
        self.extras["student_obs_buf"] = self.obs_buf[:, 0:30].clone()
        self.extras['success_buf'] = torch.zeros_like(self.reset_buf)

        self.use_teleoperation = False
        if self.use_teleoperation:
            # subscribe to input events. This allows input to be used to interact
            # with the simulation
            from policy_sequencing.teleoperation import Teleoperator
            self.teleoperator = Teleoperator(self, max_episode_length=1000000)

        self.target_euler = to_torch([0.0, 3.1415, 1.571], device=self.device).repeat((self.num_envs, 1))

        self.save_hdf5 = False
        if self.save_hdf5:
            import h5py

            hdf5_path = os.path.join("intermediate_state/", "BlockAssemblyGraspSim_datasets.hdf5")
            self.hdf5 = h5py.File(hdf5_path, "w")

            # store some metadata in the attributes of one group
            grp = self.hdf5.create_group("data")
            self.succ_grp = grp.create_group("success_dataset")
            self.fail_grp = grp.create_group("failure_dataset")

            self.success_v_count = 0
            self.failure_v_count = 0

            self.use_temporal_tvalue = False
            self.t_value_obs_buf = torch.zeros((self.num_envs, 62 * 10), dtype=torch.float32, device=self.device)

        self.test_hdf5 = False
        if self.test_hdf5:
            with h5py.File(self.data, "r") as f:
                self.f = f
                list_of_names = []
                self.f.visit(print)
                self.image = self.f["images"]
                self.pose_input = self.f["pose_input"]

                self.f.close()

        self.record_completion_time = False
        if self.record_completion_time:
            self.complete_time_list = []
            self.start_time = time.time()
            self.last_start_time = self.start_time

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim_params.physx.max_gpu_contact_pairs = int(self.sim_params.physx.max_gpu_contact_pairs)
        # self.sim_params.dt = 1./120.

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

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
                robot_dof_props['effort'][i] = 5
                robot_dof_props['stiffness'][i] = 50
                robot_dof_props['damping'][i] = 1

            robot_lower_qpos.append(robot_dof_props['lower'][i])
            robot_upper_qpos.append(robot_dof_props['upper'][i])

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
        table_dims = gymapi.Vec3(1.5, 1.0, 0.6)
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

        print(all_lego_files_name)
        all_lego_files_name = ['1x2.urdf', '1x2_curve.urdf', '1x3_curve_soft.urdf', '1x3_curve.urdf', '1x1.urdf', '1x3.urdf', '1x4.urdf', '2x2_curve_soft.urdf']

        self.collect_unseen_data = False

        lego_assets = []
        lego_start_poses = []
        self.segmentation_id = 1

        self.num_object_bodies = 0
        self.num_object_shapes = 0

        for n in range(9):
            for i, lego_file_name in enumerate(all_lego_files_name):
                lego_asset_options = gymapi.AssetOptions()
                lego_asset_options.disable_gravity = False
                # lego_asset_options.fix_base_link = True
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
                if n % 2 == 0:
                    lego_start_pose.p = gymapi.Vec3(-0.17 + 0.17 * int(i % 3) + 0.25, -0.11 + 0.11 * int(i / 3) + 0.19, 0.62 + n * 0.06)
                else:
                    lego_start_pose.p = gymapi.Vec3(0.17 - 0.17 * int(i % 3) + 0.25, 0.11 - 0.11 * int(i / 3) + 0.19, 0.62 + n * 0.06)

                lego_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0.0, 0.785)
                # Assets visualization
                
                lego_assets.append(lego_asset)
                lego_start_poses.append(lego_start_pose)

        lego_asset_options = gymapi.AssetOptions()
        lego_asset_options.disable_gravity = False
        lego_asset_options.fix_base_link = True
        lego_asset_options.thickness = 0.00001
        lego_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # lego_asset_options.density = 2000
        flat_lego_begin = len(lego_assets)        
        ran_list = [0 ,0 ,0, 1, 2, 2]
        lego_list = [0, 5, 6]
        bianchang = [0.03, 0.045, 0.06]        
        for j in range(10):
            random.shuffle(ran_list)
            lego_center = [0.254 - bianchang[ran_list[0]] + 0.25, 0.175 + 0.19 - 0.039 * j, 0.625]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0] , lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[0]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)            
            lego_center = [lego_center[0] - (bianchang[ran_list[0]] + bianchang[ran_list[1]] + 0.006), lego_center[1], lego_center[2]]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0], lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[1]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)            
            lego_center = [lego_center[0] - (bianchang[ran_list[1]] + bianchang[ran_list[2]] + 0.006), lego_center[1], lego_center[2]]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0], lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[2]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)            
            lego_center = [lego_center[0] - (bianchang[ran_list[2]] + bianchang[ran_list[3]] + 0.006), lego_center[1], lego_center[2]]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0], lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[3]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)            
            lego_center = [lego_center[0] - (bianchang[ran_list[3]] + bianchang[ran_list[4]] + 0.006), lego_center[1], lego_center[2]]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0], lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[4]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)            
            lego_center = [lego_center[0] - (bianchang[ran_list[4]] + bianchang[ran_list[5]] + 0.006), lego_center[1], lego_center[2]]
            lego_start_pose = gymapi.Transform()
            lego_start_pose.p = gymapi.Vec3(lego_center[0], lego_center[1], lego_center[2])
            lego_file_name = all_lego_files_name[lego_list[ran_list[5]]]
            lego_asset = self.gym.load_asset(self.sim, asset_root, lego_path + lego_file_name, lego_asset_options)
            lego_assets.append(lego_asset)
            lego_start_poses.append(lego_start_pose)        
        
            self.num_object_bodies += self.gym.get_asset_rigid_body_count(lego_asset) * 6
            self.num_object_shapes += self.gym.get_asset_rigid_shape_count(lego_asset) * 6
            print("num_object_shapes: ", self.num_object_shapes)
            print("num_object_bodies: ", self.num_object_bodies)

        flat_lego_end = len(lego_assets)

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

        extra_lego_assets = []

        # fake extra lego
        extra_lego_asset = self.gym.load_asset(self.sim, asset_root, "urdf/blender/assets_for_insertion/urdf/4x4x1_real.urdf", extra_lego_asset_options)
        extra_lego_assets.append(extra_lego_asset)

        self.num_object_bodies += self.gym.get_asset_rigid_body_count(extra_lego_asset)
        self.num_object_shapes += self.gym.get_asset_rigid_shape_count(extra_lego_asset)
        print("num_object_shapes: ", self.num_object_shapes)
        print("num_object_bodies: ", self.num_object_bodies)

        extra_lego_start_pose = gymapi.Transform()
        extra_lego_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0.0, 0.0)
        # Assets visualization
        extra_lego_start_pose.p = gymapi.Vec3(0.25, -0.19, 0.618)

        # compute aggregate size
        max_agg_bodies = self.num_arm_hand_bodies + 2 + 1 + self.num_object_bodies + 5 + 100
        max_agg_shapes = self.num_arm_hand_shapes + 2 + 1 + self.num_object_shapes + 5 + 100

        self.arm_hands = []
        self.envs = []

        self.object_init_state = []
        self.lego_init_states = []
        self.hand_start_states = []
        self.extra_lego_init_states = []
        self.insert_lego_init_states = []

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

        arm_hand_rb_count = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        self.object_rb_handles = [7]

        self.cameras = []
        self.camera_tensors = []
        self.camera_seg_tensors = []
        self.camera_view_matrixs = []
        self.camera_proj_matrixs = []

        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 128
        self.camera_props.height = 128
        self.camera_props.enable_tensors = True

        self.high_res_camera_tensors = []

        self.high_res_camera_props = gymapi.CameraProperties()
        self.high_res_camera_props.horizontal_fov = 65
        self.high_res_camera_props.width = 80
        self.high_res_camera_props.height = 80
        self.high_res_camera_props.enable_tensors = True

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

            arm_hand_actor_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, arm_hand_actor)
            for _, arm_hand_actor_shape_prop in enumerate(arm_hand_actor_shape_props):
                arm_hand_actor_shape_prop.friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, arm_hand_actor, arm_hand_actor_shape_props)

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

            # add box
            for box_i, box_asset in enumerate(box_assets):
                box_handle = self.gym.create_actor(env_ptr, box_asset, box_start_poses[box_i], "box_{}".format(box_i), i, 0, 0)
                # self.lego_init_state.append([lego_init_state.p.x, lego_init_state.p.y, object_start_pose.p.z,
                #                             lego_init_state.r.x, lego_init_state.r.y, object_start_pose.r.z, object_start_pose.r.w,
                #                             0, 0, 0, 0, 0, 0])
                # object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
                # self.object_indices.append(object_idx)
                self.gym.set_rigid_body_color(env_ptr, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1., 1., 1))
                # self.gym.set_rigid_body_color(env_ptr, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.4, 0))

            # add lego
            # ['1x2.urdf', '1x2.urdf', '1x3_curve_soft.urdf', '1x3_curve.urdf', '1x1.urdf', '1x3.urdf', '1x4.urdf', '2x2_curve_soft.urdf']
            color_map = [[0.8, 0.64, 0.2], [0.13, 0.54, 0.13], [0, 0.4, 0.8], [1, 0.54, 0], [0.69, 0.13, 0.13], [0.69, 0.13, 0.13], [0, 0.4, 0.8], [0.8, 0.64, 0.2]]
            lego_idx = []
            self.segmentation_id = i % 8

            if self.segmentation_id in [3, 4, 7] and not self.collect_unseen_data:
                self.segmentation_id = 0

            for lego_i, lego_asset in enumerate(lego_assets):
                lego_handle = self.gym.create_actor(env_ptr, lego_asset, lego_start_poses[lego_i], "lego_{}".format(lego_i), i, -1, lego_i + 1)
                self.lego_init_states.append([lego_start_poses[lego_i].p.x, lego_start_poses[lego_i].p.y, lego_start_poses[lego_i].p.z,
                                            lego_start_poses[lego_i].r.x, lego_start_poses[lego_i].r.y, lego_start_poses[lego_i].r.z, lego_start_poses[lego_i].r.w,
                                            0, 0, 0, 0, 0, 0])
                idx = self.gym.get_actor_index(env_ptr, lego_handle, gymapi.DOMAIN_SIM)

                if lego_i == self.segmentation_id:
                    self.lego_segmentation_indices.append(idx)

                lego_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, lego_handle)
                for lego_body_prop in lego_body_props:
                    if flat_lego_end > lego_i > flat_lego_begin:
                        lego_body_prop.mass *= 1
                    lego_body_prop.mass *= 1

                self.gym.set_actor_rigid_body_properties(env_ptr, lego_handle, lego_body_props)
                lego_idx.append(idx)

                color = color_map[lego_i % 8]
                if flat_lego_end > lego_i > flat_lego_begin:
                    color = color_map[random.randint(0, 7)]
                self.gym.set_rigid_body_color(env_ptr, lego_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(color[0], color[1], color[2]))
            self.lego_indices.append(lego_idx)

            extra_lego_handle = self.gym.create_actor(env_ptr, extra_lego_assets[0], extra_lego_start_pose, "extra_lego", i, 0, 0)
            self.extra_lego_init_states.append([extra_lego_start_pose.p.x, extra_lego_start_pose.p.y, extra_lego_start_pose.p.z,
                                        extra_lego_start_pose.r.x, extra_lego_start_pose.r.y, extra_lego_start_pose.r.z, extra_lego_start_pose.r.w,
                                        0, 0, 0, 0, 0, 0])
            self.gym.get_actor_index(env_ptr, extra_lego_handle, gymapi.DOMAIN_SIM)
            extra_object_idx = self.gym.get_actor_index(env_ptr, extra_lego_handle, gymapi.DOMAIN_SIM)
            extra_lego_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, extra_lego_handle)
            self.gym.set_rigid_body_color(env_ptr, extra_lego_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 1, 1))
            self.extra_object_indices.append(extra_object_idx)

            self.mount_rigid_body_index = self.gym.find_actor_rigid_body_index(env_ptr, arm_hand_actor, "panda_link7", gymapi.DOMAIN_ENV)

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

            self.envs.append(env_ptr)
            self.arm_hands.append(arm_hand_actor)

        self.emergence_reward = torch.zeros_like(self.rew_buf, device=self.device, dtype=torch.float)
        self.emergence_pixel = torch.zeros_like(self.rew_buf, device=self.device, dtype=torch.float)
        self.last_emergence_pixel = torch.zeros_like(self.rew_buf, device=self.device, dtype=torch.float)

        self.heap_movement_penalty= torch.zeros_like(self.rew_buf, device=self.device, dtype=torch.float)

        # Acquire specific links.
        sensor_handles = [1, 2, 3, 4, 5, 6]
        self.sensor_handle_indices = to_torch(sensor_handles, dtype=torch.int64)

        self.fingertip_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, name) for name in self.fingertip_names]
        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.02
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.lego_init_states = to_torch(self.lego_init_states, device=self.device).view(self.num_envs, len(lego_assets), 13)
        self.insert_lego_init_states = to_torch(self.insert_lego_init_states, device=self.device).view(self.num_envs, -1, 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.lego_indices = to_torch(self.lego_indices, dtype=torch.long, device=self.device)
        self.lego_segmentation_indices = to_torch(self.lego_segmentation_indices, dtype=torch.long, device=self.device)
        self.insert_lego_indices = to_torch(self.insert_lego_indices, dtype=torch.long, device=self.device)
        self.pre_exchange_lego_segmentation_indices = self.lego_segmentation_indices.clone()

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            torch.tensor(self.spin_coef).to(self.device), self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes, self.hand_reset_step, self.contacts, self.segmentation_target_rot, self.extra_target_pos, self.extra_target_rot,
            self.max_episode_length, self.object_pos, self.object_rot, self.object_angvel, self.goal_pos, self.goal_rot, self.segmentation_target_pos, self.hand_base_pos, self.emergence_reward, self.arm_hand_ff_pos, self.arm_hand_rf_pos, self.arm_hand_mf_pos, self.arm_hand_th_pos, self.heap_movement_penalty, self.segmentation_target_init_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty, self.rotation_id,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen"), self.segmentation_target_linvel, self.num_envs, self.z_unit_tensor, self.arm_hand_th_rot, self.x_unit_tensor, self.segmentation_target_init_rot, self.tvalue,
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

    def compute_observations(self, is_searching=False, last_action=0):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.robot_base_pos = self.root_state_tensor[self.hand_indices, 0:3]
        self.robot_base_rot = self.root_state_tensor[self.hand_indices, 3:7]

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

        self.segmentation_target_state = self.root_state_tensor[self.lego_segmentation_indices, 0:13]
        self.segmentation_target_pose = self.root_state_tensor[self.lego_segmentation_indices, 0:7]
        self.segmentation_target_pos = self.root_state_tensor[self.lego_segmentation_indices, 0:3]
        self.segmentation_target_rot = self.root_state_tensor[self.lego_segmentation_indices, 3:7]
        self.segmentation_target_linvel = self.root_state_tensor[self.lego_segmentation_indices, 7:10]
        self.segmentation_target_angvel = self.root_state_tensor[self.lego_segmentation_indices, 10:13]

        self.extra_target_pose = self.root_state_tensor[self.extra_object_indices, 0:7]
        self.extra_target_pos = self.root_state_tensor[self.extra_object_indices, 0:3]
        self.extra_target_rot = self.root_state_tensor[self.extra_object_indices, 3:7]

        self.extra_target_pos[:] += quat_apply(self.extra_target_rot[:], to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.0375 * 1)
        self.extra_target_pos[:] += quat_apply(self.extra_target_rot[:], to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.015)

        self.arm_hand_ff_pos = self.rigid_body_states[:, self.fingertip_handles[0], 0:3]
        self.arm_hand_ff_rot = self.rigid_body_states[:, self.fingertip_handles[0], 3:7]
        self.arm_hand_ff_linvel = self.rigid_body_states[:, self.fingertip_handles[0], 7:10]
        self.arm_hand_ff_angvel = self.rigid_body_states[:, self.fingertip_handles[0], 10:13]

        self.arm_hand_mf_pos = self.rigid_body_states[:, self.fingertip_handles[1], 0:3]
        self.arm_hand_mf_rot = self.rigid_body_states[:, self.fingertip_handles[1], 3:7]
        self.arm_hand_mf_linvel = self.rigid_body_states[:, self.fingertip_handles[1], 7:10]
        self.arm_hand_mf_angvel = self.rigid_body_states[:, self.fingertip_handles[1], 10:13]

        self.arm_hand_rf_pos = self.rigid_body_states[:, self.fingertip_handles[2], 0:3]
        self.arm_hand_rf_rot = self.rigid_body_states[:, self.fingertip_handles[2], 3:7]
        self.arm_hand_rf_linvel = self.rigid_body_states[:, self.fingertip_handles[2], 7:10]
        self.arm_hand_rf_angvel = self.rigid_body_states[:, self.fingertip_handles[2], 10:13]
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
        contacts = contacts[:, self.sensor_handle_indices, :] # 12
        contacts = torch.norm(contacts, dim=-1)
        self.contacts = torch.where(contacts >= 0.1, 1.0, 0.0)

        self.arm_hand_finger_dist = (torch.norm(self.segmentation_target_pos - self.arm_hand_ff_pos, p=2, dim=-1) + torch.norm(self.segmentation_target_pos - self.arm_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(self.segmentation_target_pos - self.arm_hand_rf_pos, p=2, dim=-1) + torch.norm(self.segmentation_target_pos - self.arm_hand_th_pos, p=2, dim=-1))
        
        fingertip_handle_indices = to_torch(self.fingertip_handles, device=self.device, dtype=torch.long).view(-1)        
        for i in range(4):
            self.gym.set_rigid_body_color(
                    self.envs[0], self.hand_indices[0], fingertip_handle_indices[i], gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.3, 1.0))
        
        self.q_robot_base_inv, self.p_robot_base_inv = tf_inverse(self.robot_base_rot, self.robot_base_pos)
        self.hand_base_view_hand_rot, self.hand_base_view_hand_pos = tf_combine(self.q_robot_base_inv, self.p_robot_base_inv, self.hand_base_rot, self.hand_base_pos)

        # object 6d pose randomization
        self.mount_pos = self.rigid_body_states[:, self.mount_rigid_body_index, 0:3]
        self.mount_rot = self.rigid_body_states[:, self.mount_rigid_body_index, 3:7]

        self.q_camera, self.p_camera = tf_combine(self.mount_rot, self.mount_pos, self.camera_offset_quat.repeat(self.num_envs, 1), self.camera_offset_pos.repeat(self.num_envs, 1))
        self.q_camera_inv, self.p_camera_inv = tf_inverse(self.q_camera, self.p_camera)

        self.camera_view_segmentation_target_rot, self.camera_view_segmentation_target_pos = tf_combine(self.q_camera_inv, self.p_camera_inv, self.segmentation_target_rot, self.segmentation_target_pos)

        pose_rand_floats = torch_rand_float(-1, 1, (self.num_envs, 7), device=self.device)
        self.camera_view_segmentation_target_pos_noise = self.camera_view_segmentation_target_pos + pose_rand_floats[:, 0:3] * 0.003
        self.camera_view_segmentation_target_rot_noise = self.camera_view_segmentation_target_rot + pose_rand_floats[:, 3:7] * 0.2
        self.camera_view_segmentation_target_init_rot, self.camera_view_segmentation_target_init_pos = tf_combine(self.q_camera_inv, self.p_camera_inv, self.segmentation_target_init_rot, self.segmentation_target_init_pos)

        self.perturbation_pos = torch.ones_like(self.actions[:, 0:3]) * self.perturb_direction[:, 0:3]
        self.perturbation_rot = torch.ones_like(self.actions[:, 0:3]) * self.perturb_direction[:, 3:6]
        
        self.apply_teleoper_perturbation = False
        self.apply_teleoper_perturbation_env_id = torch.where(abs(self.progress_buf - self.perturb_steps.squeeze(-1)) < 4, 1, 0).nonzero(as_tuple=False)

        axis1 = quat_apply(self.segmentation_target_rot, self.z_unit_tensor)
        axis2 = self.z_unit_tensor
        dot1 = torch.bmm(axis1.view(self.num_envs, 1, 3), axis2.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
        self.lego_z_align_reward = (torch.sign(dot1) * dot1 ** 2)

        self.tvalue_predict_confident = self.t_value(self.camera_view_segmentation_target_rot)
        self.tvalue = torch.sigmoid(self.tvalue_predict_confident)[:, 1]

        if is_searching:
            self.compute_searching_observations(last_action)
        elif self.obs_type == "partial_contact":
            self.compute_sim_observations()
            self.compute_contact_asymmetric_observations()

        # compute temporal tvalue
        if self.save_hdf5:
            if self.use_temporal_tvalue:
                for i in range(10):
                    if i == 10-1:
                        self.temp_obs = self.obs_buf[:, 0:62].clone()
                        self.temp_obs[:, 26:30] = self.camera_view_segmentation_target_rot
                        self.t_value_obs_buf[:, i*62:(i+1)*62] = self.temp_obs.clone()
                    else:
                        self.t_value_obs_buf[:, i*62:(i+1)*62] = self.t_value_obs_buf[:, (i+1)*62:(i+2)*62]

    def compute_contact_asymmetric_observations(self):
            self.states_buf[:, 0:23] = unscale(self.arm_hand_dof_pos[:, 0:23],
                                                                self.arm_hand_dof_lower_limits[0:23],
                                                                self.arm_hand_dof_upper_limits[0:23])
            self.states_buf[:, 23:46] = self.vel_obs_scale * self.arm_hand_dof_vel[:, 0:23]

            self.states_buf[:, 46:49] = self.arm_hand_ff_pos
            self.states_buf[:, 49:52] = self.arm_hand_rf_pos
            self.states_buf[:, 52:55] = self.arm_hand_mf_pos
            self.states_buf[:, 55:58] = self.arm_hand_th_pos

            self.states_buf[:, 58:81] = self.actions
            self.states_buf[:, 81:88] = self.hand_base_pose

            self.states_buf[:, 88:95] = self.segmentation_target_pose

            self.states_buf[:, 95:98] = self.hand_base_linvel
            self.states_buf[:, 98:101] = self.hand_base_angvel

            self.states_buf[:, 101:105] = self.arm_hand_ff_rot  
            self.states_buf[:, 105:108] = self.arm_hand_ff_linvel
            self.states_buf[:, 108:111] = self.arm_hand_ff_angvel

            self.states_buf[:, 111:115] = self.arm_hand_mf_rot  
            self.states_buf[:, 115:118] = self.arm_hand_mf_linvel
            self.states_buf[:, 118:121] = self.arm_hand_mf_angvel

            self.states_buf[:, 121:125] = self.arm_hand_rf_rot  
            self.states_buf[:, 125:128] = self.arm_hand_rf_linvel
            self.states_buf[:, 128:131] = self.arm_hand_rf_angvel

            self.states_buf[:, 131:135] = self.arm_hand_th_rot  
            self.states_buf[:, 135:138] = self.arm_hand_th_linvel
            self.states_buf[:, 138:141] = self.arm_hand_th_angvel

            self.states_buf[:, 142:145] = self.segmentation_target_linvel
            self.states_buf[:, 145:148] = self.segmentation_target_angvel

            self.states_buf[:, 148:151] = self.segmentation_target_init_pos
            self.states_buf[:, 151:154] = self.segmentation_target_pos - self.segmentation_target_init_pos

            # self.states_buf[:, 154:166] = self.multi_object_index
            self.states_buf[:, 154:157] = self.hand_base_pos - self.segmentation_target_pos
            self.states_buf[:, 157:161] = quat_mul(self.hand_base_rot, quat_conjugate(self.segmentation_target_rot))

            self.states_buf[:, 161:164] = self.segmentation_target_pos - self.arm_hand_ff_pos
            self.states_buf[:, 164:167] = self.segmentation_target_pos - self.arm_hand_rf_pos
            self.states_buf[:, 167:170] = self.segmentation_target_pos - self.arm_hand_mf_pos
            self.states_buf[:, 170:173] = self.segmentation_target_pos - self.arm_hand_th_pos

            self.states_buf[:, 173:174] = self.arm_hand_finger_dist.unsqueeze(-1)

            self.states_buf[:, 174:177] = self.camera_view_segmentation_target_pos
            self.states_buf[:, 177:181] = self.camera_view_segmentation_target_rot

            self.states_buf[:, 181:184] = self.camera_view_segmentation_target_pos
            self.states_buf[:, 184:188] = self.camera_view_segmentation_target_rot

            for i in range(len(self.state_buf_stack_frames) - 1):
                self.states_buf[:, (i+1) * self.one_frame_num_states:(i+2) * self.one_frame_num_states] = self.state_buf_stack_frames[i]
                self.state_buf_stack_frames[i] = self.states_buf[:, (i) * self.one_frame_num_states:(i+1) * self.one_frame_num_states].clone()

    def compute_real_observations(self):
        self.obs_buf[:, 0:16] = unscale(self.arm_hand_dof_pos[:, 7:23],
                                                            self.arm_hand_dof_lower_limits[7:23],
                                                            self.arm_hand_dof_upper_limits[7:23])
    
        # self.obs_buf[:, 16:19] = self.hand_base_pos
        # self.obs_buf[:, 19:23] = self.hand_base_rot

        # self.obs_buf[:, 23:26] = self.segmentation_target_pos
        # self.obs_buf[:, 26:30] = self.segmentation_target_rot

        self.obs_buf[:, 30:46] = self.actions[:, 7:23] - unscale(self.arm_hand_dof_pos[:, 7:23],
                                                            self.arm_hand_dof_lower_limits[7:23],
                                                            self.arm_hand_dof_upper_limits[7:23])
        
        self.obs_buf[:, 46:62] = self.actions[:, 7:23]

    def compute_sim_observations(self, full_contact=True):
        self.obs_buf[:, 0:16] = unscale(self.arm_hand_dof_pos[:, 7:23],
                                                            self.arm_hand_dof_lower_limits[7:23],
                                                            self.arm_hand_dof_upper_limits[7:23])
    
        self.obs_buf[:, 16:19] = self.hand_base_view_hand_pos
        self.obs_buf[:, 19:23] = self.hand_base_view_hand_rot

        self.obs_buf[:, 23:26] = self.camera_view_segmentation_target_pos
        self.obs_buf[:, 26:30] = self.camera_view_segmentation_target_rot

        self.obs_buf[:, 30:46] = self.vel_obs_scale * self.arm_hand_dof_vel[:, 7:23]

        self.obs_buf[:, 46:59] = self.arm_hand_ff_state
        self.obs_buf[:, 59:72] = self.arm_hand_rf_state
        self.obs_buf[:, 72:85] = self.arm_hand_mf_state
        self.obs_buf[:, 85:98] = self.arm_hand_th_state

        self.obs_buf[:, 98:111] = self.segmentation_target_state

        self.obs_buf[:, 111:114] = self.hand_base_pos
        self.obs_buf[:, 114:118] = self.hand_base_rot

        self.obs_buf[:, 118:121] = self.segmentation_target_init_pos
        self.obs_buf[:, 121:125] = self.segmentation_target_init_rot

        self.obs_buf[:, 125:128] = self.segmentation_target_pos - self.segmentation_target_init_pos
        self.obs_buf[:, 128:131] = self.hand_base_pos - self.segmentation_target_pos

        # self.obs_buf[:, 131:132] = self.progress_buf.unsqueeze(-1) / self.max_episode_length

        for i in range(len(self.obs_buf_stack_frames) - 1):
            self.obs_buf[:, (i+1) * self.one_frame_num_obs:(i+2) * self.one_frame_num_obs] = self.obs_buf_stack_frames[i]
            self.obs_buf_stack_frames[i] = self.obs_buf[:, (i) * self.one_frame_num_obs:(i+1) * self.one_frame_num_obs].clone()

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

    # default robot pose: [0.00, 0.782, -1.087, 3.487, 2.109, -1.415]
    def reset_idx(self, env_ids, goal_env_ids):
        if self.record_completion_time:
            self.end_time = time.time()
            self.complete_time_list.append(self.end_time - self.last_start_time)
            self.last_start_time = self.end_time
            print("complete_time_mean: ", np.array(self.complete_time_list).mean())
            print("complete_time_std: ", np.array(self.complete_time_list).std())
            if len(self.complete_time_list) == 25:
                with open("output_video/grasp_complete_time.pkl", "wb") as f:
                    pickle.dump(self.complete_time_list, f)
                exit()

        axis1 = quat_apply(self.segmentation_target_rot, self.z_unit_tensor)
        axis2 = self.z_unit_tensor
        dot1 = torch.bmm(axis1.view(self.num_envs, 1, 3), axis2.view(self.num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
        lego_z_align_reward = (torch.sign(dot1) * dot1 ** 2)

        if self.total_steps > 0:
            self.success_cond = torch.where(self.arm_hand_finger_dist < 0.4, torch.ones_like(self.reset_buf), torch.zeros_like(self.reset_buf))
            print("grasp_success_rate: ", self.success_cond.float().mean())
            self.record_8_type = [0, 0, 0, 0, 0, 0, 0, 0]
            for i in range(self.num_envs):
                object_idx = i % 8
                if self.success_cond[i] >= 1:
                    self.record_8_type[object_idx] += 1 

            for i in range(8):
                self.record_8_type[i] /= (self.num_envs / 8)
            print("grasp_success_rate_index: ", self.record_8_type)

        if self.save_hdf5:
            print("success_v_count: ", self.success_v_count)
            print("failure_v_count: ", self.failure_v_count)

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # train V value
        if self.total_steps > 0:
            self.saved_grasp_hand_ternimal_state = self.dof_state.clone().view(self.num_envs, -1, 2)[:, :self.num_arm_hand_dofs]
            self.saved_grasp_object_ternimal_state = self.root_state_tensor.clone()[self.lego_segmentation_indices]
            for i in env_ids:
                object_i = i % 8
                if self.segmentation_target_pos[i, 1] < 0.0:
                    if self.arm_hand_finger_dist[i] < 0.6:
                        if self.tvalue[i] > 0.8:
                            if self.save_hdf5:
                                if self.use_temporal_tvalue:
                                    self.succ_grp.create_dataset("{}th_success_data".format(self.success_v_count), data=self.t_value_obs_buf[i].cpu().numpy())
                                else:
                                    self.succ_grp.create_dataset("{}th_success_data".format(self.success_v_count), data=self.camera_view_segmentation_target_rot[i].cpu().numpy())
                                
                                self.success_v_count += 1
                            
                            self.saved_grasp_hand_ternimal_states_list[object_i][self.saved_grasp_ternimal_states_index_list[object_i]:self.saved_grasp_ternimal_states_index_list[object_i] + 1] = self.saved_grasp_hand_ternimal_state[i]
                            self.saved_grasp_object_ternimal_states_list[object_i][self.saved_grasp_ternimal_states_index_list[object_i]:self.saved_grasp_ternimal_states_index_list[object_i] + 1] = self.saved_grasp_object_ternimal_state[i]
                            self.saved_grasp_ternimal_states_index_list[object_i] += 1
                        else:
                            if self.save_hdf5:
                                if self.use_temporal_tvalue:
                                    self.fail_grp.create_dataset("{}th_failure_data".format(self.failure_v_count), data=self.t_value_obs_buf[i].cpu().numpy())
                                else:
                                    self.fail_grp.create_dataset("{}th_failure_data".format(self.failure_v_count), data=self.camera_view_segmentation_target_rot[i].cpu().numpy())
                                self.failure_v_count += 1
                    else:
                        if self.save_hdf5:
                            if self.use_temporal_tvalue:
                                self.fail_grp.create_dataset("{}th_failure_data".format(self.failure_v_count), data=self.t_value_obs_buf[i].cpu().numpy())
                            else:
                                self.fail_grp.create_dataset("{}th_failure_data".format(self.failure_v_count), data=self.camera_view_segmentation_target_rot[i].cpu().numpy())
                            self.failure_v_count += 1
                else:
                    if self.save_hdf5:
                        if self.use_temporal_tvalue:
                            self.fail_grp.create_dataset("{}th_failure_data".format(self.failure_v_count), data=self.t_value_obs_buf[i].cpu().numpy())
                        else:
                            self.fail_grp.create_dataset("{}th_failure_data".format(self.failure_v_count), data=self.camera_view_segmentation_target_rot[i].cpu().numpy())
                        self.failure_v_count += 1

                if self.saved_grasp_ternimal_states_index_list[object_i] > 5000:
                    self.saved_grasp_ternimal_states_index_list[object_i] = 0
                    self.can_save[object_i] = 1

            for j in range(8):            
                print("saved_grasp_ternimal_states_index_{}: ".format(j), self.saved_grasp_ternimal_states_index_list[j])

            # if all([i == 1 for i in self.can_save]):
            #     with open("intermediate_state/saved_grasping_hand_ternimal_states_good_mo_sim.pkl", "wb") as f:
            #         pickle.dump(self.saved_grasp_hand_ternimal_states_list, f)
            #     with open("intermediate_state/saved_grasping_object_ternimal_states_good_mo_sim.pkl", "wb") as f:
            #         pickle.dump(self.saved_grasp_object_ternimal_states_list, f)

            #     for j in range(8):            
            #         self.saved_grasp_ternimal_states_index_list[j] = 0

            #     if self.save_hdf5:
            #         self.hdf5.close()
            #     exit()

        self.perturb_steps[env_ids] = torch_rand_float(0, 150, (len(env_ids), 1), device=self.device).squeeze(-1)
        self.perturb_direction[env_ids] = torch_rand_float(-1, 1, (len(env_ids), 6), device=self.device).squeeze(-1)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_hand_dofs * 2 + 5), device=self.device)
        self.base_pos[env_ids, :] = self.rigid_body_states[env_ids, 0, 0:3] + rand_floats[:, 7:10] * 0.00
        self.base_pos[env_ids, 2] += 0.17
        
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

        self.target_rot_rand = random.sample(range(4), 1)

        # randomize segmentation object
        quat = gymapi.Quat().from_euler_zyx(self.target_rot_rand[0] * 1.57, 0.0, rand_floats[0, 2] * 3.14)
        self.root_state_tensor[self.lego_segmentation_indices[env_ids], 0] = rand_floats[:, 0] * 0.05 + 0.1
        self.root_state_tensor[self.lego_segmentation_indices[env_ids], 1] = rand_floats[:, 1] * 0.05 + 0.2
        self.root_state_tensor[self.lego_segmentation_indices[env_ids], 2] = 0.65
        self.root_state_tensor[self.lego_segmentation_indices[env_ids], 3] = quat.x
        self.root_state_tensor[self.lego_segmentation_indices[env_ids], 4] = quat.y
        self.root_state_tensor[self.lego_segmentation_indices[env_ids], 5] = quat.z
        self.root_state_tensor[self.lego_segmentation_indices[env_ids], 6] = quat.w
        self.root_state_tensor[self.lego_segmentation_indices[env_ids], 7:13] = 0

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids],
                                                 self.lego_indices[env_ids].view(-1)]).to(torch.int32))
        # tvalue 50
        # policy sequencing 500
        self.env_rand_range = range(0, 5000)
        for env_id in env_ids:
            object_i = env_id % 8
            self.env_rand = random.sample(self.env_rand_range, 1)
            self.root_state_tensor[self.lego_indices[env_id].view(-1), :] = self.saved_searching_ternimal_states_list[object_i][self.env_rand].clone().view(132, 13)
        
        self.root_state_tensor[self.lego_indices[env_ids].view(-1), 7:13] = 0
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))
        
        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                    * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        # reset shadow hand
        self.arm_hand_prepare_dof_poses[:, :] = self.arm_hand_prepare_dof_pos_list[0]
        self.end_effector_rotation[:, :] = self.end_effector_rot_list[0]

        self.arm_hand_dof_pos[env_ids, 0:23] = self.arm_hand_prepare_dof_poses[0]
        self.prev_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_prepare_dof_poses[0]
        self.cur_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_prepare_dof_poses[0]
        self.arm_hand_dof_vel[env_ids, :] = self.arm_hand_dof_default_vel 

        self.arm_hand_dof_pos[env_ids, 7:23] = scale(torch.tensor([0, 0, -1, 0.5, 1, 0, -1, 0.5, 0, 0, -1, 0.5, 0, 0, -1, 0.5], dtype=torch.float, device=self.device), 
                                        self.arm_hand_dof_lower_limits[7:23], self.arm_hand_dof_upper_limits[7:23])
        self.prev_targets[env_ids, 7:self.num_arm_hand_dofs] = scale(torch.tensor([0, 0, -1, 0.5, 1, 0, -1, 0.5, 0, 0, -1, 0.5, 0, 0, -1, 0.5], dtype=torch.float, device=self.device), 
                                        self.arm_hand_dof_lower_limits[7:23], self.arm_hand_dof_upper_limits[7:23])
        self.cur_targets[env_ids, 7:self.num_arm_hand_dofs] = scale(torch.tensor([0, 0, -1, 0.5, 1, 0, -1, 0.5, 0, 0, -1, 0.5, 0, 0, -1, 0.5], dtype=torch.float, device=self.device), 
                                        self.arm_hand_dof_lower_limits[7:23], self.arm_hand_dof_upper_limits[7:23])

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.segmentation_target_init_pos[env_ids, :] = self.root_state_tensor[self.lego_segmentation_indices[env_ids], 0:3].clone()
        self.segmentation_target_init_rot[env_ids, :] = self.root_state_tensor[self.lego_segmentation_indices[env_ids], 3:7].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        self.meta_rew_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)

        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        ##############################################
        ########     test robot controller  ########
        ##############################################
        if self.test_robot_controller:
            real_world_obs = self.obs_buf[:, 0:30].clone()
            real_world_obs[:, 0:16] = self.arm_hand_dof_pos[:, 7:23].clone()
            real_world_obs[:, 0:16] = scale(self.obs_buf[:, 0:16],
                                                            self.arm_hand_dof_lower_limits[7:23],
                                                            self.arm_hand_dof_upper_limits[7:23])
            targets = self.seq_policy.predict(input=real_world_obs)

            self.cur_targets[:, :] = targets[:, :]

        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 7:23],
                                                                    self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                    self.arm_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        insert_pre_0_ids = [self.progress_buf > 75]
        insert_pre_1_ids = [self.progress_buf > 100]
        insert_pre_2_ids = [self.progress_buf > 125]

        pos_err = self.actions[:, 0:3] * 0.64
        rot_err = self.actions[:, 3:6] * 0.2
        pos_err[:, 2][insert_pre_0_ids] = 0.2 + 0.22 + (self.segmentation_target_init_pos - self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3])[:, 2][insert_pre_0_ids]
        pos_err[:, 0][insert_pre_0_ids] = 0
        pos_err[:, 1][insert_pre_0_ids] = 0

        dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
        delta = control_ik(self.jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :7], self.device, dpose, self.num_envs)
        self.cur_targets[:, :7] = self.arm_hand_dof_pos[:, 0:7] + delta[:, :7]

        self.cur_targets[:, :7][insert_pre_1_ids] = self.arm_hand_insertion_prepare_dof_pos_list[0][:7]
        self.cur_targets[:, :7][insert_pre_2_ids] = self.arm_hand_insertion_prepare_dof_pos_list[1][:7]
        self.cur_targets[:, 7:23][insert_pre_0_ids] = self.prev_targets[:, 7:23][insert_pre_0_ids]

        ##############################################
        ########     use_teleoperation        ########
        ##############################################
        if self.use_teleoperation == True:
            self.cur_targets[:, :7] = self.teleoperator.calc_teleop_targets(self)

        ##############################################
        ####            bc_act_label              ####
        ##############################################
        self.bc_act_label = unscale(self.cur_targets[:, 0:23],
                            self.arm_hand_dof_lower_limits[0:23],
                            self.arm_hand_dof_upper_limits[0:23])

        ##############################################
        ####      apply_teleoper_perturbation     ####
        ##############################################
        if self.apply_teleoper_perturbation:
            # IK control robotic arm
            pos_err = self.perturbation_pos * 0.15
            rot_err = self.perturbation_rot * 0.05
            dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
            delta = control_ik(self.jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :7], self.device, dpose, self.num_envs)
            targets = self.arm_hand_dof_pos[:, 0:7] + delta[:, :7]
            self.cur_targets[self.apply_teleoper_perturbation_env_id, :7] = targets[self.apply_teleoper_perturbation_env_id]

        self.cur_targets[:, :] = tensor_clamp(self.cur_targets[:, :],
                                                    self.arm_hand_dof_lower_limits[:],
                                                    self.arm_hand_dof_upper_limits[:])
        self.prev_targets[:, :] = self.cur_targets[:, :]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)
        # self.gym.clear_lines(self.viewer)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)

        # self.add_debug_lines(self.envs[0], self.extra_target_pos[0], self.extra_target_rot[0])

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.segmentation_target_pos[i], self.segmentation_target_rot[i])
                self.add_debug_lines(self.envs[i], self.hand_base_pos[i], self.hand_base_rot[i])

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 50, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 50, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 50, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])

    def camera_rgb_visulization(self, camera_tensors, env_id=0, is_depth_image=False):
        torch_rgba_tensor = camera_tensors[env_id].clone()
        camera_image = torch_rgba_tensor.cpu().numpy()
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)
        
        return camera_image

    def camera_segmentation_visulization(self, camera_tensors, camera_seg_tensors, segmentation_id=0, env_id=0, is_depth_image=False):
        torch_rgba_tensor = camera_tensors[env_id].clone()
        torch_seg_tensor = camera_seg_tensors[env_id].clone()
        torch_rgba_tensor[torch_seg_tensor != self.segmentation_id] = 0

        camera_image = torch_rgba_tensor.cpu().numpy()
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_BGR2RGB)

        return camera_image

    def compute_emergence_reward(self, camera_tensors, camera_seg_tensors, segmentation_id=0):
        for i in range(self.num_envs):
            torch_seg_tensor = camera_seg_tensors[i]
            self.emergence_pixel[i] = torch_seg_tensor[torch_seg_tensor == segmentation_id].shape[0]

        self.emergence_reward = (self.emergence_pixel - self.last_emergence_pixel) / 10
        self.last_emergence_pixel = self.emergence_pixel.clone()

    def compute_heap_movement_penalty(self, all_lego_brick_linvel, segmentation_id):
        # self.heap_movement_penalty = torch.where(abs(all_lego_brick_pos[:self.num_envs, :, 0] - 1) > 0.25,
        #                                     torch.where(abs(all_lego_brick_pos[:self.num_envs, :, 1]) > 0.35, torch.ones_like(all_lego_brick_pos[:self.num_envs, :, 0]), torch.zeros_like(all_lego_brick_pos[:self.num_envs, :, 0])), torch.zeros_like(all_lego_brick_pos[:self.num_envs, :, 0]))
        all_lego_brick_linvel[:, segmentation_id, :] = 0
        # self.heap_movement_penalty = torch.sum(self.heap_movement_penalty, dim=1, keepdim=False)
        self.heap_movement_penalty = torch.mean(torch.norm(all_lego_brick_linvel, p=2, dim=-1), dim=-1, keepdim=False) * 20
        
#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_hand_reward(
    spin_coef, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes, max_hand_reset_length: int, arm_contacts, segmentation_target_rot, extra_target_pos, extra_target_rot, 
    max_episode_length: float, object_pos, object_rot, object_angvel, target_pos, target_rot, segmentation_target_pos, hand_base_pos, emergence_reward, arm_hand_ff_pos, arm_hand_rf_pos, arm_hand_mf_pos, arm_hand_th_pos, heap_movement_penalty, segmentation_target_init_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, rotation_id: int, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool, segmentation_target_linvel, num_envs: int, z_unit_tensor, arm_hand_th_rot, x_unit_tensor, segmentation_target_init_rot, tvalue
):
    # Distance from the hand to the object
    # goal_dist = torch.norm(hand_base_pos - segmentation_target_pos, p=2, dim=-1)
    # dist_rew = goal_dist * dist_reward_scale

    arm_hand_finger_dist = (torch.norm(segmentation_target_pos - arm_hand_ff_pos, p=2, dim=-1) + torch.norm(segmentation_target_pos - arm_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(segmentation_target_pos - arm_hand_rf_pos, p=2, dim=-1) + 3 * torch.norm(segmentation_target_pos - arm_hand_th_pos, p=2, dim=-1))

    axis1 = quat_apply(segmentation_target_rot, z_unit_tensor)
    axis2 = z_unit_tensor
    dot1 = torch.bmm(axis1.view(num_envs, 1, 3), axis2.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1)  # alignment of forward axis for gripper
    lego_z_align_reward = (torch.sign(dot1) * dot1 ** 2)

    resets = torch.where(arm_hand_finger_dist <= -1, torch.ones_like(reset_buf), reset_buf)

    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    lego_move_dist = torch.norm(segmentation_target_pos - segmentation_target_init_pos, p=2, dim=-1)
    lego_move_dist = torch.where(progress_buf > 50, torch.zeros_like(lego_move_dist), lego_move_dist)

    lego_z_align_reward_rew = 1 - lego_z_align_reward

    success_rew = torch.where(segmentation_target_pos[:, 2]-segmentation_target_init_pos[:, 2] > 0.1,
                              torch.where(lego_z_align_reward > 0.6, torch.ones_like(lego_z_align_reward_rew), torch.zeros_like(lego_z_align_reward_rew)), torch.zeros_like(lego_z_align_reward_rew))

    arm_hand_finger_dist = (torch.norm(segmentation_target_pos - arm_hand_ff_pos, p=2, dim=-1) + torch.norm(segmentation_target_pos - arm_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(segmentation_target_pos - arm_hand_rf_pos, p=2, dim=-1) + 3 * torch.norm(segmentation_target_pos - arm_hand_th_pos, p=2, dim=-1))
    dist_rew = torch.exp(-2 * torch.clamp(arm_hand_finger_dist-0.5, 0, None)) * 0.1

    object_up_reward = torch.clamp(segmentation_target_pos[:, 2]-segmentation_target_init_pos[:, 2], 0, 0.2) * 100
    object_up_reward = torch.clamp(torch.where(arm_hand_finger_dist < 0.5, object_up_reward, torch.zeros_like(object_up_reward)), min=None, max=20)

    # reward = torch.exp(- 5 * torch.clamp(arm_hand_finger_dist-0.1, 0, None)) + tvalue

    # reward = torch.exp(- 5 * torch.clamp(arm_hand_finger_dist-0.1, 0, None))

    reward = dist_rew + object_up_reward

    # reward = torch.where(progress_buf > 75, reward * 2, reward)
    resets = torch.where(progress_buf >= 75, 
                torch.where(arm_hand_finger_dist >= 0.6, torch.ones_like(resets), resets), resets)
    
    # resets = torch.where(progress_buf >= 75, 
    #             torch.where(arm_hand_finger_dist >= 0.6, torch.ones_like(reset_buf), reset_buf), reset_buf)

    if reward[0] >= 0:
        print("reward: ", reward[0].item())
        print("success_rew: ",  success_rew[0].item())

    # Fall penalty: distance to the goal is larger than a threshold
    # Check env termination conditions, including maximum success number

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(timed_out, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, reset_goal_buf, progress_buf, successes, cons_successes


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