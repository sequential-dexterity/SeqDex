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

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask
import pytorch3d.transforms as transform
import torch.nn.functional as F

def xyzw_to_wxyz(quat):
    # holy****, isaacgym uses xyzw format. pytorch3d uses wxyz format.
    new_quat = quat.clone()
    new_quat[:, :1] = quat[:, -1:]
    new_quat[:, 1:] = quat[:, :-1]
    return new_quat


# Debug script: python ./isaacgymenvs/train.py test=False task=AllegroArmLeftContinuous pipeline=cpu
class AllegroArmMOAR(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.control_penalty_scale = self.cfg["env"]["controlPenaltyScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.m_lower = self.cfg["env"].get("m_low", 0.03)
        self.m_upper = self.cfg["env"].get("m_up", 0.3)

        # self.skill = self.cfg["env"].get("skill", "all")

        self.relative_scale = self.cfg["env"].get("relScale", 0.5)

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
        self.rotation_axis = self.cfg["env"]["axis"]

        # Jan11: Previous single object version...
        # self.randomize_friction_lower = 0.2
        # self.randomize_friction_upper = 3.0
        # self.randomize_mass_lower = 0.3
        # self.randomize_mass_upper = 3.0

        self.randomize_friction_lower = 0.2
        self.randomize_friction_upper = 3.0
        self.randomize_mass_lower = self.m_lower
        self.randomize_mass_upper = self.m_upper

        self.sensor_thresh = self.cfg["env"].get("sensorThresh", 0.8)
        self.sensor_noise = self.cfg["env"].get("sensorNoise", 0.2)

        if self.rotation_axis == "x":
            self.rotation_id = 0
        elif self.rotation_axis == "y":
            self.rotation_id = 1
        else:
            self.rotation_id = 2

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.use_prev_target = self.cfg["env"]["usePrevTarget"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.object_type = self.cfg["env"]["objectType"]
        self.spin_coef = self.cfg["env"].get("spin_coef", 1.0)
        self.aux_coef = self.cfg["env"].get("aux_coef", -0.1)
        self.contact_coef = self.cfg["env"].get("contact_coef", 1.0)
        self.main_coef = self.cfg["env"].get("main_coef", 1.0)
        self.vel_coef = self.cfg["env"].get("vel_coef", -0.3)
        self.torque_coef = self.cfg["env"].get("torque_coef", -0.01)
        self.work_coef = self.cfg['env'].get("work_coef", -0.0002)
        self.latency = self.cfg['env'].get("latency", 0.25)

        #self.obs_stack = self.cfg['env'].get('obs_stack', 4)

        print(self.torque_coef, self.work_coef)
        self.torque_control = self.cfg["env"].get("torqueControl", True)

        self.skill_step = self.cfg["env"].get("skill_step", 50)
        #assert self.object_type in ["block", "egg", "pen"]

        self.ignore_z = (self.object_type == "pen")

        self.robot_asset_files_dict = {
            "normal": "urdf/xarm6/xarm6_allegro_right_fsr_2023.urdf",
            "thick": "urdf/xarm6/xarm6_allegro_right_fsr_2023_thin.urdf",
            "large":  "urdf/xarm6/xarm6_allegro_right_fsr_2023.urdf"
        }
        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "obj0": "urdf/objects/obj0.urdf",
            "obj1": "urdf/objects/obj1.urdf",
            "obj2": "urdf/objects/obj2.urdf",
            "obj3": "urdf/objects/obj3.urdf",
            "obj4": "urdf/objects/obj4.urdf",
            "obj5": "urdf/objects/obj5.urdf",
            "obj6": "urdf/objects/obj6.urdf",
            "obj7": "urdf/objects/obj7.urdf",
            "obj8": "urdf/objects/obj8.urdf",
            "obj9": "urdf/objects/obj9.urdf",
            "obj10": "urdf/objects/obj10.urdf",
            "tless0": "urdf/objects/tless0.urdf",
            "final_obj2_cross": "urdf/objects/final_obj2_cross.urdf",
            "final_calibration": "urdf/objects/final_calibration.urdf",
            "final_obj9_irregular_cube": "urdf/objects/final_obj9_irregular_cube.urdf",
            "final_obj1": "urdf/objects/final_obj1.urdf",
            "final_obj14_cathead": "urdf/objects/final_obj14_cathead.urdf",
            "final_obj12_cube_extrude1": "urdf/objects/final_obj12_cube_extrude1.urdf",
            "final_obj3_flatcan": "urdf/objects/final_obj3_flatcan.urdf",
            "final_obj10_cube_stair": "urdf/objects/final_obj10_cube_stair.urdf",
            "final_obj11_cube_ir2": "urdf/objects/final_obj11_cube_ir2.urdf",
            "final_obj5_s": "urdf/objects/final_obj5_s.urdf",
            "final_obj4_t": "urdf/objects/final_obj4_t.urdf",
            "final_obj7_ball_flatter": "urdf/objects/final_obj7_ball_flatter.urdf",
            "final_obj6_ball": "urdf/objects/final_obj6_ball.urdf",
            "final_obj8_cylinder": "urdf/objects/final_obj8_cylinder.urdf",
            "final_obj13_cube_extrude2": "urdf/objects/final_obj13_cube_extrude2.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            "set_obj10_thin_block_corner": "urdf/objects/set_obj10_thin_block_corner.urdf",
            "set_obj11_cylinder": "urdf/objects/set_obj11_cylinder.urdf",
            "set_obj12_cylinder_corner": "urdf/objects/set_obj12_cylinder_corner.urdf",
            "set_obj13_irregular_block": "urdf/objects/set_obj13_irregular_block.urdf",
            "set_obj14_irregular_block_cross": "urdf/objects/set_obj14_irregular_block_cross.urdf",
            "set_obj15_irregular_block_time": "urdf/objects/set_obj15_irregular_block_time.urdf",
            "set_obj16_cylinder_axis": "urdf/objects/set_obj16_cylinder_axis.urdf",
            "set_obj1_regular_block": "urdf/objects/set_obj1_regular_block.urdf",
            "set_obj2_block": "urdf/objects/set_obj2_block.urdf",
            "set_obj3_block": "urdf/objects/set_obj3_block.urdf",
            "set_obj4_block": "urdf/objects/set_obj4_block.urdf",
            "set_obj5_block": "urdf/objects/set_obj5_block.urdf",
            "set_obj6_block_corner": "urdf/objects/set_obj6_block_corner.urdf",
            "set_obj7_block": "urdf/objects/set_obj7_block.urdf",
            "set_obj8_short_block": "urdf/objects/set_obj8_short_block.urdf",
            "set_obj9_thin_block": "urdf/objects/set_obj9_thin_block.urdf",
        }

        self.object_sets = {
            "0": ['block'], #['block', 'obj7', 'obj9', 'obj10'],
            "1": ['block', 'obj7', 'obj9'],
            "2": ['block', 'obj3', 'obj4', 'obj6', 'obj7', 'obj8', 'obj9', 'obj10'],
            "3": ['block', 'obj0', 'obj2', 'obj3', 'obj4', 'obj6', 'obj7', 'obj8', 'obj9', 'obj10'],
            "4": ['block', 'final_obj2_cross', 'final_obj1', 'final_obj3_flatcan', 'final_obj5_s', 'final_obj4_t'],
            "5": ['block', 'final_obj2_cross',
                  'final_obj9_irregular_cube',
                  'final_obj1',
                  #'final_obj14_cathead',
                  'final_obj12_cube_extrude1',
                  'final_obj10_cube_stair',
                  'final_obj11_cube_ir2',
                  #'final_obj7_ball_flatter',
                  #'final_obj6_ball',
                  'final_obj8_cylinder',
                  'final_obj13_cube_extrude2'],
            "6": ['block',
                  'final_obj9_irregular_cube',
                  'final_obj12_cube_extrude1',
                  'final_obj10_cube_stair',
                  'final_obj11_cube_ir2',
                  'final_obj13_cube_extrude2'],
            "7": ['block',
                  'final_obj9_irregular_cube',
                  'final_obj12_cube_extrude1',
                  'final_obj13_cube_extrude2'],
            "8": ['block',
                  'final_obj9_irregular_cube',
                  'final_obj13_cube_extrude2'],
            "9": ['block',
                  'final_obj9_irregular_cube'],

            "A": ['set_obj1_regular_block',
                  'set_obj4_block', 'set_obj2_block', 'set_obj5_block',
                  'set_obj7_block', 'set_obj3_block',],

            "B": ['set_obj12_cylinder_corner', 'set_obj11_cylinder', 'set_obj9_thin_block', 'set_obj1_regular_block',
                  'set_obj4_block', 'set_obj2_block', 'set_obj5_block', 'set_obj8_short_block', 'set_obj6_block_corner',
                  'set_obj7_block', 'set_obj3_block', 'set_obj10_thin_block_corner'],

            "C": ['set_obj10_thin_block_corner', 'set_obj11_cylinder', 'set_obj12_cylinder_corner',
                  'set_obj13_irregular_block', 'set_obj14_irregular_block_cross', 'set_obj15_irregular_block_time',
                  'set_obj16_cylinder_axis', 'set_obj1_regular_block', 'set_obj2_block', 'set_obj3_block',
                  'set_obj4_block', 'set_obj5_block', 'set_obj6_block_corner',
                  'set_obj7_block', 'set_obj8_short_block', 'set_obj9_thin_block']
        }
        self.object_set_id = self.cfg["env"].get("objSet", "0")

        self.used_training_objects = self.object_sets[str(self.object_set_id)]#['block', 'obj7', 'obj9', 'obj10'] #['block', 'obj6', 'obj7', 'obj8', 'obj9', 'obj10']
        self.num_training_objects = len(self.used_training_objects)

        self.obj_init_pos_shift = {
            "org": (0.56, 0.0, 0.36),
            "new": (0.66, -0.01, 0.40)
        }

        self.obj_init_type = self.cfg["env"].get("objInit", "org")

        self.init_hand_qpos_override_dict = {
            "default" : {
                "joint_0.0": 0.0,
                "joint_1.0": 0.0,
                "joint_2.0": 0.0,
                "joint_3.0": 0.0,
                "joint_12.0": 1.3815,
                "joint_13.0": 0.0868,
                "joint_14.0": 0.1259,
                "joint_15.0": 0.0,
                "joint_4.0": 0.0048,
                "joint_5.0": 0.0,
                "joint_6.0": 0.0,
                "joint_7.0": 0.0,
                "joint_8.0": 0.0,
                "joint_9.0": 0.0,
                "joint_10.0": 0.0,
                "joint_11.0": 0.0
            },
            "thumb_up": {
                "joint_12.0": 1.3815,
                "joint_13.0": 0.0868,
                "joint_14.0": 0.1259
            },

            "stable": {
                "joint_0.0": 0.0261,
                "joint_1.0": 0.5032,
                "joint_2.0": 0.0722,
                "joint_3.0": 0.7050,
                "joint_12.0": 0.8353,
                "joint_13.0": -0.0388,
                "joint_14.0": 0.3703,
                "joint_15.0": 0.3444,
                "joint_4.0": 0.0048,
                "joint_5.0": 0.6514,
                "joint_6.0": -0.0147,
                "joint_7.0": 0.4276,
                "joint_8.0": -0.0868,
                "joint_9.0": 0.4106,
                "joint_10.0": 0.3233,
                "joint_11.0": 0.2792
            }
        }


        self.hand_init_type = self.cfg["env"].get("handInit", "default")
        self.hand_qpos_init_override = self.init_hand_qpos_override_dict[self.hand_init_type]

        assert self.obj_init_type in self.obj_init_pos_shift

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])

        # can be "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["full_no_vel", "full", "full_state", "full_contact", "partial_contact", "partial_stack"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.palm_name = "palm"
        self.contact_sensor_names = ["link_1.0_fsr", "link_2.0_fsr", "link_3.0_tip_fsr", "link_5.0_fsr",
                                     "link_6.0_fsr", "link_7.0_tip_fsr", "link_9.0_fsr", "link_10.0_fsr",
                                     "link_11.0_tip_fsr", "link_14.0_fsr", "link_15.0_fsr", "link_15.0_tip_fsr",
                                     "link_0.0_fsr", "link_4.0_fsr", "link_8.0_fsr", "link_13.0_fsr"]

        self.arm_sensor_names = ["link1", "link2", "link3", "link4", "link5", "link6"]

        self.n_stack = self.cfg['env'].get('obs_stack', 4)
        self.n_obs_dim = 85

        self.num_obs_dict = {
            "full_no_vel": 50,
            "full": 72,
            "full_state": 88,
            "full_contact": 93,
            "partial_contact": 45+16+24,
            "partial_stack": (45+16+24) * self.n_stack,
        }

        self.reward_mode = self.cfg["env"].get("rewardType", "free")
        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        self.robot_stiffness = self.cfg["env"].get("robotStiffness", 10.0)

        num_states = 0
        if self.asymmetric_obs:
            num_states = 101 + 24 + 49 + self.num_training_objects

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = 22

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.last_obs_buf = torch.zeros((self.num_envs, self.n_obs_dim), device=self.device, dtype=torch.float)

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        if self.viewer:
            self.debug_contacts = np.zeros((16, 50), dtype=np.float32)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # Zhaoheng: Add the contact!
        contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
        #     sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        #     self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

             dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
             self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_arm_hand_dofs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Zhaoheng: Contact.
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.spin_axis = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.arm_hand_default_dof_pos = torch.zeros(self.num_arm_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_arm_hand_dofs]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        # New spinning objectives...
        # self.all_spin_choices = torch.tensor([[1.0, 0.0, 0.0],
        #                                       [-1.0, 0.0, 0.0],
        #                                       [0.0, 1.0, 0.0],
        #                                       [0.0, -1.0, 0.0],
        #                                       [0.0, 0.0, 1.0],
        #                                       [0.0, 0.0, -1.0]], device=self.device)

        if self.rotation_axis == "all":
            self.all_spin_choices = torch.tensor([[1.0, 0.0, 0.0],
                                                  [0.0, -1.0, 0.0],
                                                  [0.0, 0.0, 1.0]], device=self.device)
        elif self.rotation_axis == "x":
            self.all_spin_choices = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)

        elif self.rotation_axis == "y":
            self.all_spin_choices = torch.tensor([[0.0, -1.0, 0.0]], device=self.device)

        elif self.rotation_axis == "z":
            self.all_spin_choices = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)

        else:
            assert False, "wrong spin axis"

        # Zhaoheng: Contact.
        self.contact_tensor = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1)

        print("Contact Tensor Dimension", self.contact_tensor.shape)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

        self.last_actions = torch.zeros((self.num_envs, 22), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.object_init_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.p_gain_val = 100.0
        self.d_gain_val = 4.0
        self.p_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.p_gain_val
        self.d_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.d_gain_val

        self.reset_goal_buf = self.reset_buf.clone()
        self.init_stack_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.last_contacts = torch.zeros((self.num_envs, 16), dtype=torch.float, device=self.device)
        self.contact_thresh = torch.zeros((self.num_envs, 16), dtype=torch.float, device=self.device)
        # for the sim2real debugging use
        self.debug_target = []
        self.debug_qpos = []

        self.post_init()

    def get_internal_state(self):
        return self.root_state_tensor[self.object_indices, 3:7]

    def get_internal_info(self, key):
        if key == 'target':
            return self.debug_target
        elif key == 'qpos':
            return self.debug_qpos
        elif key == 'contact':
            return self.finger_contacts
        return None
        # return {'target': self.debug_target, 'gt': self.debug_qpos}

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.create_object_asset_dict(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets'))

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def create_object_asset_dict(self, asset_root):
        self.object_asset_dict = {}
        print("ENTER ASSET CREATING!")
        for used_objects in self.used_training_objects:
            object_asset_file = self.asset_files_dict[used_objects]
            object_asset_options = gymapi.AssetOptions()
            # object_asset_options.vhacd_enabled = True
            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

            object_asset_options.disable_gravity = True

            goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

            self.object_asset_dict[used_objects] = {'obj': object_asset, 'goal': goal_asset}

            self.object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
            print("ORB_COUNT", self.object_rb_count)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')

        arm_hand_asset_file = self.robot_asset_files_dict[self.cfg["env"]["sensor"]]

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)

        # load arm and hand.
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        arm_hand_asset = self.gym.load_asset(self.sim, asset_root, arm_hand_asset_file, asset_options)
        self.num_arm_hand_bodies = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        self.num_arm_hand_shapes = self.gym.get_asset_rigid_shape_count(arm_hand_asset)
        self.num_arm_hand_dofs = self.gym.get_asset_dof_count(arm_hand_asset)
        print("Num dofs: ", self.num_arm_hand_dofs)
        self.num_arm_hand_actuators = self.num_arm_hand_dofs #self.gym.get_asset_actuator_count(shadow_hand_asset)

        # Set up each DOF.
        self.actuated_dof_indices = [i for i in range(self.num_arm_hand_dofs)]

        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        self.arm_hand_dof_default_pos = []
        self.arm_hand_dof_default_vel = []

        robot_lower_qpos = []
        robot_upper_qpos = []

        robot_dof_props = self.gym.get_asset_dof_properties(arm_hand_asset)

        # Zhaoheng. This part is very important (damping)
        for i in range(22):
            robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
            if i < 6:
                robot_dof_props['velocity'][i] = 1.0
            else:
                robot_dof_props['velocity'][i] = 3.14

            print("Max effort: ",robot_dof_props['effort'][i])
            robot_dof_props['effort'][i] = 20.0

            robot_dof_props['friction'][i] = 0.1
            robot_dof_props['stiffness'][i] = 0  # self.robot_stiffness
            robot_dof_props['armature'][i] = 0.1

            if i < 6:
                robot_dof_props['damping'][i] = 100.0
            else:
                robot_dof_props['damping'][i] = 0.0 #0.2 Early version is 0.2
            robot_lower_qpos.append(robot_dof_props['lower'][i])
            robot_upper_qpos.append(robot_dof_props['upper'][i])

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.arm_hand_dof_lower_limits = to_torch(robot_lower_qpos, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(robot_upper_qpos, device=self.device)
        self.arm_hand_dof_lower_qvel = to_torch(-robot_dof_props["velocity"], device=self.device)
        self.arm_hand_dof_upper_qvel = to_torch(robot_dof_props["velocity"], device=self.device)

        print("DOF_LOWER_LIMITS", robot_lower_qpos)
        print("DOF_UPPER_LIMITS", robot_upper_qpos)

        # Set up default arm position.
        # Zhaoheng: We can set this to different positions...
        self.default_arm_pos = [0.00, 0.782, -1.4239, 3.2866, 2.459, -1.4822]

        # We may need to some constraint for the thumb....
        for i in range(self.num_arm_hand_dofs):
            if i < 6:
                self.arm_hand_dof_default_pos.append(self.default_arm_pos[i])
            else:
                self.arm_hand_dof_default_pos.append(0.0)
            self.arm_hand_dof_default_vel.append(0.0)

        self.arm_hand_dof_default_pos = to_torch(self.arm_hand_dof_default_pos, device=self.device)
        self.arm_hand_dof_default_vel = to_torch(self.arm_hand_dof_default_vel, device=self.device)

        # Put objects in the scene.
        arm_hand_start_pose = gymapi.Transform()
        arm_hand_start_pose.p = gymapi.Vec3(0, 0.0, 0.0)
        arm_hand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()

        pose_dx, pose_dy, pose_dz = self.obj_init_pos_shift[self.obj_init_type]
        object_start_pose.p.x = arm_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = arm_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = arm_hand_start_pose.p.z + pose_dz
        object_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        if self.object_type == "pen":
            object_start_pose.p.z = arm_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(0.2, 0.2, 0.2)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement


        # compute aggregate size
        max_agg_bodies = self.num_arm_hand_bodies + 2
        max_agg_shapes = self.num_arm_hand_shapes + 2

        self.arm_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        self.object_class_indices = []
        # self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]

        arm_hand_rb_count = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        self.object_rb_handles = list(range(arm_hand_rb_count, arm_hand_rb_count + self.object_rb_count))

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

            # create fingertip force-torque sensors
            # if self.obs_type == "full_state" or self.asymmetric_obs:
            #     for ft_handle in self.fingertip_handles:
            #         env_sensors = []
            #         env_sensors.append(self.gym.create_force_sensor(env_ptr, ft_handle, sensor_pose))
            #         self.sensors.append(env_sensors)

            #     self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)

            # add object
            obj_class_indice = np.random.randint(0, len(self.used_training_objects), 1)[0]
            select_obj = self.used_training_objects[obj_class_indice]
            object_handle = self.gym.create_actor(env_ptr, self.object_asset_dict[select_obj]['obj'], object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.object_class_indices.append(obj_class_indice)
            # Randomize the object scale
            obj_scale = np.random.uniform(0.9, 1.1)
            self.gym.set_actor_scale(env_ptr, object_handle, obj_scale)
            # Do some friction randomization
            rand_friction =  np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
            hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, arm_hand_actor)
            for p in hand_props:
                p.friction = rand_friction
            self.gym.set_actor_rigid_shape_properties(env_ptr, arm_hand_actor, hand_props)

            object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            for p in object_props:
                p.friction = rand_friction

            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)


            prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            for p in prop:
                # print(p)
                p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
            self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, self.object_asset_dict[select_obj]['goal'], goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            # Set up object...
            # if self.object_type != "block":
            #     self.gym.set_rigid_body_color(
            #         env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98)
            #     )
            #     self.gym.set_rigid_body_color(
            #         env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98)
            #     )

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.arm_hands.append(arm_hand_actor)

        # Acquire specific links.
        palm_handles = self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, self.palm_name)
        self.palm_indices = to_torch(palm_handles, dtype=torch.int64)

        sensor_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, sensor_name)
                          for sensor_name in self.contact_sensor_names]
        self.sensor_handle_indices = to_torch(sensor_handles, dtype=torch.int64)

        arm_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, sensor_name)
                          for sensor_name in self.arm_sensor_names]
        self.arm_handle_indices = to_torch(arm_handles, dtype=torch.int64)
        self.object_class_indices = to_torch(self.object_class_indices, dtype=torch.int64) # now we turn this to a one hot vector
        self.object_one_hot_vector = F.one_hot(self.object_class_indices, num_classes=self.num_training_objects).float()

        # override!
        self.hand_override_info = [(self.gym.find_actor_dof_handle(env_ptr, arm_hand_actor, finger_name), self.hand_qpos_init_override[finger_name]) for finger_name in self.hand_qpos_init_override]

        print("PALM", self.palm_indices, self.sensor_handle_indices)

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]


        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        # self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

    def post_init(self):
        all_qpos = {}

        arm_hand_dof_default_pos = []
        arm_hand_dof_default_vel = []
        for (idx, qpos) in self.hand_override_info:
            print("Hand QPos Overriding: Idx:{} QPos: {}".format(idx, qpos))
            self.arm_hand_default_dof_pos[idx] = qpos
            all_qpos[idx] = qpos

        for i in range(self.num_arm_hand_dofs):
            if i < 6:
                arm_hand_dof_default_pos.append(self.default_arm_pos[i])
            elif i in all_qpos:
                arm_hand_dof_default_pos.append(all_qpos[i])
            else:
                arm_hand_dof_default_pos.append(0.0)
            arm_hand_dof_default_vel.append(0.0)

        self.arm_hand_dof_default_pos = to_torch(arm_hand_dof_default_pos, device=self.device)
        self.arm_hand_dof_default_vel = to_torch(arm_hand_dof_default_vel, device=self.device)

    def compute_reward(self, actions):
        self.control_error = torch.norm(self.cur_targets - self.arm_hand_dof_pos, dim=1)

        # Lets do some calculation

        # Generate a normal vector to the spinning axis.
        tmp_vector = self.spin_axis + torch.ones_like(self.spin_axis)
        vector_1 = torch.cross(self.spin_axis, tmp_vector)
        vector_1 = torch.nn.functional.normalize(vector_1, dim=-1)

        # print("VEC1", vector_1)
        # Generate another vector to form a basis [spin_axis, v1, v2]
        vector_2 = torch.cross(self.spin_axis, vector_1)
        vector_2 = torch.nn.functional.normalize(vector_2, dim=-1)
        # print("VEC2", vector_2)
        # last_rotation_inverse = self.last_object_rot
        # last_rotation_inverse[:, 1:] = - last_rotation_inverse[:, 1:]
        #
        # vector_1_new = quat_apply(last_rotation_inverse, vector_1)
        # vector_1_new = quat_apply(self.object_rot, vector_1_new)
        inverse_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(self.last_object_rot)).transpose(1, 2)
        # f_m = transform.quaternion_to_matrix(self.last_object_rot)

        # print("VALIDATE", vector_1 - torch.bmm(f_m, torch.bmm(inverse_rotation_matrix, vector_1.unsqueeze(-1))).squeeze())
        forward_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(self.object_rot))


        vector_1_new = torch.bmm(inverse_rotation_matrix, vector_1.unsqueeze(-1))
        vector_1_new = torch.bmm(forward_rotation_matrix, vector_1_new).squeeze()
        # print("V11", vector_1[0], vector_1_new[0])

        rot_vec_coordinate_1 = (vector_1_new * vector_1).sum(-1).reshape(-1, 1)
        rot_vec_coordinate_2 = (vector_1_new * vector_2).sum(-1).reshape(-1, 1)
        rot_vec_coordinate_3 = (vector_1_new * self.spin_axis).sum(-1)

        dev_angle = 3.1415926 / 2 - torch.arccos(rot_vec_coordinate_3)
        dev_angle = torch.abs(dev_angle)

        # print(rot_vec_coordinate_1[:2], rot_vec_coordinate_2[:2])
        rot_vec = torch.cat((rot_vec_coordinate_1, rot_vec_coordinate_2), dim=-1)
        rot_vec = torch.nn.functional.normalize(rot_vec, dim=-1)

        inner_prod = rot_vec[:, 0]
        theta_sign = torch.sign(rot_vec[:, 1])
        theta = theta_sign * torch.arccos(inner_prod)
        # print("DA", dev_angle[:4], torch.arccos(inner_prod)[:4])

        torque_penalty = (self.torques ** 2).sum(-1)
        work_penalty = (torch.abs(self.torques) * torch.abs(self.dof_vel_finite_diff)).sum(-1)

        if self.reward_mode == 'new':
            self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], \
            self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward_new(
                torch.tensor(self.spin_coef).to(self.device),
                torch.tensor(self.aux_coef).to(self.device),
                torch.tensor(self.main_coef).to(self.device),
                torch.tensor(self.vel_coef).to(self.device),
                torch.tensor(self.torque_coef).to(self.device),
                torch.tensor(self.work_coef).to(self.device),
                torch.tensor(self.contact_coef).to(self.device),
                self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes,
                self.consecutive_successes,
                self.max_episode_length, self.object_pos, self.object_rot, self.object_init_quat, self.object_linvel, self.object_angvel,
                self.goal_pos, self.goal_rot, self.finger_contacts, self.contact_coef,
                self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.control_error,
                self.control_penalty_scale, self.actions, self.action_penalty_scale,
                self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty, self.spin_axis,
                theta * 20, dev_angle * 20, torque_penalty, work_penalty,
                self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
            )
        elif self.reward_mode == 'noclip':
            self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], \
            self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward_noclip(
                torch.tensor(self.spin_coef).to(self.device),
                torch.tensor(self.aux_coef).to(self.device),
                torch.tensor(self.main_coef).to(self.device),
                torch.tensor(self.vel_coef).to(self.device),
                torch.tensor(self.torque_coef).to(self.device),
                torch.tensor(self.work_coef).to(self.device),
                torch.tensor(self.contact_coef).to(self.device),
                self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes,
                self.consecutive_successes,
                self.max_episode_length, self.object_pos, self.object_rot, self.object_init_quat, self.object_linvel,
                self.object_angvel,
                self.goal_pos, self.goal_rot, self.finger_contacts, self.contact_coef,
                self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.control_error,
                self.control_penalty_scale, self.actions, self.action_penalty_scale,
                self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty, self.spin_axis,
                theta * 20, dev_angle * 20, torque_penalty, work_penalty,
                self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
            )
        else:
            assert False

        self.extras['consecutive_successes'] = self.consecutive_successes.mean()

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(
                direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))


        #print(self.rew_buf[0], self.object_angvel[0])#, self.angvel_estimate[0])

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # Zhaoheng.
        self.gym.refresh_net_contact_force_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        # print("STATE", self.rigid_body_states[:, self.palm_indices][:, 0:3])
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        # self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        # self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        if self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
            self.compute_full_state()
        elif self.obs_type == "full_contact":
            self.compute_contact_observations('full')
        elif self.obs_type == "partial_contact":
            self.compute_contact_observations('pc')
        elif self.obs_type == "partial_stack":
            self.compute_contact_observations('ps')
        else:
            print("Unknown observations type!")

        # if self.asymmetric_obs:
        #     self.compute_full_state(True)

    def compute_contact_observations(self, mode='full'):
        if mode == 'full':
            self.obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                self.arm_hand_dof_lower_limits,
                                                                self.arm_hand_dof_upper_limits)
            # self.obs_buf[:, 16:23] = self.goal_pose
            self.obs_buf[:, 0:6] = 0
            self.obs_buf[:, 22:23] = 0
            self.obs_buf[:, 23:45] = 0 #self.actions
            contacts = self.contact_tensor.reshape(-1, 48, 3)  # 39+27
            # print(torch.norm(contacts, dim=-1))
            contacts = torch.norm(contacts, dim=-1)
            contacts = torch.where(contacts >= 0.1, 1.0, 0.0)

            if self.viewer:
                self.debug_contacts = contacts.detach().cpu().numpy()

            self.obs_buf[:, 45:93] = contacts
            self.finger_contacts = contacts[:, self.sensor_handle_indices]

        elif mode == 'pc':
            if self.asymmetric_obs:
                self.states_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                       self.arm_hand_dof_lower_limits,
                                                                       self.arm_hand_dof_upper_limits)
                self.states_buf[:, self.num_arm_hand_dofs:2 * self.num_arm_hand_dofs] = self.vel_obs_scale * self.arm_hand_dof_vel
                self.states_buf[:, 2 * self.num_arm_hand_dofs:3 * self.num_arm_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

                obj_obs_start = 3 * self.num_arm_hand_dofs  # 66
                self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
                self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
                self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

                # goal_obs_start = obj_obs_start + 13  # 79
                # self.states_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
                # self.states_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot,
                #                                                                       quat_conjugate(self.goal_rot))

                # fingertip observations, state(pose and vel) + force-torque sensors
                # todo - add later
                # num_ft_states = 13 * self.num_fingertips  # 65
                # num_ft_force_torques = 6 * self.num_fingertips  # 30

                # fingertip_obs_start = obj_obs_start + 13  # 72
                # self.states_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
                # self.states_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
                #                 num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor

                # obs_end = 96 + 65 + 30 = 191
                # obs_total = obs_end + num_actions = 72 + 16 = 88
                obs_end = 79 #+ 22 = 101#fingertip_obs_start  # + num_ft_states + num_ft_force_torques
                self.states_buf[:, obs_end:obs_end + self.num_actions] = self.actions
                self.states_buf[:, obs_end + self.num_actions: obs_end + self.num_actions + 24] = self.spin_axis.repeat(1, 8)

            self.obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                self.arm_hand_dof_lower_limits,
                                                                self.arm_hand_dof_upper_limits)
            self.obs_buf[:, 0:6] = 0.0
            self.obs_buf[:, 6:22] += torch.randn_like(self.obs_buf[:, 6:22]) * 0.05
            # self.obs_buf[:, 16:23] = self.goal_pose


            self.obs_buf[:, 22:45] = 0#self.actions

            #print("CONTACT_SHAPE", self.contact_tensor.shape)
            contacts = self.contact_tensor.reshape(-1, 49, 3)  # 39+27
            contacts = contacts[:, self.sensor_handle_indices, :] # 12
            contacts = torch.norm(contacts, dim=-1)
            gt_contacts = torch.where(contacts >= 10.0, 1.0, 0.0).clone()

            # we use some randomized threshold.
            # threshold = 0.2 + torch.rand_like(contacts) * self.sensor_thresh
            contacts = torch.where(contacts >= self.contact_thresh, 1.0, 0.0)

            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < 0.25, 1, 0) # with 0.25 probability, the signal is lagged
            self.last_contacts = self.last_contacts * latency + contacts * (1 - latency)

            mask = torch.rand_like(self.last_contacts)
            mask = torch.where(mask < self.sensor_noise, 0.0, 1.0)

            # random mask out the signal.
            sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)

            # Do some data augmentation to the contact....

            if self.viewer:
                self.debug_contacts = contacts.detach().cpu().numpy()

            # Observation randomization.
            #self.last_obs_buf[:, 6:22] += torch.randn_like(self.last_obs_buf[:, 6:22]) * 0.05
            self.obs_buf[:, 45:61] = sensed_contacts
            self.obs_buf[:, 61:85] = self.spin_axis.repeat(1, 8)
            # print("OS", self.obs_buf[:, 60:84])
            self.finger_contacts = gt_contacts

        else:
            if self.asymmetric_obs:
                self.states_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                       self.arm_hand_dof_lower_limits,
                                                                       self.arm_hand_dof_upper_limits)
                self.states_buf[:, self.num_arm_hand_dofs:2 * self.num_arm_hand_dofs] = self.vel_obs_scale * self.arm_hand_dof_vel
                self.states_buf[:, 2 * self.num_arm_hand_dofs:3 * self.num_arm_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

                obj_obs_start = 3 * self.num_arm_hand_dofs  # 66
                self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
                self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
                self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

                # goal_obs_start = obj_obs_start + 13  # 79
                # self.states_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
                # self.states_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot,
                #                                                                       quat_conjugate(self.goal_rot))

                # fingertip observations, state(pose and vel) + force-torque sensors
                # todo - add later
                # num_ft_states = 13 * self.num_fingertips  # 65
                # num_ft_force_torques = 6 * self.num_fingertips  # 30

                # fingertip_obs_start = obj_obs_start + 13  # 72
                # self.states_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
                # self.states_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
                #                 num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor

                # obs_end = 96 + 65 + 30 = 191
                # obs_total = obs_end + num_actions = 72 + 16 = 88
                obs_end = 79 #+ 22 = 101#fingertip_obs_start  # + num_ft_states + num_ft_force_torques
                self.states_buf[:, obs_end:obs_end + self.num_actions] = self.actions
                self.states_buf[:, obs_end + self.num_actions: obs_end + self.num_actions + 24] = self.spin_axis.repeat(1, 8)

                all_contact = self.contact_tensor.reshape(-1, 49, 3).clone()
                all_contact = torch.norm(all_contact, dim=-1).float()
                all_contact = torch.where(all_contact >= 20.0, torch.ones_like(all_contact), all_contact / 20.0)
                self.states_buf[:, obs_end + self.num_actions + 24: obs_end + self.num_actions + 24 + 49] = all_contact
                self.states_buf[:, obs_end + self.num_actions + 24 + 49:
                                   obs_end + self.num_actions + 24 + 49 + self.num_training_objects] = self.object_one_hot_vector

            self.last_obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                self.arm_hand_dof_lower_limits,
                                                                self.arm_hand_dof_upper_limits)
            self.last_obs_buf[:, 0:6] = 0.0
            # self.obs_buf[:, 16:23] = self.goal_pose


            self.last_obs_buf[:, 22:45] = 0#self.actions

            #print("CONTACT_SHAPE", self.contact_tensor.shape)
            contacts = self.contact_tensor.reshape(-1, 49, 3).clone()  # 39+27
            contacts = contacts[:, self.sensor_handle_indices, :] # 12
            contacts = torch.norm(contacts, dim=-1)

            gt_contacts = torch.where(contacts >= 1.0, 1.0, 0.0).clone()

            # we use some randomized threshold.
            # threshold = 0.2 + torch.rand_like(contacts) * self.sensor_thresh
            contacts = torch.where(contacts >= self.contact_thresh, 1.0, 0.0)

            latency_samples = torch.rand_like(self.last_contacts)
            latency = torch.where(latency_samples < self.latency, 1, 0)  # with 0.25 probability, the signal is lagged
            self.last_contacts = self.last_contacts * latency + contacts * (1 - latency)

            mask = torch.rand_like(self.last_contacts)
            mask = torch.where(mask < self.sensor_noise, 0.0, 1.0)

            # random mask out the signal.
            sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)

            # Do some data augmentation to the contact....

            if self.viewer:
                self.debug_contacts = sensed_contacts.detach().cpu().numpy()

            self.last_obs_buf[:, 45:61] = sensed_contacts
            self.last_obs_buf[:, 61:85] = self.spin_axis.repeat(1, 8)

            # Observation randomization.
            self.last_obs_buf[:, 6:22] += torch.randn_like(self.last_obs_buf[:, 6:22]) * 0.05#75
            # we need to do randomization by ourselves....

            init_obs_ids = torch.where(self.init_stack_buf == 1)
            self.init_stack_buf[init_obs_ids] = 0
            self.obs_buf[init_obs_ids] = self.last_obs_buf[init_obs_ids].repeat(1, self.n_stack)
            self.obs_buf = torch.cat((self.last_obs_buf.clone(), self.obs_buf[:, :-self.n_obs_dim]), dim=-1)
            # print("OS", self.obs_buf[0])
            self.finger_contacts = gt_contacts

    def compute_full_observations(self, no_vel=False):
        if no_vel:
            '''
                Legacy code.
            '''
            self.obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                self.arm_hand_dof_lower_limits,
                                                                self.arm_hand_dof_upper_limits)

            self.obs_buf[:, 16:23] = self.object_pose
            self.obs_buf[:, 23:30] = self.goal_pose
            self.obs_buf[:, 30:34] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            # 3*self.num_fingertips = 15
            #self.obs_buf[:, 42:57] = self.fingertip_pos.reshape(self.num_envs, 15)

            self.obs_buf[:, 34:56] = self.actions

        else:
            self.obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                self.arm_hand_dof_lower_limits,
                                                                self.arm_hand_dof_upper_limits)
            self.obs_buf[:, self.num_arm_hand_dofs:2*self.num_arm_hand_dofs] = self.vel_obs_scale * self.arm_hand_dof_vel

            # 2*16 = 32 -16
            self.obs_buf[:, 32:39] = self.object_pose
            self.obs_buf[:, 39:42] = self.object_linvel
            self.obs_buf[:, 42:45] = self.vel_obs_scale * self.object_angvel

            self.obs_buf[:, 45:52] = self.goal_pose
            self.obs_buf[:, 52:56] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            # 13*self.num_fingertips = 65 4*13 = 52
            # self.obs_buf[:, 72:137] = self.fingertip_state.reshape(self.num_envs, 65)

            self.obs_buf[:, 56:78] = self.actions

    def compute_full_state(self, asymm_obs=False):
        if asymm_obs:
            self.states_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                      self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits)
            self.states_buf[:, self.num_arm_hand_dofs:2*self.num_arm_hand_dofs] = self.vel_obs_scale * self.arm_hand_dof_vel
            self.states_buf[:, 2*self.num_arm_hand_dofs:3*self.num_arm_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

            obj_obs_start = 3*self.num_arm_hand_dofs  # 48
            self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
            self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
            self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 61
            self.states_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
            self.states_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            # fingertip observations, state(pose and vel) + force-torque sensors
            # todo - add later
            # num_ft_states = 13 * self.num_fingertips  # 65
            # num_ft_force_torques = 6 * self.num_fingertips  # 30

            fingertip_obs_start = goal_obs_start + 11  # 72
            # self.states_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            # self.states_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
            #                 num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor

            # obs_end = 96 + 65 + 30 = 191
            # obs_total = obs_end + num_actions = 72 + 16 = 88
            obs_end = fingertip_obs_start #+ num_ft_states + num_ft_force_torques
            self.states_buf[:, obs_end:obs_end + self.num_actions] = self.actions
        else:
            self.obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                self.arm_hand_dof_lower_limits,
                                                                self.arm_hand_dof_upper_limits)
            self.obs_buf[:, self.num_arm_hand_dofs:2*self.num_arm_hand_dofs] = self.vel_obs_scale * self.arm_hand_dof_vel
            self.obs_buf[:, 2*self.num_arm_hand_dofs:3*self.num_arm_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

            obj_obs_start = 3*self.num_arm_hand_dofs  # 48
            self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
            self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
            self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 61
            self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
            self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            # fingertip observations, state(pose and vel) + force-torque sensors
            # todo - add later
            # num_ft_states = 13 * self.num_fingertips  # 65
            # num_ft_force_torques = 6 * self.num_fingertips  # 30

            fingertip_obs_start = goal_obs_start + 11  # 72
            # self.states_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            # self.states_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
            #                 num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor

            # obs_end = 96 + 65 + 30 = 191
            # obs_total = obs_end + num_actions = 72 + 16 = 88
            obs_end = fingertip_obs_start #+ num_ft_states + num_ft_force_torques
            self.obs_buf[:, obs_end:obs_end + self.num_actions] = self.actions

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1],
                                     self.x_unit_tensor[env_ids],
                                     self.y_unit_tensor[env_ids])

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset_spin_axis(self, env_ids):
        env_ids_torch = torch.tensor(env_ids,  device=self.device).long()
        # Reset the init_quat...
        self.object_init_quat[env_ids_torch] = self.root_state_tensor[self.object_indices[env_ids_torch], 3:7]

        # Reset the axis
        self.spin_axis[env_ids_torch] = self.all_spin_choices[torch.randint(0, int(self.all_spin_choices.size(0)),
                                                                            (len(env_ids), ))]
        return

    def update_controller(self):
        previous_dof_pos = self.arm_hand_dof_pos.clone()
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # Zhaoheng.
        self.gym.refresh_net_contact_force_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        if self.torque_control:
            dof_pos = self.arm_hand_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.dt
            self.dof_vel_finite_diff = dof_vel.clone()
            torques = self.p_gain * (self.cur_targets - dof_pos) - self.d_gain * dof_vel
            # print("TORQUE", torques[-1])
            self.torques = torques.clone() #torch.clip(torques, -0.5, 0.5).clone()
            #print(self.torques)
            # print(torch.max(self.torques[:, 6:], dim=-1))
            self.torques = torch.clip(self.torques, -20.0, 20.0)
            if self.debug_viz:
                self.debug_target.append(self.cur_targets[:, 6:].clone())
                self.debug_qpos.append(self.arm_hand_dof_pos[:, 6:].clone())
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            #print("CONTROL!", dof_pos[0], self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_arm_hand_dofs][..., 0][0])
        return

    def refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

    # default robot pose: [0.00, 0.782, -1.087, 3.487, 2.109, -1.415]
    def reset_idx(self, env_ids, goal_env_ids):
        # generate random values
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        for env_id in env_ids:
            env = self.envs[env_id]
            handle = self.gym.find_actor_handle(env, 'object')
            prop = self.gym.get_actor_rigid_body_properties(env, handle)
            for p in prop:
                p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
            self.gym.set_actor_rigid_body_properties(env, handle, prop)
            # self._update_priv_buf(env_id=env_id, name='obj_mass', value=prop[0].mass, lower=0, upper=0.2)

        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset spinning axis
        self.reset_spin_axis(env_ids)

        # reset contact
        self.contact_thresh[env_ids] = torch.rand_like(self.contact_thresh[env_ids]) * self.sensor_thresh + 1.0
        self.last_contacts[env_ids] = 0.0

        # reset the pd-gain.
        self.randomize_p_gain_lower = self.p_gain_val * 0.30
        self.randomize_p_gain_upper = self.p_gain_val * 0.60
        self.randomize_d_gain_lower = self.d_gain_val * 0.75
        self.randomize_d_gain_upper = self.d_gain_val * 1.05

        if self.obs_type == 'partial_stack':
            self.obs_buf[env_ids] = 0
            self.init_stack_buf[env_ids] = 1

        self.p_gain[env_ids] = torch_rand_float(
            self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions),
            device=self.device).squeeze(1)
        self.d_gain[env_ids] = torch_rand_float(
            self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions),
            device=self.device).squeeze(1)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        if self.obs_type == "full_contact":
            new_object_rot = randomize_rotation(torch.zeros_like(rand_floats[:, 3]),
                                                torch.zeros_like(rand_floats[:, 4]),
                                                self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        else:
            new_object_rot = randomize_rotation(torch.zeros_like(rand_floats[:, 3]),
                                                torch.zeros_like(rand_floats[:, 4]), self.x_unit_tensor[env_ids],
                                                self.y_unit_tensor[env_ids])

        if self.object_type == "pen":
            rand_angle_y = torch.tensor(0.3)
            new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
                                                    self.x_unit_tensor[env_ids],
                                                    self.y_unit_tensor[env_ids],
                                                    self.z_unit_tensor[env_ids])

        #print("NEW", new_object_rot)
        #new_rot = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 3:7])
        #new_rot[:, -1] = 1.0
        self.root_state_tensor[self.object_indices[env_ids], 3:7] =  new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                    * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        # reset shadow hand
        #delta_max = self.arm_hand_dof_upper_limits - self.arm_hand_dof_default_pos
        #delta_min = self.arm_hand_dof_lower_limits - self.arm_hand_dof_default_pos
        #rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5+self.num_arm_hand_dofs]

        # pos =  #+ self.reset_dof_pos_noise * rand_delta
        self.arm_hand_dof_pos[env_ids, :] = self.arm_hand_dof_default_pos
        self.arm_hand_dof_vel[env_ids, :] = self.arm_hand_dof_default_vel #+ \
            #self.reset_dof_vel_noise * rand_floats[:, 5+self.num_arm_hand_dofs:5+self.num_arm_hand_dofs*2]
        self.prev_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_dof_default_pos
        self.cur_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_dof_default_vel

        hand_indices = self.hand_indices[env_ids].to(torch.int32)

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        # print("CONSIDERING" env_ids)
        # print('DOF_SHAPE', self.arm_hand_dof_pos.shape)
        for env_id in env_ids:
            for (idx, qpos) in self.hand_override_info:
                #print("Hand QPos Overriding: Idx:{} QPos: {}".format(idx, qpos))
                # print("RESETING", env_id * self.num_arm_hand_dofs + idx)
                self.dof_state[env_id * self.num_arm_hand_dofs + idx, 0] = qpos

        # print(self.dof_state[:, 0])
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

        for env_id in env_ids:
            self.object_init_quat[env_id] = self.root_state_tensor[self.object_indices[env_id], 3:7]

    def pre_physics_step(self, actions):
        # print("GET", actions)
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

        self.last_object_rot = self.root_state_tensor[self.object_indices, 3:7].clone()
        self.actions = actions.clone().to(self.device)

        if self.debug_viz:
            self.debug_qpos = []
            self.debug_target = []

        if self.use_relative_control:
            # if self.use_prev_target:
            #     targets = self.prev_targets[:,
            #               self.actuated_dof_indices] + 0.5 * self.actions
            # else:
            self.actions = self.actions * self.act_moving_average + self.last_actions * (1.0 - self.act_moving_average)
            targets = self.arm_hand_dof_pos + self.relative_scale * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.arm_hand_dof_lower_limits[
                                                                              self.actuated_dof_indices],
                                                                          self.arm_hand_dof_upper_limits[
                                                                              self.actuated_dof_indices])
            self.last_actions = self.actions.clone().to(self.device)

        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
                                                                   self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                   self.arm_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,self.actuated_dof_indices] + \
                                                             (1.0 - self.act_moving_average) * self.prev_targets[:,self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.actuated_dof_indices],
                self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
                self.arm_hand_dof_upper_limits[self.actuated_dof_indices])


        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        if self.force_scale > 0.0:
            # print("Apply force!")
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # # apply new forces
            # force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            # # print("Apply", force_indices)
            # self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
            #     self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device) * self.object_rb_masses * self.force_scale
            #
            # self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

            # New
            obj_mass = to_torch(
                [self.gym.get_actor_rigid_body_properties(env, self.gym.find_actor_handle(env, 'object'))[0].mass for
                 env in self.envs], device=self.device)
            prob = 0.25 #self.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * obj_mass[force_indices, None] * self.force_scale
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        #print("ERROR", self.arm_hand_dof_pos - self.cur_targets)
        # print("ACTIONS", self.actions)

        # Now we disable this...
        # env_ids = list(torch.where(torch.rand(self.num_envs) < 1 / self.skill_step)[0])  # On average: 50 steps.
        # #print("RST", env_ids)
        # self.reset_spin_axis(env_ids)

        if self.debug_viz:
            self.debug_qpos = torch.stack(self.debug_qpos, dim=0)
            self.debug_target = torch.stack(self.debug_target, dim=0) # [control_steps, num_envs, 16]

        # print("ANG", self.object_angvel[:4, 0], self.object_angvel[:4, 1], self.object_angvel[:4, 2])
        # env_ids = torch.where(self.progress_buf % 100 == 0)[0]
        # self.reset_spin_axis(list(env_ids))
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            main_vector = self.spin_axis.clone()
            inverse_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(self.object_init_quat))# .transpose(1, 2)
            inverse_rotation_matrix = inverse_rotation_matrix.permute(0, 2, 1)
            # print(self.object_init_quat[-1], self.object_rot[-1])
            forward_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(self.object_rot))

            inverse_main_vector = torch.bmm(inverse_rotation_matrix, main_vector.unsqueeze(-1))
            current_main_vector = torch.bmm(forward_rotation_matrix, inverse_main_vector).squeeze()#.cpu().numpy()

            for i in range(self.num_envs):
                # targetx = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # targety = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # targetz = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
                #
                # p0 = self.goal_pos[i].cpu().numpy() + self.goal_displacement_tensor.cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                objectm = (self.object_pos[i] + current_main_vector[i]).cpu().numpy()
                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2], objectm[0], objectm[1], objectm[2]], [0.85, 0.1, 0.85])

                # Also we render the angular velocity..
                # angvel_estimate_norm = torch.nn.functional.normalize(self.angvel_estimate, dim=-1) * 2.0
                # self.gym.add_lines(self.viewer, self.envs[i],
                #                    1, [p0[0], p0[1], p0[2],
                #                        p0[0] + angvel_estimate_norm[i, 0].cpu().numpy(),
                #                        p0[1] + angvel_estimate_norm[i, 1].cpu().numpy(),
                #                        p0[2] + angvel_estimate_norm[i, 2].cpu().numpy()],
                #                    [0.85, 0.85, 0.1])

        # We do some debug visualization.
        if self.viewer:
            for env in range(len(self.envs)):
                for i, contact_idx in enumerate(list(self.sensor_handle_indices)):

                    if self.debug_contacts[env, i] > 0.0:
                        self.gym.set_rigid_body_color(self.envs[env], self.arm_hands[env],
                                                      contact_idx, gymapi.MESH_VISUAL_AND_COLLISION,
                                                      gymapi.Vec3(0.0, 1.0, 0.0))
                    else:
                        self.gym.set_rigid_body_color(self.envs[env], self.arm_hands[env],
                                                      contact_idx, gymapi.MESH_VISUAL_AND_COLLISION,
                                                      gymapi.Vec3(1.0, 0.0, 0.0))

                import math
                if math.fabs(float(self.spin_axis[env, 0])) > 0.0:
                    color = (0.0, 0.0, 1.0)
                elif math.fabs(float(self.spin_axis[env, 1])) > 0.0:
                    color = (0.0, 1.0, 0.0)
                else:
                    color = (1.0, 0.0, 0.0)

                for i, contact_idx in enumerate(list(self.arm_handle_indices)):
                    self.gym.set_rigid_body_color(self.envs[env], self.arm_hands[env],
                                                  contact_idx, gymapi.MESH_VISUAL_AND_COLLISION,
                                                  gymapi.Vec3(*color))


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_hand_reward_new(
    spin_coef, aux_coef, main_coef, vel_coef, torque_coef, work_coef, contact_coef,
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, object_init_rot, object_linvel, object_angvel, target_pos, target_rot,
    finger_contacts, contact_scale: float, dist_reward_scale: float, rot_reward_scale: float,  rot_eps: float,
    control_error, control_penalty_scale: float, actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, main_vector, spinned_theta, dev_theta, torque_penalty, work_penalty,
    max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # Main vector specifies the spinning_axis
    # main_vector = torch.zeros(object_rot.size(0), 3).to(object_rot.device)
    # main1_vector[:, rotation_id] = 1.0

    inverse_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(object_init_rot)).transpose(1, 2)
    forward_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(object_rot))

    inverse_main_vector = torch.bmm(inverse_rotation_matrix, main_vector.unsqueeze(-1))
    current_main_vector = torch.bmm(forward_rotation_matrix, inverse_main_vector).squeeze()
    angle_difference = torch.arccos(torch.sum(main_vector * current_main_vector, dim=-1)) # The cosine similarity.
    #angle_difference = torch.where(angle_difference < 3.1415 * 0.23, torch.zeros_like(angle_difference), angle_difference)
    #angle_rew = - angle_difference

    object_angvel_norm = torch.norm(object_angvel, dim=-1)

    # If the object is not rotating, we will penalize it.
    #angle_difference_clip = torch.where(object_angvel_norm >= 0.2, angle_difference, -0.5 * torch.ones_like(angle_difference))
    #angle_difference = torch.minimum(angle_difference, angle_difference_clip)
    # print(angle_difference[:3])

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    # dist_rew = goal_dist * dist_reward_scale
    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    # Old reward. Dec 26.
    #print("ANGLE", object_angvel)
    # spin_reward = torch.sum(object_angvel * main_vector, dim=-1) #[:, rotation_id]
    # aux_spin_component = object_angvel - torch.sum(object_angvel * (main_vector), dim=-1, keepdim=True) * (main_vector)
    # aux_spin_reward = torch.norm(aux_spin_component, dim=-1)
    #
    # spin_reward = torch.where(spin_reward < 0.0, spin_reward * main_coef, spin_reward)
    # spin_reward = torch.clip(spin_reward, -10.0, 5.0)
    # # spin_reward = torch.where(spin_reward < 2.0, torch.zeros_like(spin_reward), spin_reward - 2.0)
    # spin_reward = spin_coef * spin_reward #+ aux_coef * aux_spin_reward#- 2.0 * object_angvel[:, 1:3].sum(dim=-1)
    #
    spinned_theta = torch.clip(spinned_theta, -3.14, 3.14)
    spin_reward = spin_coef * spinned_theta #- spin_coef * dev_theta * main_coef
    vel_reward = vel_coef * torch.norm(object_linvel, dim=-1)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    finger_contact_sum = finger_contacts.sum(dim=-1).float()
    finger_contact_sum = torch.clip(finger_contact_sum, 0.0, 5.0)

    # The hand must hold the object. Otherwise it is penalized.
    contact_reward = finger_contact_sum * contact_coef #torch.where(finger_contact_sum >= 6.0, 6.0, finger_contact_sum)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    #reward = dist_rew + rot_rew + action_penalty * action_penalty_scale
    reward = spin_reward + vel_reward + contact_reward + \
             torque_penalty * torque_coef + work_penalty * work_coef + \
             action_penalty * action_penalty_scale + control_error * control_penalty_scale
    # + contact_reward * contact_scale + \
    # print(spin_reward, angle_rew)

    # print("CONTROL ERROR:", control_error * control_penalty_scale)
    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) > 100.0, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    # reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)
    # reward = torch.where(angle_difference > 0.4 * 3.1415926, reward + fall_penalty, reward)
    # Looks like this is too hard.

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    #resets = torch.where(angle_difference > 0.6 * 3.1415926, torch.ones_like(reset_buf), resets)
    # print(angle_difference)
    resets = torch.where(angle_difference > 0.4 * 3.1415926, torch.ones_like(reset_buf), resets)
    # reward = torch.where(angle_difference > 0.5 * 3.1415926, reward + fall_penalty, reward)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_corand_floatsnsecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) > 100.0, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)

    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    # if max_consecutive_successes > 0:
    #     reward = torch.where(timed_out, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes

@torch.jit.script
def compute_hand_reward_noclip(
    spin_coef, aux_coef, main_coef, vel_coef, torque_coef, work_coef, contact_coef,
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, object_init_rot, object_linvel, object_angvel, target_pos, target_rot,
    finger_contacts, contact_scale: float, dist_reward_scale: float, rot_reward_scale: float,  rot_eps: float,
    control_error, control_penalty_scale: float, actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, main_vector, spinned_theta, dev_theta, torque_penalty, work_penalty,
    max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # Main vector specifies the spinning_axis
    # main_vector = torch.zeros(object_rot.size(0), 3).to(object_rot.device)
    # main1_vector[:, rotation_id] = 1.0

    inverse_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(object_init_rot)).transpose(1, 2)
    forward_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(object_rot))

    inverse_main_vector = torch.bmm(inverse_rotation_matrix, main_vector.unsqueeze(-1))
    current_main_vector = torch.bmm(forward_rotation_matrix, inverse_main_vector).squeeze()
    angle_difference = torch.arccos(torch.sum(main_vector * current_main_vector, dim=-1)) # The cosine similarity.
    #angle_difference = torch.where(angle_difference < 3.1415 * 0.23, torch.zeros_like(angle_difference), angle_difference)
    #angle_rew = - angle_difference

    object_angvel_norm = torch.norm(object_angvel, dim=-1)

    # If the object is not rotating, we will penalize it.
    #angle_difference_clip = torch.where(object_angvel_norm >= 0.2, angle_difference, -0.5 * torch.ones_like(angle_difference))
    #angle_difference = torch.minimum(angle_difference, angle_difference_clip)
    # print(angle_difference[:3])

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    # dist_rew = goal_dist * dist_reward_scale
    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    # Old reward. Dec 26.
    #print("ANGLE", object_angvel)
    # spin_reward = torch.sum(object_angvel * main_vector, dim=-1) #[:, rotation_id]
    # aux_spin_component = object_angvel - torch.sum(object_angvel * (main_vector), dim=-1, keepdim=True) * (main_vector)
    # aux_spin_reward = torch.norm(aux_spin_component, dim=-1)
    #
    # spin_reward = torch.where(spin_reward < 0.0, spin_reward * main_coef, spin_reward)
    # spin_reward = torch.clip(spin_reward, -10.0, 5.0)
    # # spin_reward = torch.where(spin_reward < 2.0, torch.zeros_like(spin_reward), spin_reward - 2.0)
    # spin_reward = spin_coef * spin_reward #+ aux_coef * aux_spin_reward#- 2.0 * object_angvel[:, 1:3].sum(dim=-1)
    #
    #spinned_theta = torch.clip(spinned_theta, -1.7, 1.5)
    spin_reward = spin_coef * spinned_theta #- spin_coef * dev_theta * main_coef
    vel_reward = vel_coef * torch.norm(object_linvel, dim=-1)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    finger_contact_sum = finger_contacts.sum(dim=-1).float()
    finger_contact_sum = torch.clip(finger_contact_sum, 0.0, 5.0)

    # The hand must hold the object. Otherwise it is penalized.
    contact_reward = finger_contact_sum * contact_coef #torch.where(finger_contact_sum >= 6.0, 6.0, finger_contact_sum)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    #reward = dist_rew + rot_rew + action_penalty * action_penalty_scale
    reward = spin_reward + vel_reward + contact_reward + \
             torque_penalty * torque_coef + work_penalty * work_coef + \
             action_penalty * action_penalty_scale + control_error * control_penalty_scale
    # + contact_reward * contact_scale + \
    # print(spin_reward, angle_rew)

    # print("CONTROL ERROR:", control_error * control_penalty_scale)
    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) > 100.0, torch.ones_like(reset_goal_buf), reset_goal_buf)
    # successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    # reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)
    # reward = torch.where(angle_difference > 0.4 * 3.1415926, reward + fall_penalty, reward)
    # Looks like this is too hard.

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    #resets = torch.where(angle_difference > 0.6 * 3.1415926, torch.ones_like(reset_buf), resets)
    # print(angle_difference)
    resets = torch.where(angle_difference > 0.4 * 3.1415926, torch.ones_like(reset_buf), resets)
    # reward = torch.where(angle_difference > 0.5 * 3.1415926, reward + fall_penalty, reward)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_corand_floatsnsecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) > 100.0, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)

    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    # if max_consecutive_successes > 0:
    #     reward = torch.where(timed_out, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot
