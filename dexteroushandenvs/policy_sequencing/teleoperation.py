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

class Teleoperator():
    def __init__(self, env, max_episode_length=10000000):
        # control xyzrpy
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_W, "W")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_S, "S")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_A, "A")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_D, "D")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_Q, "Q")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_E, "E")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_T, "T")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_G, "G")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_F, "F")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_H, "H")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R, "R")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_Y, "Y")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_L, "L")
        # switch policy
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_M, "M")

        env.max_episode_length = max_episode_length
        # env.target_euler = to_torch([0.0, 3.1415, 2.356], device=env.device).repeat((env.num_envs, 1))
        env.target_euler = to_torch([0.0, 3.1415, 1.571], device=env.device).repeat((env.num_envs, 1))

    def calc_teleop_targets(self, env):
        pos_err = to_torch([0.0, 0, 0.0], dtype=torch.float, device=env.device).repeat((env.num_envs, 1))
        
        for evt in env.gym.query_viewer_action_events(env.viewer):
            if evt.action == "W":
                pos_err[:, 2] += 0.1
            if evt.action == "S":
                pos_err[:, 2] -= 0.1
            if evt.action == "D":
                pos_err[:, 1] += 0.1
            if evt.action == "A":
                pos_err[:, 1] -= 0.1
            if evt.action == "Q":
                pos_err[:, 0] += 0.1
            if evt.action == "E":
                pos_err[:, 0] -= 0.1
            if evt.action == "T":
                env.target_euler[:, 2] += 0.1
            if evt.action == "G":
                env.target_euler[:, 2] -= 0.1
            if evt.action == "F":
                env.target_euler[:, 1] += 0.1
            if evt.action == "H":
                env.target_euler[:, 1] -= 0.1
            if evt.action == "R":
                env.target_euler[:, 0] += 0.1
            if evt.action == "Y":
                env.target_euler[:, 0] -= 0.1
            if evt.action == "L":
                env.reset_buf[:] = 1

            if evt.action == "":
                env.switch_policy = True

        # rot_err = torch.zeros_like(pos_err)
        target_rot = quat_from_euler_xyz(env.target_euler[:, 0], env.target_euler[:, 1], env.target_euler[:, 2])
        rot_err = orientation_error(target_rot, env.rigid_body_states[:, env.hand_base_rigid_body_index, 3:7].clone())
        dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
        delta = control_ik(env.jacobian_tensor[:, env.hand_base_rigid_body_index - 1, :, :7], env.device, dpose, env.num_envs)
        targets = env.arm_hand_dof_pos[:, 0:7] + delta[:, :7]

        env.progress_buf[:] = 0

        return targets

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