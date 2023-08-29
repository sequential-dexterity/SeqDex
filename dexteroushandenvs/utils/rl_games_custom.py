# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""
Custom classes that extend the RL Games functionality to integrate with IsaacGym
"""
import copy
import time
import ctypes
import numpy as np
from collections import Iterable

import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import autocast

from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
import rl_games.algos_torch.model_builder as mb
import rl_games.torch_runner as tr
from rl_games.common import common_losses, datasets
from rl_games.common.player import BasePlayer
from rl_games.common.tr_helpers import unsqueeze_obs
from rl_games.algos_torch import torch_ext, central_value, ppg_aux
from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common.a2c_common import ContinuousA2CBase, swap_and_flatten01, rescale_actions
from rl_games.algos_torch.players import PpoPlayerContinuous
# from oscar.policies.robot_policy import RobotArmPolicy
# from oscar.utils.obs_utils import DictConverter
# import oscar.utils.macros as macros
import gym

POLICY_CONTROLLER_MAPPING = {
    "robot_arm": RobotArmPolicy,
}

# Pre-calculated values for efficiency
LOG2PI = np.log(2.0 * np.pi)


class ContinuousNetworkBuilder(NetworkBuilder):
    """
    Generic network class that outputs continuous actions.
    """
    class Network(NetworkBuilder.BaseNetwork):
        """
        The following kwargs should be defined:

        num_seqs (int): Sequence length if using RNN
        input_shape (tuple): Shape of inputs into this network
        output_num (int): Size of outputs
        num_agents (int): Number of agents for this network
        distribution (None or str): If specified, output will be a distribution, rather than single value. Current
            valid options are {None, gaussian}
        """
        def __init__(
            self,
            params,
            input_shape,
            output_num,
            num_seqs=1,
            num_agents=1,
            distribution=None,

        ):
            self.num_seqs = num_seqs
            self.input_shape = input_shape
            self.output_num = output_num
            self.distribution = distribution
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            # Create networks
            self.net_cnn = nn.Sequential()
            self.net_mlp = nn.Sequential()

            if self.has_cnn:
                input_shape = torch_ext.shape_whc_to_cwh(input_shape)
                cnn_args = {
                    'ctype': self.cnn['type'],
                    'input_shape': input_shape,
                    'convs': self.cnn['convs'],
                    'activation': self.cnn['activation'],
                    'norm_func_name': self.normalization,
                }
                self.net_cnn = self._build_conv(**cnn_args)

            mlp_input_shape = self._calc_input_size(input_shape, self.net_cnn)

            if self.use_joint_obs_actions:
                use_embedding = self.joint_obs_actions_config['embedding']
                emb_size = self.joint_obs_actions_config['embedding_scale']
                mlp_out = mlp_input_shape // self.joint_obs_actions_config['mlp_scale']
                self.joint_actions = torch_ext.DiscreteActionsEncoder(output_num, mlp_out, emb_size, num_agents,
                                                                      use_embedding)
                mlp_input_shape = mlp_input_shape + mlp_out

            in_mlp_shape = mlp_input_shape
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    out_size = self.rnn_units
                    if self.rnn_concat_input:
                        rnn_in_size += in_mlp_shape
                else:
                    rnn_in_size = in_mlp_shape
                    in_mlp_shape = self.rnn_units

                self.net_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                if self.rnn_ln:
                    self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            mlp_args = {
                'input_size': in_mlp_shape,
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
                'd2rl': self.is_d2rl,
                'norm_only_first_layer': self.norm_only_first_layer
            }
            self.net_mlp = self._build_mlp(**mlp_args)

            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, output_num)
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, output_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                if self.distribution is None:
                    self.sigma = None
                    self.sigma_act = None
                elif self.distribution == "gaussian":
                    self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation'])
                    # Create multiple sigma inits if the values associated are a list
                    sigma_val = self.space_config['sigma_init'].get('val', None)
                    if isinstance(sigma_val, list):
                        _tmp_dicts = [
                            {k: v[i] if isinstance(v, list) else v for k, v in self.space_config['sigma_init'].items()}
                            for i in range(len(sigma_val))
                        ]
                        sigma_init = [self.init_factory.create(**_tmp_dict) for _tmp_dict in _tmp_dicts]
                    # Otherwise, we create the sigma as normal
                    else:
                        sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                    if self.space_config['fixed_sigma']:
                        self.sigma = nn.Parameter(torch.zeros(output_num, requires_grad=True, dtype=torch.float32),
                                                  requires_grad=True)
                    else:
                        self.sigma = torch.nn.Linear(out_size, output_num)
                else:
                    raise ValueError(f"Invalid distribution specified: {self.distribution}")

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.distribution == "gaussian":
                    if self.space_config['fixed_sigma']:
                        if isinstance(sigma_init, list):
                            # Custom per-element initialization
                            for i, _init in enumerate(sigma_init):
                                _init(self.sigma[i])
                        else:
                            sigma_init(self.sigma)
                    else:
                        if isinstance(sigma_init, list):
                            # Doesn't really make sense to have multiple values here, so we raise an error
                            raise ValueError("Multiple values for sigma initializer when using non fixed "
                                             "sigma is not accepted!")
                        else:
                            sigma_init(self.sigma.weight)

        @autocast(enabled=macros.MIXED_PRECISION)
        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)
            if self.has_cnn:
                # for obs shape 4
                # input expected shape (B, W, H, C)
                # convert to (B, C, W, H)
                if len(obs.shape) == 4:
                    obs = obs.permute((0, 3, 1, 2))

            out = obs
            out = self.net_cnn(out)
            out = out.flatten(1)

            if self.use_joint_obs_actions:
                actions = obs_dict['actions']
                actions_out = self.joint_actions(actions)
                out = torch.cat([out, actions_out], dim=-1)

            if self.has_rnn:
                out_in = out
                if not self.is_rnn_before_mlp:
                    out_in = out
                    out = self.net_mlp(out)
                    if self.rnn_concat_input:
                        out = torch.cat([out, out_in], dim=1)

                batch_size = out.size()[0]
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1)

                if len(states) == 1:
                    states = states[0]

                if self.rnn_name == 'sru':
                    out = out.transpose(0, 1)

                out, states = self.net_rnn(out, states)
                out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                if self.rnn_name == 'sru':
                    out = out.transpose(0, 1)
                if self.rnn_ln:
                    out = self.layer_norm(out)
                if self.is_rnn_before_mlp:
                    out = self.net_mlp(out)
                if type(states) is not tuple:
                    states = (states,)
            else:
                out = self.net_mlp(out)

            if self.is_discrete:
                logits = self.logits(out)
                return logits, None, states
            if self.is_multi_discrete:
                logits = [logit(out) for logit in self.logits]
                return logits, None, states
            if self.is_continuous:
                mu = self.mu_act(self.mu(out)) * self.out_scale
                if self.distribution is None:
                    sigma = None
                elif self.distribution == "gaussian":
                    if self.space_config['fixed_sigma']:
                        sigma = self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(out))
                return mu, sigma, states

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            if not self.has_rnn:
                return None

            num_layers = self.rnn_layers
            if self.rnn_name == 'identity':
                rnn_units = 1
            else:
                rnn_units = self.rnn_units
            if self.rnn_name == 'lstm':
                return (torch.zeros((num_layers, self.num_seqs, rnn_units)),
                        torch.zeros((num_layers, self.num_seqs, rnn_units)))
            else:
                return (torch.zeros((num_layers, self.num_seqs, rnn_units)),)

        def load(self, params):
            # settings
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.normalization = params.get('normalization', None)
            self.has_rnn = 'rnn' in params
            self.has_space = 'space' in params
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)
            self.use_joint_obs_actions = self.joint_obs_actions_config is not None

            if self.has_space:
                self.is_multi_discrete = 'multi_discrete' in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous' in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
                elif self.is_multi_discrete:
                    self.space_config = params['space']['multi_discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False

            self.out_scale = self.space_config['out_scale']

            # This configuration only works for continuous settings
            if self.is_discrete or self.is_multi_discrete or not self.is_continuous:
                raise ValueError("Continuous Network only supported for continuous action space!")

            if self.has_rnn:
                self.rnn_units = params['rnn']['units']
                self.rnn_layers = params['rnn']['layers']
                self.rnn_name = params['rnn']['name']
                self.rnn_ln = params['rnn'].get('layer_norm', False)
                self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
                self.rnn_concat_input = params['rnn'].get('concat_input', False)

            if 'cnn' in params:
                self.has_cnn = True
                self.cnn = params['cnn']
            else:
                self.has_cnn = False

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = ContinuousNetworkBuilder.Network(self.params, **kwargs)
        return net


class A2CControllerBuilder(A2CBuilder):
    """
    Custom A2C class that extends functionality to allow for a model-based controller to be included
    as part of the network.
    """
    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            input_shape = kwargs.pop('input_shape')
            self.input_shape = input_shape
            self.extrinsics_shape = kwargs.pop('extrinsics_shape')
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            command_num = self.policy_controller.input_dim
            actions_num = self.policy_controller.output_dim if self.differentiate_controller else command_num

            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()

            if self.has_cnn:
                input_shape = torch_ext.shape_whc_to_cwh(input_shape)
                cnn_args = {
                    'ctype': self.cnn['type'],
                    'input_shape': input_shape,
                    'convs': self.cnn['convs'],
                    'activation': self.cnn['activation'],
                    'norm_func_name': self.normalization,
                }
                self.actor_cnn = self._build_conv(**cnn_args)

                if self.separate:
                    self.critic_cnn = self._build_conv(**cnn_args)

            mlp_input_shape = self._calc_input_size(input_shape, self.actor_cnn)

            if self.use_joint_obs_actions:
                use_embedding = self.joint_obs_actions_config['embedding']
                emb_size = self.joint_obs_actions_config['embedding_scale']
                num_agents = kwargs.pop('num_agents')
                mlp_out = mlp_input_shape // self.joint_obs_actions_config['mlp_scale']
                self.joint_actions = torch_ext.DiscreteActionsEncoder(actions_num, mlp_out, emb_size, num_agents,
                                                                      use_embedding)
                mlp_input_shape = mlp_input_shape + mlp_out

            in_mlp_shape = mlp_input_shape
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    out_size = self.rnn_units
                    if self.rnn_concat_input:
                        rnn_in_size += in_mlp_shape
                else:
                    rnn_in_size = in_mlp_shape
                    in_mlp_shape = self.rnn_units

                if self.separate:
                    self.a_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    self.c_rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.a_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                        self.c_layer_norm = torch.nn.LayerNorm(self.rnn_units)
                else:
                    self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                    if self.rnn_ln:
                        self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            mlp_args = {
                'input_size': in_mlp_shape,
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
                'd2rl': self.is_d2rl,
                'norm_only_first_layer': self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            self.value = torch.nn.Linear(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            '''
                for multidiscrete actions num is a tuple
            '''
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, command_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation'])
                # Create multiple sigma inits if the values associated are a list
                sigma_val = self.space_config['sigma_init'].get('val', None)
                if isinstance(sigma_val, list):
                    _tmp_dicts = [
                        {k: v[i] if isinstance(v, list) else v for k, v in self.space_config['sigma_init'].items()}
                        for i in range(len(sigma_val))
                    ]
                    sigma_init = [self.init_factory.create(**_tmp_dict) for _tmp_dict in _tmp_dicts]
                # Otherwise, we create the sigma as normal
                else:
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.space_config['fixed_sigma']:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                                              requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.space_config['fixed_sigma']:
                    if isinstance(sigma_init, list):
                        # Custom per-element initialization
                        for i, _init in enumerate(sigma_init):
                            _init(self.sigma[i])
                    else:
                        sigma_init(self.sigma)
                else:
                    if isinstance(sigma_init, list):
                        # Doesn't really make sense to have multiple values here, so we raise an error
                        raise ValueError("Multiple values for sigma initializer when using non fixed "
                                         "sigma is not accepted!")
                    else:
                        sigma_init(self.sigma.weight)

        @autocast(enabled=macros.MIXED_PRECISION)
        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            seq_length = obs_dict.get('seq_length', 1)
            if self.has_cnn:
                # for obs shape 4
                # input expected shape (B, W, H, C)
                # convert to (B, C, W, H)
                if len(obs.shape) == 4:
                    obs = obs.permute((0, 3, 1, 2))

            if self.separate:
                a_out = c_out = obs
                a_out = self.actor_cnn(a_out)
                a_out = a_out.contiguous().view(a_out.size(0), -1)

                c_out = self.critic_cnn(c_out)
                c_out = c_out.contiguous().view(c_out.size(0), -1)

                if self.has_rnn:
                    if not self.is_rnn_before_mlp:
                        a_out_in = a_out
                        c_out_in = c_out
                        a_out = self.actor_mlp(a_out_in)
                        c_out = self.critic_mlp(c_out_in)

                        if self.rnn_concat_input:
                            a_out = torch.cat([a_out, a_out_in], dim=1)
                            c_out = torch.cat([c_out, c_out_in], dim=1)

                    batch_size = a_out.size()[0]
                    num_seqs = batch_size // seq_length
                    a_out = a_out.reshape(num_seqs, seq_length, -1)
                    c_out = c_out.reshape(num_seqs, seq_length, -1)

                    if self.rnn_name == 'sru':
                        a_out = a_out.transpose(0, 1)
                        c_out = c_out.transpose(0, 1)

                    if len(states) == 2:
                        a_states = states[0]
                        c_states = states[1]
                    else:
                        a_states = states[:2]
                        c_states = states[2:]
                    a_out, a_states = self.a_rnn(a_out, a_states)
                    c_out, c_states = self.c_rnn(c_out, c_states)

                    if self.rnn_name == 'sru':
                        a_out = a_out.transpose(0, 1)
                        c_out = c_out.transpose(0, 1)
                    else:
                        if self.rnn_ln:
                            a_out = self.a_layer_norm(a_out)
                            c_out = self.c_layer_norm(c_out)
                    a_out = a_out.contiguous().reshape(a_out.size()[0] * a_out.size()[1], -1)
                    c_out = c_out.contiguous().reshape(c_out.size()[0] * c_out.size()[1], -1)

                    if type(a_states) is not tuple:
                        a_states = (a_states,)
                        c_states = (c_states,)
                    states = a_states + c_states

                    if self.is_rnn_before_mlp:
                        a_out = self.actor_mlp(a_out)
                        c_out = self.critic_mlp(c_out)
                else:
                    a_out = self.actor_mlp(a_out)
                    c_out = self.critic_mlp(c_out)

                value = self.value_act(self.value(c_out))

                if self.is_discrete:
                    logits = self.logits(a_out)
                    return logits, value, states

                if self.is_multi_discrete:
                    logits = [logit(a_out) for logit in self.logits]
                    return logits, value, states

                if self.is_continuous:
                    mu = self.mu(a_out)
                    # If we're differentiating our controller, we also pass through the policy controller
                    if self.differentiate_controller:
                        mu = self.mu_act(self.policy_controller.get_control(control_dict=obs_dict['control_dict'], command=mu))
                    else:
                        # Otherwise, immediately pass through activation function (if any)
                        mu = self.mu_act(mu)
                    if self.space_config['fixed_sigma']:
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(a_out))

                    return mu, sigma, value, states
            else:
                out = obs
                out = self.actor_cnn(out)
                out = out.flatten(1)

                if self.use_joint_obs_actions:
                    actions = obs_dict['actions']
                    actions_out = self.joint_actions(actions)
                    out = torch.cat([out, actions_out], dim=-1)

                if self.has_rnn:
                    out_in = out
                    if not self.is_rnn_before_mlp:
                        out_in = out
                        out = self.actor_mlp(out)
                        if self.rnn_concat_input:
                            out = torch.cat([out, out_in], dim=1)

                    batch_size = out.size()[0]
                    num_seqs = batch_size // seq_length
                    out = out.reshape(num_seqs, seq_length, -1)

                    if len(states) == 1:
                        states = states[0]

                    if self.rnn_name == 'sru':
                        out = out.transpose(0, 1)

                    out, states = self.rnn(out, states)
                    out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                    if self.rnn_name == 'sru':
                        out = out.transpose(0, 1)
                    if self.rnn_ln:
                        out = self.layer_norm(out)
                    if self.is_rnn_before_mlp:
                        out = self.actor_mlp(out)
                    if type(states) is not tuple:
                        states = (states,)
                else:
                    out = self.actor_mlp(out)
                value = self.value_act(self.value(out))

                if self.central_value:
                    return value, states

                if self.is_discrete:
                    logits = self.logits(out)
                    return logits, value, states
                if self.is_multi_discrete:
                    logits = [logit(out) for logit in self.logits]
                    return logits, value, states
                if self.is_continuous:
                    mu = self.mu(out)
                    # If we're differentiating our controller, we also pass through the policy controller
                    if self.differentiate_controller:
                        mu = self.mu_act(
                            self.policy_controller.get_control(control_dict=obs_dict['control_dict'], command=mu))
                    else:
                        # Otherwise, immediately pass through activation function (if any)
                        mu = self.mu_act(mu)
                    if self.space_config['fixed_sigma']:
                        sigma = self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(out))
                    return mu, sigma, value, states

        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            if not self.has_rnn:
                return None

            num_layers = self.rnn_layers
            if self.rnn_name == 'identity':
                rnn_units = 1
            else:
                rnn_units = self.rnn_units
            if self.rnn_name == 'lstm':
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)),
                            torch.zeros((num_layers, self.num_seqs, rnn_units)),
                            torch.zeros((num_layers, self.num_seqs, rnn_units)),
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)),
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
            else:
                if self.separate:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)),
                            torch.zeros((num_layers, self.num_seqs, rnn_units)))
                else:
                    return (torch.zeros((num_layers, self.num_seqs, rnn_units)),)

        def load(self, params):
            # Load controller
            controller_params = copy.deepcopy(params['controller'])
            controller_cls = POLICY_CONTROLLER_MAPPING[controller_params.pop('type')]

            self.policy_controller = controller_cls(
                obs_dim=self.input_shape[-1],
                extrinsics_dim=self.extrinsics_shape[-1],
                **controller_params
            )
            self.differentiate_controller = self.policy_controller.differentiate_controller
            self.control_steps_per_policy_step = self.policy_controller.control_steps_per_policy_step

            # Other settings
            self.separate = params.get('separate', False)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_rnn = 'rnn' in params
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)
            self.use_joint_obs_actions = self.joint_obs_actions_config is not None

            if self.has_space:
                self.is_multi_discrete = 'multi_discrete' in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous' in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
                elif self.is_multi_discrete:
                    self.space_config = params['space']['multi_discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False

            # This configuration only works for continuous settings
            if self.is_discrete or self.is_multi_discrete or not self.is_continuous:
                raise ValueError("Controller Actor-Critic only supported for continuous action space!")

            if self.has_rnn:
                self.rnn_units = params['rnn']['units']
                self.rnn_layers = params['rnn']['layers']
                self.rnn_name = params['rnn']['name']
                self.rnn_ln = params['rnn'].get('layer_norm', False)
                self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
                self.rnn_concat_input = params['rnn'].get('concat_input', False)

            if 'cnn' in params:
                self.has_cnn = True
                self.cnn = params['cnn']
            else:
                self.has_cnn = False

    def build(self, name, **kwargs):
        net = A2CControllerBuilder.Network(self.params, **kwargs)
        return net


class ModelA2CContinuousLogStdController(ModelA2CContinuousLogStd):
    """
    Custom model class that extends the a2c continuous log std class to incorporate a controller component
    """
    def build(self, config):
        # Create NN
        net = self.network_builder.build('a2c', **config)

        # Print out NN structure
        for name, _ in net.named_parameters():
            print(name)

        # Create model
        return ModelA2CContinuousLogStdController.Network(net)

    class Network(nn.Module):
        def __init__(self, a2c_network):
            nn.Module.__init__(self)
            self.a2c_network = a2c_network

            # Determine whether to use tanh or not
            self.use_tanh = self.a2c_network.space_config['use_tanh']

            # Other variables for handling frozen case
            self._frozen = False
            self._deterministic_when_frozen = False

            # Store controller
            self.policy_controller = self.a2c_network.policy_controller
            self.differentiate_controller = self.policy_controller.differentiate_controller
            self.control_steps_per_policy_step = self.policy_controller.control_steps_per_policy_step

            assert self.control_steps_per_policy_step == 1, "Currently no support for multiple control steps per policy step ):"

        @property
        def action_dim(self):
            """
            Action dimension for this network. Could be DIFFERENT than the action dimension for the environment,
            since we may use a policy controller to convert this model's ouputted action into a deployable control
            action in the env

            Returns:
                int: Action dimension for this network
            """
            return self.policy_controller.output_dim if self.differentiate_controller else \
                self.policy_controller.input_dim

        def is_rnn(self):
            return self.a2c_network.is_rnn()

        def get_default_rnn_state(self):
            return self.a2c_network.get_default_rnn_state()

        @autocast(enabled=macros.MIXED_PRECISION)
        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            prev_pre_actions = input_dict.get('prev_pre_actions', None)
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, prev_pre_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result
            else:
                action = mu if self._frozen and self._deterministic_when_frozen else distr.sample()
                # Determine pre-/action based on whether we're using tanh or not
                if self.use_tanh:
                    selected_action, selected_pre_action = torch.tanh(action), action
                else:
                    selected_action, selected_pre_action = action, torch.zeros_like(action)     # dummy fill for pre_action
                neglogp = self.neglogp(selected_action, selected_pre_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : value,
                    'actions' : selected_action,
                    'pre_actions' : selected_pre_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result

        def neglogp(self, x, pre_x, mean, std, logstd):
            return _calc_neglogp(x=x, pre_x=pre_x, mean=mean, std=std, logstd=logstd, use_tanh=self.use_tanh)

        @property
        def policy_controller_info(self):
            """
            Returns keyword-mapped relevant info from the policy controller

            Returns:
                dict: Keyword-mapped relevant values from policy controller
            """
            return {
                "actor_loss_scale": self.policy_controller.actor_loss_scale,
            }

        @property
        def control_models(self):
            """
            Grabs policy controller's learned models as a dict

            Returns:
                dict: Keyward-mapped learned models owned by policy controller
            """
            return self.policy_controller.learned_models

        def prepare_control_models_for_train_loop(self):
            """
            Conduct any necessary steps to prepare for a training loop (e.g.: prepping a dataset) for all control models
            """
            for model in self.policy_controller.learned_models.values():
                model.prepare_for_train_loop()

        def get_control_models_loss(self, control_dict, **kwargs):
            """
            Iterates over all learned models in this policy controller and calculates loss for each model.

            Args:
                control_dict (dict): Keyword-mapped values relevant for controller computations
                kwargs (any): Any additional relevant keyword values for computing this loss

            Returns:
                dict: Keyword-mapped scalar tensor losses for each learned model owned by self.policy_controller
            """
            return {name: model.loss(control_dict=control_dict, **kwargs)
                    for name, model in self.policy_controller.learned_models.items()}

        def control_models_gradient_step(self, losses, retain_graph=False):
            """
            Iterates over all losses and runs a gradient step for each learned model owned by self.policy_controller

            Args:
                losses (dict): Keyword-mapped scalar tensor losses for each learned model owned
                    by self.policy_controller
                retain_graph (bool): If set, will retain computation graph when taking gradient step
            """
            for name, loss in losses.items():
                self.policy_controller.learned_models[name].gradient_step(loss, retain_graph=retain_graph)

        def _action_to_control(self, control_dict, command):
            """
            Passes the input @action through the policy controller

            Args:
                control_dict (dict): Dictionary of state tensors including relevant info for controller computation

                    Expected keys:
                        eef_state: shape of (N, 13), the (lin_pos, quat_ori, lin_vel, ang_vel) state of the eef body
                        q: shape of (N, N_dof), current joint positions
                        qd: shape of (N, N_dof), current joint velocities
                        mm: shape of (N, N_dof, N_dof), current mass matrix
                        j_eef: shape of (N, 6, N_dof), current jacobian matrix for end effector frame

                    Note that N_dof can be greater than control_dim; we assume the first control_dim indexes correspond to
                        the relevant elements to be used for the osc computations

                command (tensor): (N, N_cmd) raw action tensor outputted from model

            Returns:
                tensor: (N, N_act) action tensor processed by policy controller
                    (note: dimension may change by this process!)
            """
            return self.policy_controller.get_control(control_dict=control_dict, command=command)

        def get_policy_controller_goals(self):
            """
            Grabs the policy controller's current goals and returns them

            Returns:
                torch.tensor: (N, goal_dim) goal tensor
            """
            return self.policy_controller.controller.get_flattened_goals()

        def policy_action_to_control_action(self, control_dict, action):
            """
            Possibly passes the input @action through the policy controller, depending on whether
                self.differentiate_controller is True. Should be used by external object that owns this object
                to postprocess actions returned from this object's self.forward pass BEFORE deploying them in sim.

            Args:
                control_dict (dict): Dictionary of state tensors including relevant info for controller computation

                    Expected keys:
                        eef_state: shape of (N, 13), the (lin_pos, quat_ori, lin_vel, ang_vel) state of the eef body
                        q: shape of (N, N_dof), current joint positions
                        qd: shape of (N, N_dof), current joint velocities
                        mm: shape of (N, N_dof, N_dof), current mass matrix
                        j_eef: shape of (N, 6, N_dof), current jacobian matrix for end effector frame

                    Note that N_dof can be greater than control_dim; we assume the first control_dim indexes correspond to
                        the relevant elements to be used for the osc computations

                action (tensor): (N, N_cmd) policy action gathered from self.forward

            Returns:
                tensor: (N, N_act) action tensor to be deployed in sim
                    (note: dimension may change by this process!)
            """
            # Only pass through policy controller if we are NOT differentiating controller (i.e.: we didn't include
            # the controller during our forward pass)
            return action if self.differentiate_controller else \
                self._action_to_control(control_dict=control_dict, command=action)

        def control_models_pre_env_step(self, control_dict, train=False, **kwargs):
            """
            Runs any pre environment step computations necessary for each control model owned by the policy controller.

            Args:
                control_dict (dict): Keyword-mapped values relevant for controller computations
                train (bool): Whether we're currently training or in eval mode
                kwargs (any): Any additional relevant keyword values for computing values.

            Returns:
                dict: Keyword-mapped outputs from this model
            """
            ret = {}
            for model in self.policy_controller.learned_models.values():
                out = model.pre_env_step(control_dict=control_dict, train=train, **kwargs)
                ret.update(out)
                kwargs.update(out)

            return ret

        def control_models_post_env_step(self, control_dict, train=False, **kwargs):
            """
            Runs any post environment step computations necessary for each control model owned by the policy controller.

            Args:
                control_dict (dict): Keyword-mapped values relevant for controller computations
                train (bool): Whether we're currently training or in eval mode
                kwargs (any): Any additional relevant keyword values for computing values.

            Returns:
                dict: Keyword-mapped outputs from this model
            """
            ret = {}
            for model in self.policy_controller.learned_models.values():
                out = model.post_env_step(control_dict=control_dict, train=train, **kwargs)
                ret.update(out)
                kwargs.update(out)

            return ret

        def set_frozen(self, freeze=True, deterministic_when_frozen=True):
            """
            Sets this model to be frozen, and potentially deterministic as well.
            If frozen, train mode will not do anything.
            If deterministic, running forward pass will always produce mean as the output

            Args:
                freeze (bool): If True, will freeze this model
                deterministic_when_frozen (bool): If True, will not sample actions during the forward pass if
                    this model is frozen
            """
            self._frozen = freeze
            self._deterministic_when_frozen = deterministic_when_frozen

            # If we're frozen, we set all parameters belonging to this model with requires_grad=False
            if freeze:
                for param in self.parameters():
                    param.requires_grad = False

        def train(self, mode=True):
            # Parse mode, we might force eval based on settings
            mode = mode and not self._frozen
            # Run super method
            super().train(mode)
            # Also set policy controller to training mode
            self.policy_controller.train() if mode else self.policy_controller.eval()

        def eval(self):
            # Run super method
            super().eval()
            # Also set policy controller to training mode
            self.policy_controller.eval()


class A2CControllerAgent(A2CAgent):
    """
    Custom Continuous A2C class that extends base class to leverage a policy controller.
    """
    def __init__(self, base_name, config):
        """
        We unfortunately have to SKIP the immediate super method because we need to set the action space based on
        whether we're differentiating our controller or not
        """
        ContinuousA2CBase.__init__(self, base_name, config)

        # Set the L2 fetch granularity
        _libcudart = ctypes.CDLL('libcudart.so')
        # Set device limit on the current device
        # cudaLimitMaxL2FetchGranularity = 0x05
        pValue = ctypes.cast((ctypes.c_int * 1)(), ctypes.POINTER(ctypes.c_int))
        _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
        assert pValue.contents.value == 128

        # Set mixed precision macro
        macros.MIXED_PRECISION = self.mixed_precision

        # Get whether we should freeze this model or not
        self.freeze = self.config["freeze"]

        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        self.reset_envs()
        config = {
            'input_shape' : obs_shape,
            'extrinsics_shape': self.obs['control_dict']['extrinsics'].shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1)
        }

        self.model = self.network.build(config)
        self.model.to(self.ppo_device)

        # Set model to potentially be frozen
        self.model.set_frozen(freeze=self.freeze, deterministic_when_frozen=self.config['deterministic_when_frozen'])

        # Initialize states
        self.states = None

        # Overwrite actions based on network
        self.actions_num = self.model.action_dim
        self.env_info['action_space'] = gym.spaces.Box(low=0.0, high=1.0, shape=(self.model.action_dim,))

        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)

        # Compose all parameters that will be trained by this model
        self.params = list(self.model.parameters())

        # Iterate over all learned control models in case we also want to add them to this optimizer
        for model in self.model.control_models.values():
            if model.train_with_actor_loss:
                self.params += model.trainable_parameters

        self.optimizer = optim.Adam(self.params, float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)

        if self.normalize_input:
            if isinstance(self.observation_space,gym.spaces.Dict):
                self.running_mean_std = RunningMeanStdObs(obs_shape).to(self.ppo_device)
            else:
                self.running_mean_std = RunningMeanStd(obs_shape).to(self.ppo_device)

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape,
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device,
                'num_agents' : self.num_agents,
                'num_steps' : self.steps_num,
                'num_actors' : self.num_actors,
                'num_actions' : self.actions_num,
                'seq_len' : self.seq_len,
                'model' : self.central_value_config['network'],
                'config' : self.central_value_config,
                'writter' : self.writer,
                'multi_gpu' : self.multi_gpu
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)
        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)

        if 'phasic_policy_gradients' in self.config:
            self.has_phasic_policy_gradients = True
            self.ppg_aux_loss = ppg_aux.PPGAux(self, self.config['phasic_policy_gradients'])
        self.has_value_loss = (self.has_central_value \
                                and self.use_experimental_cv) \
                                or not self.has_phasic_policy_gradients
        self.algo_observer.after_init(self)

        # Register control models in env task
        for name, model in self.model.control_models.items():
            self.vec_env.env.task.register_control_model(name=name, model=model)

    def init_tensors(self):
        # Run super method
        super().init_tensors()

        # Add pre_action to update dict
        self.update_list += ['pre_actions']

        # In addition to pre_actions, also add control dict, goal states, and controls for
        # backpropping through controller to tensor list
        self.tensor_list += ['pre_actions', 'control_dicts', 'next_control_dicts', 'control_goals', 'controls']
        #
        # Get batch size
        batch_size = self.num_agents * self.num_actors
        #
        # # Add pre_action batch in case we're using tanh on the action output
        # self.mb_pre_actions = torch.zeros((self.steps_num, batch_size, self.actions_num), dtype=torch.float32,
        #                               device=self.ppo_device)
        # self.update_list.append('pre_action')
        # self.update_dict['pre_action'] = 'pre_actions'
        # self.tensors_dict['pre_action'] = self.mb_pre_actions
        #
        # self.mb_control_dicts = torch.zeros((self.steps_num, batch_size, self.control_dict_converter.flattened_dim),
        #                                     dtype=torch.float32, device=self.ppo_device)
        #
        # # If we're differentiating our controller, we also need to store the goal states for backpropping, so we create a buffer anyways
        # self.mb_control_goals = torch.zeros((self.steps_num, batch_size, self.model.policy_controller.goal_dim),
        #                                     dtype=torch.float32, device=self.ppo_device)
        #
        # # Store controls
        # self.mb_controls = torch.zeros((self.steps_num, batch_size, self.model.policy_controller.output_dim),
        #                                dtype=torch.float32, device=self.ppo_device)

        # Add tensor for storing pre actions
        self.experience_buffer.add_tensor(
            name="pre_actions",
            space=gym.spaces.Box(low=-1.0, high=1.0, shape=(self.actions_num,)),
        )

        # Add tensor for storing control dict info and goal info (the latter in case we're backpropping through the controller)
        self.control_dict_converter = DictConverter(self.obs['control_dict'])
        self.experience_buffer.add_tensor(
            name="control_dicts",
            space=gym.spaces.Box(low=-1.0, high=1.0, shape=(self.control_dict_converter.flattened_dim,)),
        )

        # Add tensor for storing final control dict info and goal info (resulting states occurring at the end of a sequence)
        self.experience_buffer.add_tensor(
            name="next_control_dicts",
            space=gym.spaces.Box(low=-1.0, high=1.0, shape=(self.control_dict_converter.flattened_dim,)),
        )

        # Add tensor for storing control goals
        self.experience_buffer.add_tensor(
            name="control_goals",
            space=gym.spaces.Box(low=-1.0, high=1.0, shape=(self.model.policy_controller.goal_dim,)),
        )

        # Add tensor and attribute for storing current controls
        self.experience_buffer.add_tensor(
            name="controls",
            space=gym.spaces.Box(low=-1.0, high=1.0, shape=(self.model.policy_controller.output_dim,)),
        )
        self._controls = torch.zeros((batch_size, self.model.policy_controller.output_dim),
            dtype=torch.float32, device=self.ppo_device)


        # self.control_gains_converter = DictConverter(self.model.policy_controller.)

    def train(self):
        # Reset env first so self.obs is populated
        self.obs = self.postprocess_obs(self.env_reset())

        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.time()
        total_time = 0
        rep_count = 0
        self.frame = 0
        self.obs = self.postprocess_obs(self.env_reset())

        if self.multi_gpu:
            self.hvd.setup_algo(self)

        while True:
            epoch_num = self.update_epoch()
            play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul, custom_stats = self.train_epoch()

            if self.multi_gpu:
                self.hvd.sync_stats(self)
            total_time += sum_time
            if self.rank == 0:
                scaled_time = sum_time  # self.num_agents * sum_time
                scaled_play_time = play_time  # self.num_agents * play_time
                curr_frames = self.curr_frames
                self.frame += curr_frames
                frame = self.frame
                if self.print_stats:
                    fps_step = curr_frames / scaled_play_time
                    fps_total = curr_frames / scaled_time
                    print(f'fps step: {fps_step:.1f} fps total: {fps_total:.1f}')

                self.writer.add_scalar('performance/total_fps', curr_frames / scaled_time, frame)
                self.writer.add_scalar('performance/step_fps', curr_frames / scaled_play_time, frame)
                self.writer.add_scalar('performance/update_time', update_time, frame)
                self.writer.add_scalar('performance/play_time', play_time, frame)
                self.writer.add_scalar('losses/a_loss', torch_ext.mean_list(a_losses).item(), frame)
                self.writer.add_scalar('losses/c_loss', torch_ext.mean_list(c_losses).item(), frame)
                self.writer.add_scalar('losses/entropy', torch_ext.mean_list(entropies).item(), frame)
                self.writer.add_scalar('info/last_lr', last_lr * lr_mul, frame)
                self.writer.add_scalar('info/lr_mul', lr_mul, frame)
                self.writer.add_scalar('info/e_clip', self.e_clip * lr_mul, frame)
                self.writer.add_scalar('info/kl', torch_ext.mean_list(kls).item(), frame)
                self.writer.add_scalar('info/epochs', epoch_num, frame)

                # Loop over all custom stats and add them to logging
                for stat, vals in custom_stats.items():
                    self.writer.add_scalar(f'custom/{stat}', torch_ext.mean_list(vals).item(), frame)

                self.algo_observer.after_print_stats(frame, epoch_num, total_time)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/frame'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/frame', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    task_name = self.config['name'].split("_")[0]

                    if self.save_freq > 0:
                        if (epoch_num % self.save_freq == 0) and (mean_rewards <= self.last_mean_rewards):
                            self.save(f"{self.logdir}/{task_name}/nn/" + 'last_' + self.config['name'] + 'ep=' + str(
                                epoch_num) + 'rew=' + str(mean_rewards))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(f"{self.logdir}/{task_name}/nn/" + self.config['name'])
                        if self.last_mean_rewards > self.config['score_to_win']:
                            print('Network won!')
                            self.save(
                                f"{self.logdir}/{task_name}/nn/" + self.config['name'] + 'ep=' + str(
                                    epoch_num) + 'rew=' + str(
                                    mean_rewards))
                            return self.last_mean_rewards, epoch_num

                if epoch_num > self.max_epochs:
                    task_name = self.config['name'].split("_")[0]
                    self.save(
                        f"{self.logdir}/{task_name}/nn/" + 'last_' + self.config['name'] + 'ep=' + str(
                            epoch_num) + 'rew=' + str(
                            mean_rewards))
                    print('MAX EPOCHS NUM!')
                    return self.last_mean_rewards, epoch_num
                update_time = 0

    def train_epoch(self):
        self.set_eval()
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()
        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []
        custom_stats = {}  # Maps custom keys to arrays of custom values

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())

        # Prepare control models for training
        self.model.prepare_control_models_for_train_loop()

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss, custom_stat = self.train_actor_critic(
                    self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                # Loop over all custom stats and add them
                for stat, val in custom_stat.items():
                    if stat not in custom_stats:
                        custom_stats[stat] = [val]
                    else:
                        custom_stats[stat].append(val)

                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)

                if self.schedule_type == 'legacy':
                    if self.multi_gpu:
                        kl = self.hvd.average_value(kl, 'ep_kls')
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef,
                                                                            self.epoch_num, 0, kl.item())
                    self.update_lr(self.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)

            if self.schedule_type == 'standard':
                if self.multi_gpu:
                    av_kls = self.hvd.average_value(av_kls, 'ep_kls')
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num,
                                                                        0, av_kls.item())
                self.update_lr(self.last_lr)
            kls.append(av_kls)

        if self.schedule_type == 'standard_epoch':
            if self.multi_gpu:
                av_kls = self.hvd.average_value(torch_ext.mean_list(kls), 'ep_kls')
            self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0,
                                                                    av_kls.item())
            self.update_lr(self.last_lr)

        if self.has_phasic_policy_gradients:
            self.ppg_aux_loss.train_net(self)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul, custom_stats

    def play_steps(self):
        epinfos = []
        update_list = self.update_list

        for n in range(self.steps_num):
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data('obses', n, self.obs['obs'])
            self.experience_buffer.update_data('dones', n, self.dones)
            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])
            # Add control dicts
            self.experience_buffer.update_data('control_dicts', n,
                                               self.control_dict_converter.to_tensor(self.obs['control_dict']))
            # # Add goals
            # self.experience_buffer.update_data('control_goals', n, self.model.get_policy_controller_goals())
            # Add controls
            self.experience_buffer.update_data('controls', n, self._controls)
            # Run any necessary control model pre-steps
            input_dict = {
                "control_dict": self.control_dict,
                "obs": self.obs,
                "control": self._controls,
                "commmand": res_dict["actions"],
            }
            cm_vals = self.model.control_models_pre_env_step(train=True, **input_dict)
            # Take environment step
            obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            # Always post process obs
            self.obs = self.postprocess_obs(obs)
            # Run any necessary control model post-steps
            input_dict.update({
                "next_control_dict": self.obs['control_dict'],
                "next_obs": self.obs['obs'],
                "done": self.dones,
            })
            input_dict.update(cm_vals)
            self.model.control_models_post_env_step(train=True, **input_dict)

            # Update next obs and control dicts
            self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
            self.experience_buffer.update_data('next_control_dicts', n,
                                               self.control_dict_converter.to_tensor(self.obs['control_dict']))

            # Run any post-env step needed for policy controller

            # Get rewards
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        if self.has_central_value and self.central_value_net.use_joint_obs_actions:
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                val_dict = self.get_masked_action_values(self.obs, masks)
            else:
                val_dict = self.get_action_values(self.obs)
            last_values = val_dict['values']
        else:
            last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        return batch_dict

    def play_steps_rnn(self):
        mb_rnn_states = []
        epinfos = []
        self.experience_buffer.tensor_dict['values'].fill_(0)
        self.experience_buffer.tensor_dict['rewards'].fill_(0)
        self.experience_buffer.tensor_dict['dones'].fill_(1)

        update_list = self.update_list

        batch_size = self.num_agents * self.num_actors
        mb_rnn_masks = None

        mb_rnn_masks, indices, steps_mask, steps_state, play_mask, mb_rnn_states = self.init_rnn_step(batch_size,
                                                                                                      mb_rnn_states)

        for n in range(self.steps_num):
            seq_indices, full_tensor = self.process_rnn_indices(mb_rnn_masks, indices, steps_mask, steps_state,
                                                                mb_rnn_states)
            if full_tensor:
                break
            if self.has_central_value:
                self.central_value_net.pre_step_rnn(self.last_rnn_indices, self.last_state_indices)

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            self.rnn_states = res_dict['rnn_states']
            self.experience_buffer.update_data_rnn('obses', indices, play_mask, self.obs['obs'])
            self.experience_buffer.update_data_rnn('dones', indices, play_mask, self.dones.byte())

            for k in update_list:
                self.experience_buffer.update_data_rnn(k, indices, play_mask, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data_rnn('states', indices[::self.num_agents],
                                                       play_mask[::self.num_agents] // self.num_agents,
                                                       self.obs['states'])

            # # Add goals
            # self.experience_buffer.update_data_rnn('control_goals', indices, play_mask,
            #                                        self.model.get_policy_controller_goals())
            # Add controls
            self.experience_buffer.update_data_rnn('controls', indices, play_mask, self._controls)
            # Add control dicts
            self.experience_buffer.update_data_rnn('control_dicts', indices, play_mask,
                                                   self.control_dict_converter.to_tensor(self.obs['control_dict']))

            # Run any necessary control model pre-steps
            input_dict = {
                "control_dict": self.control_dict,
                "obs": self.obs,
                "control": self._controls,
                "commmand": res_dict["actions"],
            }
            cm_vals = self.model.control_models_pre_env_step(train=True, **input_dict)

            # Take environment step
            obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            # Always post process obs
            self.obs = self.postprocess_obs(obs)

            # Run any necessary control model post-steps
            input_dict.update({
                "next_control_dict": self.obs['control_dict'],
                "next_obs": self.obs['obs'],
                "dones": self.dones,
            })
            input_dict.update(cm_vals)
            self.model.control_models_post_env_step(train=True, **input_dict)

            # Update next obs and control dicts
            self.experience_buffer.update_data_rnn('next_obses', indices, play_mask, self.obs['obs'])
            self.experience_buffer.update_data_rnn('next_control_dicts', indices, play_mask,
                                                   self.control_dict_converter.to_tensor(self.obs['control_dict']))

            # Get rewards
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data_rnn('rewards', indices, play_mask, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[::self.num_agents]

            self.process_rnn_dones(all_done_indices, indices, seq_indices)
            if self.has_central_value:
                self.central_value_net.post_step_rnn(all_done_indices)

            self.algo_observer.process_infos(infos, done_indices)

            fdones = self.dones.float()
            not_dones = 1.0 - self.dones.float()

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        if self.has_central_value and self.central_value_net.use_joint_obs_actions:
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                val_dict = self.get_masked_action_values(self.obs, masks)
            else:
                val_dict = self.get_action_values(self.obs)

            last_values = val_dict['value']
        else:
            last_values = self.get_values(self.obs)

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']

        non_finished = (indices != self.steps_num).nonzero(as_tuple=False)
        ind_to_fill = indices[non_finished]
        mb_fdones[ind_to_fill, non_finished] = fdones[non_finished]
        mb_values[ind_to_fill, non_finished] = last_values[non_finished]
        fdones[non_finished] = 1.0
        last_values[non_finished] = 0

        mb_advs = self.discount_values_masks(fdones, last_values, mb_fdones, mb_values, mb_rewards,
                                             mb_rnn_masks.view(-1, self.steps_num).transpose(0, 1))
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['rnn_states'] = mb_rnn_states
        batch_dict['rnn_masks'] = mb_rnn_masks
        batch_dict['played_frames'] = n * self.num_actors * self.num_agents

        return batch_dict

    def prepare_dataset(self, batch_dict):
        obses = batch_dict['obses']
        next_obses = batch_dict['next_obses']
        control_dicts = batch_dict['control_dicts']
        next_control_dicts = batch_dict['next_control_dicts']
        control_goals = batch_dict['control_goals']
        controls = batch_dict['controls']
        returns = batch_dict['returns']
        dones = batch_dict['dones']
        values = batch_dict['values']
        actions = batch_dict['actions']
        pre_actions = batch_dict['pre_actions']
        neglogpacs = batch_dict['neglogpacs']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)

        advantages = returns - values

        if self.normalize_value:
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)

        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['pre_actions'] = pre_actions
        dataset_dict['dones'] = dones
        dataset_dict['obs'] = obses
        dataset_dict['next_obs'] = next_obses
        dataset_dict['control_dicts'] = control_dicts
        dataset_dict['next_control_dicts'] = next_control_dicts
        dataset_dict['control_goals'] = control_goals
        dataset_dict['controls'] = controls
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas

        self.dataset.update_values_dict(dataset_dict)

        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['pre_actions'] = pre_actions
            dataset_dict['obs'] = batch_dict['states']
            dataset_dict['rnn_masks'] = rnn_masks
            self.central_value_net.update_dataset(dataset_dict)

    def obs_to_tensors(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self.cast_obs(value)
                # We also save control dict separately so we can use it later
                if key == "control_dict":
                    self.control_dict = value
        else:
            upd_obs = {'obs': self.cast_obs(obs)}
        return upd_obs

    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'prev_pre_actions': None,
            'obs': processed_obs,
            'control_dict': obs["control_dict"],
            'rnn_states': self.rnn_states
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states': states,
                    # 'actions' : res_dict['action'],
                    # 'rnn_states' : self.rnn_states
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        if self.normalize_value:
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    def get_values(self, obs):
        with torch.no_grad():
            if self.has_central_value:
                states = obs['states']
                self.central_value_net.eval()
                input_dict = {
                    'is_train': False,
                    'states': states,
                    'actions': None,
                    'pre_actions': None,
                    'is_done': self.dones,
                }
                value = self.get_central_value(input_dict)
            else:
                self.model.eval()
                processed_obs = self._preproc_obs(obs['obs'])
                input_dict = {
                    'is_train': False,
                    'prev_actions': None,
                    'prev_pre_actions': None,
                    'obs': processed_obs,
                    'control_dict': obs['control_dict'],
                    'rnn_states': self.rnn_states
                }
                result = self.model(input_dict)
                value = result['values']

            if self.normalize_value:
                value = self.value_mean_std(value, True)
            return value

    def preprocess_actions(self, actions):
        # Possibly pass these actions through the controller
        actions = self.model.policy_action_to_control_action(control_dict=self.control_dict, action=actions)
        clamped_actions = torch.clamp(actions, -1.0, 1.0)
        rescaled_actions = rescale_actions(self.actions_low, self.actions_high, clamped_actions)
        # Save controls before deploying
        self._controls = rescaled_actions
        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()
        return rescaled_actions

    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        pre_actions_batch = input_dict['pre_actions']
        dones_batch = input_dict['dones']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)
        next_obs_batch = input_dict['next_obs']
        next_obs_batch = self._preproc_obs(next_obs_batch)
        control_dicts_batch = input_dict['control_dicts']
        next_control_dicts_batch = input_dict['next_control_dicts']
        controls_batch = input_dict['controls']

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'prev_pre_actions': pre_actions_batch,
            'obs': obs_batch,
            'control_dict': self.control_dict_converter.to_dict(control_dicts_batch),
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            # Calculate actor-critic losses
            a_loss = common_losses.actor_loss(old_action_log_probs_batch, action_log_probs, advantage, self.ppo,
                                              curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch,
                                                   self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)

            b_loss = self.bound_loss(mu)

            # # Calculate additional custom losses
            # control_kwargs = {
            #     "control": controls_batch,
            # }
            # control_losses = self.model.get_control_models_loss(
            #     control_dict=batch_dict['control_dict'],
            #     **control_kwargs,
            # )

            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks
            )
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            # # Re-map control losses in dict
            # control_losses = {name: l for name, l in zip(control_losses.keys(), control_loss)}

            # # We inversely scale the actor loss if we're using delan
            # if "delan" in control_losses:
            #     scale = self.model.control_models["delan"].loss_threshold / control_losses["delan"].detach()
            #     a_loss = a_loss * torch.clip(scale, 0.0, 1.0)

            # Scale actor loss by specified value
            if a_loss > 0:
                a_loss = a_loss * self.model.policy_controller_info["actor_loss_scale"]

            # Train main Actor-Critic losses
            loss = _calc_ac_loss(a_loss=a_loss, c_loss=c_loss, critic_coef=self.critic_coef, entropy=entropy,
                                 entropy_coef=self.entropy_coef, b_loss=b_loss, bounds_loss_coef=self.bounds_loss_coef)
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        if not self.freeze:
            self.scaler.scale(loss).backward()

            if self.truncate_grads:
                if self.multi_gpu:
                    self.optimizer.synchronize()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.params, self.grad_norm)
                    with self.optimizer.skip_synchronize():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                else:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.params, self.grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            # Calculate control losses
            control_kwargs = {
                "control": controls_batch,
                "command": actions_batch,
                "next_obs": next_obs_batch,
                "next_control_dict": self.control_dict_converter.to_dict(next_control_dicts_batch),
                "done": dones_batch,
            }
            control_losses = self.model.get_control_models_loss(
                control_dict=batch_dict['control_dict'],
                **control_kwargs,
            )

            losses, _ = torch_ext.apply_masks([l.unsqueeze(1) for l in control_losses.values()], rnn_masks)

            # Re-map control losses in dict
            control_losses = {name: l for name, l in zip(control_losses.keys(), losses)}

        # Train custom control losses
        self.model.control_models_gradient_step(losses=control_losses, retain_graph=False)

        with torch.no_grad():
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        self.train_result = (a_loss, c_loss, entropy,
                             kl_dist, self.last_lr, lr_mul,
                             mu.detach(), sigma.detach(), b_loss,
                             {name: l for name, l in control_losses.items()},
                             )

    def get_full_state_weights(self):
        # Run super method first
        state = super().get_full_state_weights()

        # Add any custom models
        extra_states = {name: model.state_dict() for name, model in self.model.control_models.items()}
        state.update(extra_states)

        return state

    def set_full_state_weights(self, weights, load_optimizer_state=False):
        # Run super method first
        super().set_full_state_weights(weights=weights, load_optimizer_state=load_optimizer_state)

        # Set custom model weights
        for name, model in self.model.control_models.items():
            # Check if the weights contain this model, if not, raise a warning (but don't break runtime)
            if name in weights:
                model.load_state_dict(weights[name])
            else:
                print(f"########### WARNING: {name} not found in loaded checkpoint ###########")

    def postprocess_obs(self, obs):
        """
        Post processes raw observations from environment
        """
        # Add observations to control dict
        obs['control_dict']['obs'] = obs['obs']

        # Return obs
        return obs

    def reset_envs(self):
        self.obs = self.postprocess_obs(self.env_reset())


class PpoPlayerContinuousController(PpoPlayerContinuous):
    """
    Custom Continuous PPO Player class that extends base class to leverage a policy controller.
    """
    def __init__(self, config):
        # Run super first -- we have to skip the immediate super() call because we need to
        # override some init functionality
        BasePlayer.__init__(self, config)
        self.network = config['network']
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        obs_shape = self.obs_shape
        self.env_reset(self.env)
        config = {
            'actions_num': self.actions_num,
            'input_shape': obs_shape,
            'extrinsics_shape': self.control_dict['extrinsics'].shape,
            'num_seqs': self.num_agents
        }
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()
        if self.normalize_input:
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
            self.running_mean_std.eval()

        # Register control models in env task
        for name, model in self.model.control_models.items():
            self.env.task.register_control_model(name=name, model=model)

    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'prev_pre_actions': None,
            'obs' : obs,
            'rnn_states' : self.states,
            'control_dict': self.control_dict,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        pre_action = res_dict['pre_actions']
        self.states = res_dict['rnn_states']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action

        # current_action = action

        # Store command
        self.command = current_action

        # Pass action through controller possibly
        current_action = self.model.policy_action_to_control_action(control_dict=self.control_dict, action=current_action)

        current_action = torch.squeeze(current_action.detach())
        return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()

        # Run all pre steps
        input_dict = {
            "control_dict": self.control_dict,
            "obs": self.obs,
            "control": actions,
            "commmand": self.command,
        }
        cm_vals = self.model.control_models_pre_env_step(train=False, **input_dict)

        obs, rewards, dones, infos = env.step(actions)
        # Always postprocess obs first
        obs = self.postprocess_obs(obs)

        # Run all post steps
        input_dict.update({
            "next_control_dict": obs['control_dict'],
            "next_obs": obs['obs'],
            "dones": dones,
        })
        input_dict.update(cm_vals)
        self.model.control_models_post_env_step(train=False, **input_dict)

        # # Calculate motion prediction
        # self.model.control_models["motion_predictor"].loss(
        #     control_dict=self.control_dict,
        #     next_control_dict=obs['control_dict'],
        #     control=actions,
        #     command=self.command,
        #     done=dones,
        # )

        # Update control dict
        self.control_dict = obs['control_dict']

        obs = obs['obs']
        self.obs = obs

        if obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]

        if self.is_tensor_obses:
            return obs, rewards.cpu(), dones.cpu(), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return torch.from_numpy(obs).to(self.device), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def obs_to_torch(self, obs):
        # Store control dict
        self.control_dict = obs['control_dict']

        obs = obs['obs']
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        else:
            if self.observation_space.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)

        self.obs = obs
        return obs

    def postprocess_obs(self, obs):
        """
        Post processes raw observations from environment
        """
        # Add observations to control dict
        obs['control_dict']['obs'] = obs['obs']

        # Return obs
        return obs

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        # Restore custom models as well
        for name, model in self.model.control_models.items():
            model.load_state_dict(checkpoint[name])

    def env_reset(self, env):
        obs = env.reset()
        return self.obs_to_torch(self.postprocess_obs(obs))


# Torch JIT scripts for speed
@torch.jit.script
def _calc_neglogp(x, pre_x, mean, std, logstd, use_tanh, log2pi=LOG2PI):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, bool, float) -> Tensor
    # Two cases: If using tanh, we assume pre_x is used and we calculate neglogp that way
    # Otherwise, we use x and calculate neglogp normally
    if use_tanh:
        return 0.5 * (((pre_x - mean) / std) ** 2).sum(dim=-1) \
               + 0.5 * log2pi * pre_x.size()[-1] \
               + logstd.sum(dim=-1) + torch.log(1 - x * x + 1e-6).sum(dim=-1)
    else:
        return 0.5 * (((x - mean) / std) ** 2).sum(dim=-1) \
               + 0.5 * log2pi * x.size()[-1] \
               + logstd.sum(dim=-1)


@torch.jit.script
def _calc_ac_loss(a_loss, c_loss, critic_coef, entropy, entropy_coef, b_loss, bounds_loss_coef):
    # type: (Tensor, Tensor, float, Tensor, float, Tensor, float) -> Tensor
    return a_loss + 0.5 * c_loss * critic_coef - entropy * entropy_coef + b_loss * bounds_loss_coef

# Register these custom classes
mb.register_model('continuous_a2c_logstd_controller', ModelA2CContinuousLogStdController)
mb.register_network('actor_critic_controller', A2CControllerBuilder)
mb.register_network('continuous_network', ContinuousNetworkBuilder)
tr.register_agent('a2c_continuous_controller', A2CControllerAgent)
tr.register_player('a2c_continuous_controller', PpoPlayerContinuousController)