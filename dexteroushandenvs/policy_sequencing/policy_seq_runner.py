# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from ast import arg
from matplotlib.pyplot import get
import numpy as np
import random

from utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from utils.parse_task import parse_task
from utils.process_sarl import *
from utils.process_marl import process_MultiAgentRL, get_AgentIndex
from utils.process_mtrl import *
from utils.process_metarl import *
import os

from rl_games.common import env_configurations, experiment, vecenv
from rl_games.common.algo_observer import AlgoObserver, IsaacAlgoObserver
from rl_games.torch_runner import Runner
from rl_games.algos_torch import torch_ext
import yaml
from rl_games.common.diagnostics import DefaultDiagnostics, PpoDiagnostics

# from utils.rl_games_custom import 
from rl_games.common.a2c_common import swap_and_flatten01
import time
from copy import deepcopy
from tensorboardX import SummaryWriter
from rl_games.torch_runner import _restore, _override_sigma
import time
from gym import spaces

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class PolicySequencingRunner():
    def __init__(self, config_name, args, env, cfg_train, num_policy=2):
        def override_rlgames_cfg(config_name, args, env, cfg_train, policy_name):
            with open(config_name, 'r') as stream:
                rlgames_cfg = yaml.safe_load(stream)
                rlgames_cfg['params']['config']['name'] = args.task
                rlgames_cfg['params']['config']['num_actors'] = env.num_environments
                rlgames_cfg['params']['seed'] = cfg_train["seed"]
                rlgames_cfg['params']['config']['env_config']['seed'] = cfg_train["seed"]
                rlgames_cfg['params']['config']['vec_env'] = env
                if policy_name == "grasp":
                    rlgames_cfg['params']['config']['env_info'] = env.get_grasp_env_info()
                elif policy_name == "retri":
                    rlgames_cfg['params']['config']['env_info'] = env.get_retri_env_info()
                elif policy_name == "insert":
                    rlgames_cfg['params']['config']['env_info'] = env.get_insert_env_info()
            return rlgames_cfg
        
        self.num_policy = num_policy
        self.algo_observer = IsaacAlgoObserver()
        self.diagnostics = DefaultDiagnostics()

        # TODO: fix abs path
        self.before_episode_length = 100
        rl_games_cfg_before = override_rlgames_cfg("cfg/{}/ppo_continuous_grasp.yaml".format(args.algo), args, env, cfg_train, policy_name="grasp")
        rl_games_cfg_after = override_rlgames_cfg("cfg/{}/ppo_continuous_grasp.yaml".format(args.algo), args, env, cfg_train, policy_name="insert")

        vargs = vars(args)
        self.agents = []
        for i in range(self.num_policy):
            if i == 0:
                runner = Runner(algo_observer=self.algo_observer)
                runner.load(rl_games_cfg_before)
                runner.reset()

                agent = runner.algo_factory.create(runner.algo_name, base_name='before', params=runner.params)
                vargs["checkpoint"] = args.before_checkpoint
                _restore(agent, vargs)
                _override_sigma(agent, vargs)


            else:
                runner = Runner(algo_observer=self.algo_observer)
                runner.load(rl_games_cfg_after)
                runner.reset()
                
                agent = runner.algo_factory.create(runner.algo_name, base_name='after', params=runner.params)
                vargs["checkpoint"] = args.after_checkpoint
                _restore(agent, vargs)
                _override_sigma(agent, vargs)

            agent.init_tensors()

            agent.last_mean_rewards = -100500
            self.agents.append(agent)

        self.start_time = time.time()
        self.total_time = 0
        self.rep_count = 0
        self.writer = SummaryWriter(self.agents[0].summaries_dir)

        self.agents[1].obs = self.agents[1].env_reset()
        self.agents[0].obs = self._to_device("cuda:0", {})
        self.agents[0].obs = self.agents[0].obs_to_tensors(self.agents[0].obs)
        self.agents[0].is_tensor_obses = True

        self.agents[0].obs["obs"] = self.agents[1].obs["before_obs"].clone()
        self.agents[0].obs["states"] = self.agents[1].obs["before_states"].clone()

        for i in range(self.num_policy):
            self.agents[i].curr_frames = self.agents[0].batch_size_envs

    def run(self):
        while True:
            for i in range(self.num_policy):
                epoch_num = self.agents[i].update_epoch()

            step_time, play_time, update_time, sum_time, training_info, training_info_name = self.rl_games_train_epoch()

            self.total_time += sum_time
            frame = self.agents[0].frame // self.agents[0].num_agents

            # cleaning memory to optimize space
            for i in range(self.num_policy):
                self.agents[i].dataset.update_values_dict(None)

            if any(agent.rank for agent in self.agents) == 0:
                for i in range(self.num_policy):
                    self.diagnostics.epoch(self.agents[i], current_epoch=epoch_num)
                    curr_frames = self.agents[i].curr_frames * self.agents[i].rank_size if self.agents[i].multi_gpu else self.agents[i].curr_frames
                    self.agents[i].frame += curr_frames
                
                # do we need scaled_time?
                scaled_time = self.agents[0].num_agents * sum_time
                scaled_play_time = self.agents[0].num_agents * play_time

                # if any(agent.print_stats for agent in self.agents) == 1:
                step_time = max(step_time, 1e-6)
                fps_step = curr_frames / step_time
                fps_step_inference = curr_frames / scaled_play_time
                fps_total = curr_frames / scaled_time
                print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num}/{self.agents[0].max_epochs}')

                for i in range(self.num_policy):
                    self.rl_games_agent_logs(self.agents[i], self.total_time, epoch_num, step_time, play_time, update_time, training_info, frame, scaled_time, scaled_play_time, curr_frames, self.writer)

                update_time = 0

    def rl_games_save_agent_checkpoint(self, agent, checkpoint_name, mean_rewards, epoch_num):
        if agent.save_freq > 0:
            if (epoch_num % agent.save_freq == 0) and (mean_rewards[0] <= agent.last_mean_rewards):
                agent.save(os.path.join(agent.nn_dir, 'last_' + checkpoint_name))

        if mean_rewards[0] > agent.last_mean_rewards and epoch_num >= agent.save_best_after:
            print('saving next best rewards: ', mean_rewards)
            agent.last_mean_rewards = mean_rewards[0]
            agent.save(os.path.join(agent.nn_dir, agent.config['name']))

            if 'score_to_win' in agent.config:
                if agent.last_mean_rewards > agent.config['score_to_win']:
                    print('Network won!')
                    agent.save(os.path.join(agent.nn_dir, checkpoint_name))
                    should_exit = True

    def rl_games_agent_logs(self, agent, total_time, epoch_num, step_time, play_time, update_time, training_info, frame, scaled_time, scaled_play_time, curr_frames, writer):
        a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = training_info
        
        self.write_stats(agent, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames)
        
        if len(b_losses) > 0:
            writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

        if agent.has_soft_aug:
            writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

        if agent.game_rewards.current_size > 0:
            mean_rewards = agent.game_rewards.get_mean()
            mean_lengths = agent.game_lengths.get_mean()
            agent.mean_rewards = mean_rewards[0]

            for i in range(agent.value_size):
                rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                print("add reward")
                writer.add_scalar(agent.name + '_' + rewards_name + '/step'.format(i), mean_rewards[i], frame)
                writer.add_scalar(agent.name + '_' + rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                writer.add_scalar(agent.name + '_' + rewards_name + '/time'.format(i), mean_rewards[i], total_time)

            writer.add_scalar('episode_lengths/step', mean_lengths, frame)
            writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
            writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

            checkpoint_name = agent.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0]) + '_agent_' + agent.name
            self.rl_games_save_agent_checkpoint(agent, checkpoint_name, mean_rewards, epoch_num)

    def rl_games_train_epoch(self):
        for i in range(self.num_policy):
            self.agents[i].vec_env.set_train_info(self.agents[i].frame, self.agents[i])
            self.agents[i].set_eval()

        play_time_start = time.time()
        with torch.no_grad():
            batch_dict_before, batch_dict_after = self.rl_games_play_steps()

        play_time_end = time.time()
        update_time_start = time.time()

        # TODO
        if self.agents[0].vec_env.task.progress_buf[0] < self.before_episode_length:
            training_info = self.agent_train_epoch(self.agents[0], batch_dict_before)
            training_info_name = "before"
        else:
            training_info = self.agent_train_epoch(self.agents[1], batch_dict_after)
            training_info_name = "after"

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict_before['step_time'], play_time, update_time, total_time, training_info, training_info_name

    def rl_games_play_steps(self):
        self.update_lists = []
        for i in range(self.num_policy):
            update_list = self.agents[i].update_list
            self.update_lists.append(update_list)

        step_time = 0.0

        for n in range(self.agents[0].horizon_length):
            if self.agents[0].vec_env.task.progress_buf[0] < self.before_episode_length:
                agent = self.agents[0]
                # agent_before.obs['obs'] = agent_after.obs['obs'][:, :81].clone()
                res_dict_before = agent.get_action_values(agent.obs)

                self.pre_step(agent, self.update_lists[0], res_dict_before, n)

                step_time_start = time.time()
                concat_action = res_dict_before['actions']
                obs, rewards, dones, infos = agent.env_step(concat_action)

                agent.obs['obs'] = infos["before_obs"]
                agent.obs['states'] = infos["before_states"]
                agent.dones = infos["before_reset_buf"]
                before_rewards = infos["before_rew_buf"].unsqueeze(1)

                step_time_end = time.time()

                step_time += (step_time_end - step_time_start)

                self.post_step(agent, res_dict_before, before_rewards, n, infos)

            else:
                agent = self.agents[1]
                res_dict_after = agent.get_action_values(agent.obs)

                self.pre_step(agent, self.update_lists[1], res_dict_after, n)

                step_time_start = time.time()
                concat_action = res_dict_after['actions']
                obs, rewards, dones, infos = agent.env_step(concat_action)

                agent.obs['obs'] = infos["after_obs"]
                agent.obs['states'] = infos["after_states"]
                agent.dones = infos["after_reset_buf"]
                after_rewards = infos["after_rew_buf"].unsqueeze(1)

                step_time_end = time.time()

                step_time += (step_time_end - step_time_start)

                self.post_step(agent, res_dict_after, after_rewards, n, infos)

        batch_dict_before = self.generate_batch_dict(self.agents[0], step_time)
        batch_dict_after = self.generate_batch_dict(self.agents[1], step_time)

        return batch_dict_before, batch_dict_after

    # embedding function
    def agent_train_epoch(self, agent, batch_dict):
        rnn_masks = batch_dict.get('rnn_masks', None)

        agent.set_train()
        agent.curr_frames = batch_dict.pop('played_frames')
        agent.prepare_dataset(batch_dict)
        agent.algo_observer.after_steps()
        if agent.has_central_value:
            agent.train_central_value()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        for mini_ep in range(0, agent.mini_epochs_num):
            ep_kls = []
            for i in range(len(agent.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = agent.train_actor_critic(agent.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if agent.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                agent.dataset.update_mu_sigma(cmu, csigma)
                if agent.schedule_type == 'legacy':
                    av_kls = kl
                    if agent.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= agent.rank_size
                    agent.last_lr, agent.entropy_coef = agent.scheduler.update(agent.last_lr, agent.entropy_coef, agent.epoch_num, 0, av_kls.item())
                    agent.update_lr(agent.last_lr)

            av_kls = torch_ext.mean_list(ep_kls)

            if agent.schedule_type == 'standard':
                agent.last_lr, agent.entropy_coef = agent.scheduler.update(agent.last_lr, agent.entropy_coef, agent.epoch_num, 0, av_kls.item())
                agent.update_lr(agent.last_lr)

            kls.append(av_kls)
            self.diagnostics.mini_epoch(agent, mini_ep)
            if agent.normalize_input:
                agent.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

        return a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul
    
    # one step
    def generate_batch_dict(self, agent, step_time):
        last_values = agent.get_values(agent.obs)

        fdones = agent.dones.float()
        mb_fdones = agent.experience_buffer.tensor_dict['dones'].float()
        mb_values = agent.experience_buffer.tensor_dict['values']
        mb_rewards = agent.experience_buffer.tensor_dict['rewards']
        mb_advs = agent.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = agent.experience_buffer.get_transformed_list(swap_and_flatten01, agent.tensor_list)
        batch_dict['returns'] = swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = agent.batch_size
        batch_dict['step_time'] = step_time

        return batch_dict

    def pre_step(self, agent, update_list, res_dict, n):
        agent.experience_buffer.update_data('obses', n, agent.obs['obs'])
        agent.experience_buffer.update_data('dones', n, agent.dones)

        for k in update_list:
            agent.experience_buffer.update_data(k, n, res_dict[k]) 
        if agent.has_central_value:
            agent.experience_buffer.update_data('states', n, agent.obs['states'])

    def post_step(self, agent, res_dict, rewards, n, infos):
        shaped_rewards = agent.rewards_shaper(rewards)
        if agent.value_bootstrap and 'time_outs' in infos:
            shaped_rewards += agent.gamma * res_dict['values'] * agent.cast_obs(infos['time_outs']).unsqueeze(1).float()

        agent.experience_buffer.update_data('rewards', n, shaped_rewards)

        agent.current_rewards += rewards
        agent.current_lengths += 1
        all_done_indices = agent.dones.nonzero(as_tuple=False)
        env_done_indices = agent.dones.view(agent.num_actors, agent.num_agents).all(dim=1).nonzero(as_tuple=False)

        agent.game_rewards.update(agent.current_rewards[env_done_indices])
        agent.game_lengths.update(agent.current_lengths[env_done_indices])
        agent.algo_observer.process_infos(infos, env_done_indices)

        not_dones = 1.0 - agent.dones.float()

        agent.current_rewards = agent.current_rewards * not_dones.unsqueeze(1)
        agent.current_lengths = agent.current_lengths * not_dones

    def write_stats(self, agent, total_time, epoch_num, step_time, play_time, update_time, a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame, scaled_time, scaled_play_time, curr_frames):
        # do we need scaled time?
        self.diagnostics.send_info(self.writer)
        self.writer.add_scalar(agent.name + '_' + 'performance/step_inference_rl_update_fps', curr_frames / scaled_time, frame)
        self.writer.add_scalar(agent.name + '_' + 'performance/step_inference_fps', curr_frames / scaled_play_time, frame)
        self.writer.add_scalar(agent.name + '_' + 'performance/step_fps', curr_frames / step_time, frame)
        self.writer.add_scalar(agent.name + '_' + 'performance/rl_update_time', update_time, frame)
        self.writer.add_scalar(agent.name + '_' + 'performance/step_inference_time', play_time, frame)
        self.writer.add_scalar(agent.name + '_' + 'performance/step_time', step_time, frame)
        self.writer.add_scalar(agent.name + '_' + 'losses/a_loss', torch_ext.mean_list(a_losses).item(), frame)
        self.writer.add_scalar(agent.name + '_' + 'losses/c_loss', torch_ext.mean_list(c_losses).item(), frame)        
        self.writer.add_scalar(agent.name + '_' + 'losses/entropy', torch_ext.mean_list(entropies).item(), frame)
        self.writer.add_scalar(agent.name + '_' + 'info/last_lr', last_lr * lr_mul, frame)
        self.writer.add_scalar(agent.name + '_' + 'info/lr_mul', lr_mul, frame)
        self.writer.add_scalar(agent.name + '_' + 'info/kl', torch_ext.mean_list(kls).item(), frame)
        self.writer.add_scalar(agent.name + '_' + 'info/epochs', epoch_num, frame)
        self.algo_observer.after_print_stats(frame, epoch_num, total_time)


    #utils
    def _to_device(self, device, inp):
        """
        Maps all tensors in @inp to this object's device.

        Args:
            inp (tensor, iterable, dict): Any primitive data type that includes tensor(s)

        Returns:
            (tensor, iterable, dict): Same type as @inp, with all tensors mapped to self.rl_device
        """
        # Check all cases
        if isinstance(inp, torch.Tensor):
            inp = inp.to(device)
        elif isinstance(inp, dict):
            for k, v in inp.items():
                inp[k] = _to_device(v)
        else:
            # We assume that this is an iterable, so we loop over all entries
            for i, entry in enumerate(inp):
                inp[i] = _to_device(entry)
        return inp