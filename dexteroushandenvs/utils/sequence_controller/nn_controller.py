import torch
import copy
import numpy as np
from .nn_builder import build_network
from rl_games.algos_torch import torch_ext

class SeqNNController:
    def __init__(self, num_actors=1, dig_obs_dim=62, spin_obs_dim=62, grasp_obs_dim=62, insert_obs_dim=75, device="cuda:0"):
        self.dig_one_step_obs_dim = dig_obs_dim
        self.spin_one_step_obs_dim = spin_obs_dim
        self.grasp_one_step_obs_dim = grasp_obs_dim
        self.insert_one_step_obs_dim = insert_obs_dim
        
        self.num_actors = num_actors
        self.seq_len = 3
        
        self.dig_act_moving_average = 1.0
        self.spin_act_moving_average = 0.2
        self.grasp_act_moving_average = 0.4
        self.insert_act_moving_average = 0.2

        self.states = None
        self.device = device
        self.time_step = 0

        self.dig_obs_buffer = torch.zeros((self.num_actors, self.seq_len * self.dig_one_step_obs_dim), dtype=torch.float32, device=self.device)
        self.spin_obs_buffer = torch.zeros((self.num_actors, self.seq_len * self.spin_one_step_obs_dim), dtype=torch.float32, device=self.device)
        self.grasp_obs_buffer = torch.zeros((self.num_actors, self.seq_len * self.grasp_one_step_obs_dim), dtype=torch.float32, device=self.device)
        self.insert_obs_buffer = torch.zeros((self.num_actors, self.seq_len * self.insert_one_step_obs_dim), dtype=torch.float32, device=self.device)

        self.dig_model = build_network(
            './utils/sequence_controller/dig_network.yaml',
            obs_dim=self.dig_one_step_obs_dim * self.seq_len)
        self.spin_model = build_network(
            './utils/sequence_controller/grasp_network.yaml',
            obs_dim=self.spin_one_step_obs_dim * self.seq_len)
        self.grasp_model = build_network(
            './utils/sequence_controller/grasp_network.yaml',
            obs_dim=self.grasp_one_step_obs_dim * self.seq_len)
        self.insert_model = build_network(
            './utils/sequence_controller/insert_network.yaml',
            obs_dim=self.insert_one_step_obs_dim * self.seq_len)

        self.dof_lower = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -0.4700,
                                       -0.1960, -0.1740, -0.2270, 0.2630, -0.1050, -0.1890, -0.1620, -0.4700,
                                       -0.1960, -0.1740, -0.2270, -0.4700, -0.1960, -0.1740, -0.2270],
                                      device='cuda:0')

        self.dof_upper = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.4700,
                                       1.6100, 1.7090, 1.6180, 1.3960, 1.1630, 1.6440, 1.7190, 0.4700,
                                       1.6100, 1.7090, 1.6180, 0.4700, 1.6100, 1.7090, 1.6180],
                                      device='cuda:0')

        self.object_init_pose = torch.tensor([0.0422, 0.0179, 0.9293, 0.0611, -0.9981, -0.0071, -0.0081],
                                             device='cuda:0')

        self.init_qpos = torch.tensor([-0.3463, -0.3414, 0.4400, -2.7079, 0.2244, 2.3851, 3.1415,
                                       0.0, -0.174, 0.785, 0.785,
                                       0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785],
                                      device='cuda:0')

        self.prev_targets = torch.zeros((self.num_actors, 23), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_actors, 23), dtype=torch.float, device=self.device)
        self.prev_targets[:, 7:23] = self.scale(torch.tensor([0, 0, -1, 0.5, 1, 0, -1, 0.5, 0, 0, -1, 0.5, 0, 0, -1, 0.5], dtype=torch.float, device=self.device), self.dof_lower[7:23], self.dof_upper[7:23])
        self.cur_targets[:, 7:23] = self.scale(torch.tensor([0, 0, -1, 0.5, 1, 0, -1, 0.5, 0, 0, -1, 0.5, 0, 0, -1, 0.5], dtype=torch.float, device=self.device), self.dof_lower[7:23], self.dof_upper[7:23])

        self.dig_current_obs = torch.zeros((self.num_actors, self.dig_one_step_obs_dim), dtype=torch.float32, device=self.device)
        self.spin_current_obs = torch.zeros((self.num_actors, self.spin_one_step_obs_dim), dtype=torch.float32, device=self.device)
        self.grasp_current_obs = torch.zeros((self.num_actors, self.grasp_one_step_obs_dim), dtype=torch.float32, device=self.device)
        self.insert_current_obs = torch.zeros((self.num_actors, self.insert_one_step_obs_dim), dtype=torch.float32, device=self.device)
        
        self.last_action = torch.zeros((self.num_actors, 23), dtype=torch.float, device=self.device)

        self.use_belif_state = True
        if self.use_belif_state == True:
            self.belif_state_model = self.load_belif_state(ckp_path="./intermediate_state/contact_slamer/model_99200.pt", obs_dim=self.spin_one_step_obs_dim, obs_len=self.seq_len)

        self.select_policy("reach")

    def _preproc_obs(self, obs_batch):
        if type(obs_batch) is dict:
            obs_batch = copy.copy(obs_batch)
            for k, v in obs_batch.items():
                if v.dtype == torch.uint8:
                    obs_batch[k] = v.float() / 255.0
                else:
                    obs_batch[k] = v
        else:
            if obs_batch.dtype == torch.uint8:
                obs_batch = obs_batch.float() / 255.0
        return obs_batch

    def scale(self, x, lower, upper):
        return (0.5 * (x + 1.0) * (upper - lower) + lower)

    def unscale(self, x, lower, upper):
        return (2.0 * x - upper - lower) / (upper - lower)

    def tensor_clamp(self, t, min_t, max_t):
        return torch.max(torch.min(t, max_t), min_t)

    def predict(self, input, deterministic=False):
        if type(input) != torch.Tensor:
            input = torch.from_numpy(input).float().cuda()

        input = self._preproc_obs(input)

        if self.policy_name == "dig":
            # unscale the propriocetion
            self.dig_current_obs[:, 0:65] = input[:, 0:65]
            self.dig_current_obs[:, 0:16] = self.unscale(input[:, 0:16], self.dof_lower[7:23], self.dof_upper[7:23])
            self.dig_current_obs[:, 16:23] = torch.tensor([ 0.4167,  0.2015,  0.3631,  0.6965, -0.7167,  0.0172, -0.0306], device='cuda:0')
            self.dig_current_obs[:, 30:46] = self.last_action[:, 7:23] - self.unscale(input[:, 0:16], self.dof_lower[7:23], self.dof_upper[7:23])
            self.dig_current_obs[:, 46:62] = self.last_action[:, 7:23]

            input_dict = self.process_history_frame(self.dig_obs_buffer, self.dig_one_step_obs_dim, self.dig_current_obs)

        if self.policy_name == "spin":
            # unscale the propriocetion
            self.spin_current_obs[:, 0:30] = input[:, 0:30]
            self.spin_current_obs[:, 0:16] = self.unscale(input[:, 0:16], self.dof_lower[7:23], self.dof_upper[7:23])

            self.spin_current_obs[:, 30:46] = self.last_action[:, 7:23] - self.unscale(input[:, 0:16], self.dof_lower[7:23], self.dof_upper[7:23])
            self.spin_current_obs[:, 46:62] = self.last_action[:, 7:23]

            input_dict = self.process_history_frame(self.spin_obs_buffer, self.spin_one_step_obs_dim, self.spin_current_obs)

            if self.use_belif_state:
                self.predict_pose, _ = self.belif_state_model(self.spin_obs_buffer)
                input_dict["obs"][:, 16:19] = torch.tensor([0.2048, 0.0404, 0.0361], device=self.device)
                input_dict["obs"][:, 19:23] = self.predict_pose[:, 0:4].detach()

        if self.policy_name == "grasp":
            # unscale the propriocetion
            self.grasp_current_obs[:, 0:30] = input[:, 0:30]
            self.grasp_current_obs[:, 0:16] = self.unscale(input[:, 0:16], self.dof_lower[7:23], self.dof_upper[7:23])

            self.grasp_current_obs[:, 30:46] = self.last_action[:, 7:23] - self.unscale(input[:, 0:16], self.dof_lower[7:23], self.dof_upper[7:23])
            self.grasp_current_obs[:, 46:62] = self.last_action[:, 7:23]

            input_dict = self.process_history_frame(self.grasp_obs_buffer, self.grasp_one_step_obs_dim, self.grasp_current_obs)

        elif self.policy_name == "insert":
            # unscale the propriocetion
            self.insert_current_obs[:, 0:30] = input[:, 0:30]
            self.insert_current_obs[:, 0:16] = self.unscale(input[:, 0:16], self.dof_lower[7:23], self.dof_upper[7:23])

            self.insert_current_obs[:, 30:46] = self.last_action[:, 7:23] - self.unscale(input[:, 0:16], self.dof_lower[7:23], self.dof_upper[7:23])
            self.insert_current_obs[:, 46:62] = self.last_action[:, 7:23]

            # 90 degree
            # self.insert_current_obs[:, 62:69] = torch.tensor([ 0.2350, -0.2000,  0.6555,  0.0000,  0.0000,  0.7070,  0.7070], device='cuda:0')
            # 0 degree
            self.insert_current_obs[:, 62:69] = torch.tensor([ 0.2650, -0.2000,  0.6555,  0.0000,  0.0000,  0.7070,  -0.7070], device='cuda:0')

            input_dict = self.process_history_frame(self.insert_obs_buffer, self.insert_one_step_obs_dim, self.insert_current_obs)

        if self.policy_name in ["dig", "grasp", "insert", "spin"]:
            with torch.no_grad():
                res_dict = self.selected_model(input_dict)

            current_action = res_dict['mus']

            current_action = torch.clip(current_action, -1.0, 1.0)

            self.last_action = current_action.clone()

        if self.policy_name == "grasp":
            targets = self._postproc_action(current_action, moving_average=self.grasp_act_moving_average)
        elif self.policy_name == "dig":
            targets = self._postproc_action(current_action, moving_average=self.dig_act_moving_average)
        elif self.policy_name == "spin":
            targets = self._postproc_action(current_action, moving_average=self.spin_act_moving_average)
        elif self.policy_name == "insert":
            targets = self._postproc_action(current_action, moving_average=self.insert_act_moving_average)
        elif self.policy_name == "reach":
            targets = self.cur_targets[:, 0:23]
        elif self.policy_name == "open_hand":
            self.open_hand()
            targets = self.open_hand_targets[:, 0:23]
            self.open_hand_step += 1
        elif self.policy_name == "hard_code_grasp":
            self.hard_code_grasp()
            targets = self.hard_code_grasp_targets[:, 0:23]
            self.hard_code_grasp_step += 1
        elif self.policy_name == "open_hand_once":
            self.open_hand_once()
            targets = self.open_hand_once_targets[:, 0:23]
            self.open_hand_once_step += 1

        return targets

    def _postproc_action(self, current_action, moving_average=0.2):
        self.cur_targets[:, 7:23] = self.scale(current_action[:, 7:23],
                                               self.dof_lower[7:23],
                                               self.dof_upper[7:23])
        self.cur_targets[:, 7:23] = moving_average * self.cur_targets[:, 7:23] + (
                    1.0 - moving_average) * self.prev_targets[:, 7:23]

        self.cur_targets[:, 7:23] = self.tensor_clamp(self.cur_targets[:, 7:23],
                                                      self.dof_lower[7:23],
                                                      self.dof_upper[7:23])

        self.cur_targets[:, 0:7] = 2 / 60 * current_action[:, :7]

        self.prev_targets[:, :] = self.cur_targets[:, :].clone()

        return self.cur_targets

    def load(self, dig_ckp=None, spin_ckp=None, grasp_ckp=None, insert_ckp=None):
        if dig_ckp != None:
            checkpoint = torch_ext.load_checkpoint(dig_ckp)
            self.dig_model.load_state_dict(checkpoint['model'])
            self.dig_model.to(self.device)
            self.dig_model.eval()
            
        if spin_ckp != None:
            checkpoint = torch_ext.load_checkpoint(spin_ckp)
            self.spin_model.load_state_dict(checkpoint['model'])
            self.spin_model.to(self.device)
            self.spin_model.eval()
            
        if grasp_ckp != None:
            checkpoint = torch_ext.load_checkpoint(grasp_ckp)
            self.grasp_model.load_state_dict(checkpoint['model'])
            self.grasp_model.to(self.device)
            self.grasp_model.eval()

        if insert_ckp != None:
            checkpoint = torch_ext.load_checkpoint(insert_ckp)
            self.insert_model.load_state_dict(checkpoint['model'])
            self.insert_model.to(self.device)
            self.insert_model.eval()

    def select_policy(self, policy_name=""):
        if policy_name == "reach":
            self.policy_name = "reach"

        elif policy_name == "dig":
            self.selected_model = self.dig_model
            self.policy_name = "dig"
            
        elif policy_name == "spin":
            self.selected_model = self.spin_model
            self.policy_name = "spin"
            
        elif policy_name == "grasp":
            self.selected_model = self.grasp_model
            self.policy_name = "grasp"

        elif policy_name == "insert":
            self.selected_model = self.insert_model
            self.policy_name = "insert"

        elif policy_name == "open_hand":
            self.policy_name = "open_hand"
            self.open_hand_step = 0

        elif policy_name == "hard_code_grasp":
            self.policy_name = "hard_code_grasp"
            self.hard_code_grasp_step = 0

        elif policy_name == "open_hand_once":
            self.policy_name = "open_hand_once"
            self.open_hand_once_step = 0
    
    def open_hand(self):
        self.stage_0_steps = 6
        self.stage_1_steps = 3
        self.stage_2_steps = 6
        if self.open_hand_step <= self.stage_0_steps:
            init_qpos = [0, 0, -1, 0.5, 1, 0, -1, 0.5, 0, 0, -1, 0.5, 0, 0, -1, 0.5]
            target_qpos = [0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1]
            self.open_hand_targets = self.set_hard_code_qpos(init_qpos=init_qpos, target_qpos=target_qpos, cur_step=self.open_hand_step, stage_step=self.stage_0_steps)
            
        elif (self.stage_0_steps + self.stage_1_steps) >= self.open_hand_step > self.stage_0_steps:
            init_qpos = [0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1]
            target_qpos = [0, -1, -1, 0.5, 0, -1, -1, 0.5, 0, -1, -1, 0.5, 0, -1, -1, 0.5]
            self.open_hand_targets = self.set_hard_code_qpos(init_qpos=init_qpos, target_qpos=target_qpos, cur_step=self.open_hand_step - self.stage_0_steps, stage_step=self.stage_1_steps)
            
        elif self.open_hand_step > (self.stage_0_steps + self.stage_1_steps):
            init_qpos = [0, -1, -1, 0.5, 0, -1, -1, 0.5, 0, -1, -1, 0.5, 0, -1, -1, 0.5]
            target_qpos = [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
            self.open_hand_targets = self.set_hard_code_qpos(init_qpos=init_qpos, target_qpos=target_qpos, cur_step=self.open_hand_step - (self.stage_0_steps + self.stage_1_steps), stage_step=self.stage_2_steps)
            
        if self.open_hand_step == (self.stage_0_steps + self.stage_1_steps + self.stage_2_steps):
            self.open_hand_step = 0
        
        self.clear_obs_buffer()

    def hard_code_grasp(self):
        self.stage_0_steps = 6
        self.stage_1_steps = 3

        if self.hard_code_grasp_step <= self.stage_0_steps:
            init_qpos = [0, 0, -1, 0.5, 1, 0, -1, 0.5, 0, 0, -1, 0.5, 0, 0, -1, 0.5]
            target_qpos = [0, 0, -1, -1, 1, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1]
            self.hard_code_grasp_targets = self.set_hard_code_qpos(init_qpos=init_qpos, target_qpos=target_qpos, cur_step=self.hard_code_grasp_step, stage_step=self.stage_0_steps)
            
        elif (self.stage_0_steps + self.stage_1_steps) >= self.hard_code_grasp_step > self.stage_0_steps:
            init_qpos = [0, 0, -1, -1, 1, 0, -1, -1, 0, 0, -1, -1, 0, 0, -1, -1]
            target_qpos = [0, 1, -1, 0.5, 1, 0, 0, -1, 0, 1, -1, 0.5, 0, 1, -1, 0.5]
            self.hard_code_grasp_targets = self.set_hard_code_qpos(init_qpos=init_qpos, target_qpos=target_qpos, cur_step=self.hard_code_grasp_step - self.stage_0_steps, stage_step=self.stage_1_steps)

        self.clear_obs_buffer()

    def open_hand_once(self):
        self.stage_0_steps = 6

        if self.open_hand_once_step <= self.stage_0_steps:
            init_qpos = [0, 0, -1, 0.5, 1, 0, -1, 0.5, 0, 0, -1, 0.5, 0, 0, -1, 0.5]
            target_qpos = [0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1]
            self.open_hand_once_targets = self.set_hard_code_qpos(init_qpos=init_qpos, target_qpos=target_qpos, cur_step=self.open_hand_once_step, stage_step=self.stage_0_steps)
            
        self.clear_obs_buffer()

    def process_history_frame(self, obs_buffer, one_step_obs_dim, current_obs):
            for i in reversed(range(self.seq_len)):
                if i == 0:
                    obs_buffer[:, i*one_step_obs_dim:(i+1)*one_step_obs_dim] = current_obs
                else:
                    obs_buffer[:, (i)*one_step_obs_dim:(i+1)*one_step_obs_dim] = obs_buffer[:, (i-1)*one_step_obs_dim:(i)*one_step_obs_dim].clone()

            obs_buffer = torch.clip(obs_buffer, -5.0, 5.0)

            input_dict = {
                'is_train': False,
                'prev_actions': None,
                'obs': obs_buffer,
                'rnn_states': self.states
            }
            
            return input_dict

    def clear_obs_buffer(self):
        self.dig_obs_buffer = torch.zeros((self.num_actors, self.seq_len * self.dig_one_step_obs_dim), dtype=torch.float32, device=self.device)
        self.spin_obs_buffer = torch.zeros((self.num_actors, self.seq_len * self.spin_one_step_obs_dim), dtype=torch.float32, device=self.device)
        self.grasp_obs_buffer = torch.zeros((self.num_actors, self.seq_len * self.grasp_one_step_obs_dim), dtype=torch.float32, device=self.device)
        self.insert_obs_buffer = torch.zeros((self.num_actors, self.seq_len * self.insert_one_step_obs_dim), dtype=torch.float32, device=self.device)
        self.last_action = torch.zeros_like(self.last_action)

    def reset_cur_targets(self):
        self.prev_targets[:, 7:23] = self.scale(
            torch.tensor([0, 0, -1, 0.5, 1, 0, -1, 0.5, 0, 0, -1, 0.5, 0, 0, -1, 0.5], dtype=torch.float, device=self.device),
            self.dof_lower[7:23], self.dof_upper[7:23])
        self.cur_targets[:, 7:23] = self.scale(
            torch.tensor([0, 0, -1, 0.5, 1, 0, -1, 0.5, 0, 0, -1, 0.5, 0, 0, -1, 0.5], dtype=torch.float, device=self.device),
            self.dof_lower[7:23], self.dof_upper[7:23])
		
        self.clear_obs_buffer()
    
    def set_hard_code_qpos(self, init_qpos, target_qpos, cur_step, stage_step):
            init_qpos = self.scale(
                torch.tensor(init_qpos, dtype=torch.float,
                             device=self.device),
                self.dof_lower[7:23], self.dof_upper[7:23])
            targets = torch.zeros_like(self.cur_targets)
            targets[:, 7:23] = (self.scale(
                torch.tensor(target_qpos, dtype=torch.float,
                             device=self.device),
                self.dof_lower[7:23], self.dof_upper[7:23]) - init_qpos) * np.clip(
                (cur_step / stage_step), 0, 1) + init_qpos
            return targets
    
    def load_belif_state(self, ckp_path, obs_dim, obs_len):
        from .contact_slamer import ContactSLAMer
        self.contact_slamer = ContactSLAMer(obs_dim * obs_len, 4).to(self.device)
        self.contact_slamer.load_state_dict(torch.load(ckp_path, map_location='cuda:0'))
        self.contact_slamer.eval()
        return self.contact_slamer
    
    def predict_belif_pose(self):
        return self.predict_pose

if __name__ == '__main__':
	seq_policy = SeqNNController(num_actors=1, obs_dim=30)
	seq_policy.load("./runs/AllegroHandLegoTestOrientGrasp_25-06-24-26/nn/last_AllegroHandLegoTestOrientGrasp_ep_32000_rew_235.23706.pth", None)

	for i in range(150):
		obs = np.random.uniform(0, 1, 30)

		''' 
			Observation description: 
			
			DIMENSION |	DESCRIPTION	 |	PROGRAMMING HINT
			-------------------------------------------------------------------------------------------
			[0:16]:   |	current_hand_qpos |	(SET BY USER. 			Set it by the previous protocol)
			[16:23]:  |	hand_pose 	      |	(SET BY USER. 			Set it by the previous protocol)
			[23:26]:  |	object pos        |	(SET BY USER. 			Set it by the previous protocol)
			[26:30]:  |	object_	rot       |	(SET BY USER. 			Set it by the previous protocol)
		'''
		targets = seq_policy.predict(input=np.clip(obs, -5, 5))[0]
		'''
			Action description: The network outputs a [-1, 1]^action_dim vector. 
		'''


	seq_policy.select_policy("insert")

	
