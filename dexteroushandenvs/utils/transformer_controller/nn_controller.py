import torch
import copy
import numpy as np
from .GPT_policy import GPT_wrapper

class NNController:
	def __init__(self, num_actors=1, obs_dim=30, act_moving_average=0.2, device="cuda:0"):
		self.one_step_obs_dim = obs_dim
		self.num_actors = num_actors
		self.seq_len = 3
		self.act_moving_average = act_moving_average

		self.states = None
		self.device = device
		self.time_step = 0

		self.obs_buffer = torch.zeros((self.num_actors, self.seq_len, self.one_step_obs_dim), dtype=torch.float32, device=self.device)
		self.grasp_gpt_model = GPT_wrapper(feat_dim=128, n_layer=4, n_head=4, gmm_modes=5, obs_dim=self.one_step_obs_dim, action_dim=23).to(self.device)
		self.insert_gpt_model = GPT_wrapper(feat_dim=128, n_layer=4, n_head=4, gmm_modes=5, obs_dim=self.one_step_obs_dim, action_dim=23).to(self.device)

		self.dof_lower = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -0.4700,
        -0.1960, -0.1740, -0.2270,  0.2630, -0.1050, -0.1890, -0.1620, -0.4700,
        -0.1960, -0.1740, -0.2270, -0.4700, -0.1960, -0.1740, -0.2270],
        device='cuda:0')
		
		self.dof_upper = torch.tensor([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,  0.4700,
        1.6100,  1.7090,  1.6180,  1.3960,  1.1630,  1.6440,  1.7190,  0.4700,
        1.6100,  1.7090,  1.6180,  0.4700,  1.6100,  1.7090,  1.6180],
        device='cuda:0')

		self.object_init_pose = torch.tensor([ 0.0422,  0.0179,  0.9293,  0.0611, -0.9981, -0.0071, -0.0081],
        device='cuda:0')

		self.init_qpos = torch.tensor([-0.3463, -0.3414,  0.4400, -2.7079,  0.2244,  2.3851, 3.1415,
										0.0, -0.174, 0.785, 0.785,
										0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785, 0.0, -0.174, 0.785, 0.785], device='cuda:0')

		self.prev_targets = torch.zeros((self.num_actors, 23), dtype=torch.float, device=self.device)
		self.cur_targets = torch.zeros((self.num_actors, 23), dtype=torch.float, device=self.device)
		self.prev_targets[:, 7:23] = self.scale(torch.tensor([0, 0, -1, 0.5, 1, 0, -1, 0.5, 0, 0, -1, 0.5, 0, 0, -1, 0.5], dtype=torch.float, device=self.device), 
                                            self.dof_lower[7:23], self.dof_upper[7:23])
		self.cur_targets[:, 7:23] = self.scale(torch.tensor([0, 0, -1, 0.5, 1, 0, -1, 0.5, 0, 0, -1, 0.5, 0, 0, -1, 0.5], dtype=torch.float, device=self.device), 
                                            self.dof_lower[7:23], self.dof_upper[7:23])

		self.current_obs = torch.zeros((self.num_actors, self.one_step_obs_dim), dtype=torch.float32, device=self.device)
		
		self.select_policy("grasp")
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

		# unscale the propriocetion
		self.current_obs[:, 0:30] = input[:, 0:30]
		self.current_obs[:, 0:16] = self.unscale(input[:, 0:16], self.dof_lower[7:23], self.dof_upper[7:23])

		self.current_obs = torch.clip(self.current_obs, -5.0, 5.0)
		current_action = self.selected_model.forward_step(self.current_obs.unsqueeze(0)).detach()
		current_action = torch.clip(current_action, -1.0, 1.0)

		targets = self._postproc_action(current_action)

		return targets

	def _postproc_action(self, current_action):
		self.cur_targets[:, 0:23] = self.scale(current_action[:, 0:23],
												self.dof_lower[0:23],
												self.dof_upper[0:23])
		
		self.cur_targets[:, 7:23] = self.act_moving_average * self.cur_targets[:, 7:23] + (1.0 - self.act_moving_average) * self.prev_targets[:, 7:23]

		self.cur_targets[:, 7:23] = self.tensor_clamp(self.cur_targets[:, 7:23],
																		self.dof_lower[7:23],
																		self.dof_upper[7:23])
            
		self.prev_targets[:, :] = self.cur_targets[:, :].clone()

		return self.cur_targets

	def load(self, fn_list):
		if fn_list[0] != None:
			self.grasp_gpt_model.load_state_dict(torch.load(fn_list[0]))
			self.grasp_gpt_model.eval()

		if fn_list[1] != None:
			self.insert_gpt_model.load_state_dict(torch.load(fn_list[1]))
			self.insert_gpt_model.eval()

	def select_policy(self, policy_name=""):
		if policy_name == "reach":
			self.policy_name = "reach"

		if policy_name == "grasp":
			self.selected_model = self.grasp_model
			self.policy_name = "grasp"

		elif policy_name == "insert":
			self.selected_model = self.insert_model
			self.policy_name = "insert"

	def set_init_qpos(self, init_qpos):
		self.init_qpos[0:23] = init_qpos

if __name__ == '__main__':
	policy = NNController(num_actors=1, obs_dim=30)

	policy.load(["./utils/transformer_controller/models/model_7000_loss_-59.607639643351234.pt",
				"./utils/transformer_controller/models/model_2000_loss_-67.71043375651041.pt"])

	for i in range(50):
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
		targets = policy.predict(input=obs)[0]
		'''
			Action description: The network outputs a [-1, 1]^action_dim vector. 
		'''
