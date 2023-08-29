import torch
import copy
import numpy as np
from .nn_builder import build_network
from rl_games.algos_torch import torch_ext

class NNController:
	def __init__(self, num_actors=1, config_path='./robot_controller/network.yaml', obs_dim=81, device="cuda:0"):
		self.model = build_network(config_path, obs_dim=obs_dim)
		self.one_step_obs_dim = obs_dim

		self.states = None
		self.device = device
		self.time_step = 0

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

	def predict(self, observation, deterministic=False):
		if type(observation) != torch.Tensor:
			observation = torch.from_numpy(observation).float()

		obs = self._preproc_obs(observation)

		input_dict = {
			'is_train': False,
			'prev_actions': None,
			'obs': obs,
			'rnn_states': self.states
		}
		with torch.no_grad():
			res_dict = self.model(input_dict)
		mu = res_dict['mus']
		action = res_dict['actions']
		self.states = res_dict['rnn_states']
		if deterministic:
			current_action = mu
		else:
			current_action = action

		self.last_action = current_action

		current_action = torch.clip(current_action, -1.0, 1.0)
		return current_action.detach()

	def load(self, fn):
		checkpoint = torch_ext.load_checkpoint(fn)
		self.model.load_state_dict(checkpoint['model'])
		self.model.to(self.device)
		# if self.normalize_input and 'running_mean_std' in checkpoint:
		#	self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])


if __name__ == '__main__':
	policy = NNController(num_actors=1,
						  config_path='./test_network.yaml', obs_dim=38)

	policy.load('./models/last_AllegroHandLegoTest_ep_3000_rew_89.97035.pth')

	for i in range(150):
		obs = np.random.uniform(0, 1, 38)

		''' 
			Observation description: 
			
			DIMENSION |	DESCRIPTION	 |	PROGRAMMING HINT
			-------------------------------------------------------------------------------------------
			[0:16]:   |	current_hand_qpos |	(SET BY USER. 			Set it by the previous protocol)
			[16:23]:  |	hand_pose 	      |	(SET BY USER. 			Set it by the previous protocol)
			[23:26]:  |	object init pos   |	(SET BY USER. 			Set it by the previous protocol)
			[26:30]:  |	object_init_rot   |	(SET BY USER. 			Set it to 0 now)
			[30:33]:  |	hand_pos - object_init_pos   |	(SET BY USER. 			Set it by the previous protocol)
			[33:37]:  |	quat_mul(hand_rot, quat_conjugate(object_init_rot))   |	(SET BY USER. 			Set it by the previous protocol)
			[37:38]:  |	time step    |	(AUTOSET BY CONTROLLER. Leave it blank)
		'''

		action = policy.predict(observation=np.clip(obs, 5, -5))[0]

		'''
			Action description: The network outputs a [-1, 1]^action_dim vector. 
		'''

		print(action)
