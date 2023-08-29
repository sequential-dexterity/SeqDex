from rl_games.algos_torch.model_builder import ModelBuilder
import yaml


def read_yaml(yaml_path):
	with open(yaml_path, "r") as stream:
		try:
			result = yaml.safe_load(stream)
			print(result)
		except yaml.YAMLError as exc:
			print(exc)
	return result


def build_network(cfg_yaml_path, obs_dim):
	cfg = read_yaml(cfg_yaml_path)
	model_builder = ModelBuilder()
	model = model_builder.load(cfg['params'])

	config = {
		'actions_num': 23,
		'input_shape': (obs_dim, ),
		'num_seqs': 1,
		'value_size': 1,
		'normalize_value': False, #self.normalize_value,
		'normalize_input': False #,self.normalize_input,
	}

	network = model.build(config)
	print(network)
	return network

if __name__ == '__main__':
	read_yaml('./robot_controller/network.yaml')
