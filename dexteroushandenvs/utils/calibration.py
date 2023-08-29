import numpy as np
import os
import sys
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import pytorch3d.transforms as transform
import matplotlib.pyplot as plt
import torch

def xyzw_to_wxyz(quat):
	# holy****, isaacgym uses xyzw format. pytorch3d uses wxyz format.
	new_quat = quat.clone()
	new_quat[:, :1] = quat[:, -1:]
	new_quat[:, 1:] = quat[:, :-1]
	return new_quat

EXISTING_SIM = None
SCREEN_CAPTURE_RESOLUTION = (1027, 768)

def _create_sim_once(gym, *args, **kwargs):
	global EXISTING_SIM
	if EXISTING_SIM is not None:
		return EXISTING_SIM
	else:
		EXISTING_SIM = gym.create_sim(*args, **kwargs)
		return EXISTING_SIM

class AllegroArm:
	def __init__(self, p_gain=10.0, d_gain=0.1, dt=0.0167, rl_device='cuda:0', sim_device='cuda:0',
				 graphics_device_id=0, virtual_screen_capture=False, headless=False, force_render=True):

		# Env base class initializations.
		self.control_freq_inv = 1
		self.headless = headless
		self.force_render = force_render
		self.device_id = 0
		self.device = "cuda:0"
		self.graphics_device_id = graphics_device_id
		self.physics_engine = gymapi.SIM_PHYSX
		self.virtual_display = None
		self.gym = gymapi.acquire_gym()
		self.set_sim_params(dt=dt)
		self.create_sim()

		self.sim_initialized = True

		self.aggregate_mode = 1 				# self.cfg["env"]["aggregateMode"]
		self.randomize_friction_lower = 0.3
		self.randomize_friction_upper = 3.0
		self.shadow_hand_dof_speed_scale = 20.0 # self.cfg["env"]["dofSpeedScale"]
		self.use_relative_control = False 		# self.cfg["env"]["useRelativeControl"]
		self.act_moving_average = 1 			# self.cfg["env"]["actionsMovingAverage"]
		self.debug_viz = True 					# self.cfg["env"]["enableDebugVis"]
		self.num_envs = 16 						# for each DOF.
		self.num_actions = 23
		self.torque_control = True 				# self.cfg["env"].get("torqueControl", True)
		self.robot_asset_files_dict = {
			"normal": "urdf/franka_description/robots/franka_panda_allegro.urdf",
			"thick": "urdf/franka_description/robots/franka_panda_allegro.urdf",
			"large":  "urdf/franka_description/robots/franka_panda_allegro.urdf"
		}
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
		self.hand_init_type = "default"
		self.hand_qpos_init_override = self.init_hand_qpos_override_dict[self.hand_init_type]
		self.palm_name = "palm"
		self.contact_sensor_names = ["link_1.0_fsr", "link_2.0_fsr", "link_3.0_tip_fsr",
									 "link_5.0_fsr", "link_6.0_fsr", "link_7.0_tip_fsr", "link_9.0_fsr",
									 "link_10.0_fsr", "link_11.0_tip_fsr", "link_14.0_fsr", "link_15.0_fsr",
									 "link_0.0_fsr", "link_4.0_fsr", "link_8.0_fsr", "link_13.0_fsr"]

		self.arm_sensor_names = ["link1", "link2", "link3", "link4", "link5", "link6"]
		self.finger_joint_names = ["joint_{}.0".format(i) for i in range(0, 16)]

		self.up_axis = 'z'
		self.use_vel_obs = False
		self.fingertip_obs = True
		self.robot_stiffness = 0.0
		self.dt = self.sim_params.dt

		self.create_envs(num_envs=16, spacing=0.75, num_per_row=4)
		self.gym.prepare_sim(self.sim)
		self.set_viewer()

		print("Videw created!")
		if self.viewer != None:
			cam_pos = gymapi.Vec3(1.0, 0.0, 1.0)
			cam_target = gymapi.Vec3(-1.0, 0.0, 0.5)
			self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

		if self.viewer:
			self.debug_contacts = np.zeros((16, 50), dtype=np.float32)

		# get gym GPU state tensors
		actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
		dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
		rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
		contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
		dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
		self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_arm_hand_dofs)

		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_dof_state_tensor(self.sim)
		self.gym.refresh_rigid_body_state_tensor(self.sim)
		self.gym.refresh_net_contact_force_tensor(self.sim)

		# create some wrapper tensors for different slices
		self.arm_hand_default_dof_pos = torch.zeros(self.num_arm_hand_dofs, dtype=torch.float, device=self.device)
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

		self.last_actions = torch.zeros((self.num_envs, 23), dtype=torch.float, device=self.device)
		self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
		self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
		self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
		self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
		self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
		self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

		self.p_gain_val = p_gain
		self.d_gain_val = d_gain
		self.p_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.p_gain_val
		self.d_gain = torch.ones((self.num_envs, self.num_actions), device=self.device, dtype=torch.float) * self.d_gain_val
		print('Initialization routine finished')
		self.post_init()

	def _create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
		"""Create an Isaac Gym sim object.

		Args:
			compute_device: ID of compute device to use.
			graphics_device: ID of graphics device to use.
			physics_engine: physics engine to use (`gymapi.SIM_PHYSX` or `gymapi.SIM_FLEX`)
			sim_params: sim params to use.
		Returns:
			the Isaac Gym sim object.
		"""
		sim = _create_sim_once(self.gym, compute_device, graphics_device, physics_engine, sim_params)
		if sim is None:
			print("*** Failed to create sim")
			quit()

		return sim

	def set_sim_params(self, dt):
		sim_params = gymapi.SimParams()

		# assign general sim parameters
		sim_params.dt = dt 			# 60Hz #config_sim["dt"]
		sim_params.num_client_threads = 0 	# config_sim.get("num_client_threads", 0)
		sim_params.use_gpu_pipeline = True 	# config_sim["use_gpu_pipeline"]
		sim_params.substeps = 1 			# config_sim.get("substeps", 2)

		# assign up-axis
		sim_params.up_axis = gymapi.UP_AXIS_Z

		# assign gravity
		sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8) #*config_sim["gravity"])

		# configure physics parameters
		setattr(sim_params.physx, "contact_collection", gymapi.ContactCollection(2))
		setattr(sim_params.physx, "solver_type", 1)
		setattr(sim_params.physx, "use_gpu", True)
		setattr(sim_params.physx, "num_position_iterations", 8)
		setattr(sim_params.physx, "num_velocity_iterations", 0)
		setattr(sim_params.physx, "max_gpu_contact_pairs", 8388608)
		setattr(sim_params.physx, "num_subscenes", 4)
		setattr(sim_params.physx, "contact_offset", 0.002)
		setattr(sim_params.physx, "rest_offset", 0)
		setattr(sim_params.physx, "bounce_threshold_velocity", 0.2)
		setattr(sim_params.physx, "max_depenetration_velocity", 1000.0)
		setattr(sim_params.physx, "default_buffer_size_multiplier", 5.0)
		self.sim_params = sim_params

	def create_sim(self):
		self.dt = self.sim_params.dt
		self.up_axis_idx = 2 		# index of up axis: Y=1, Z=2
		self.sim = self._create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
		self._create_ground_plane()

	def _create_ground_plane(self):
		plane_params = gymapi.PlaneParams()
		plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
		self.gym.add_ground(self.sim, plane_params)

	def create_envs(self, num_envs, spacing, num_per_row):
		lower = gymapi.Vec3(-spacing, -spacing, 0.0)
		upper = gymapi.Vec3(spacing, spacing, spacing)

		asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
		arm_hand_asset_file = self.robot_asset_files_dict["thick"]

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
		for i in range(23):
			robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
			if i < 6:
				robot_dof_props['velocity'][i] = 1.0
			else:
				robot_dof_props['velocity'][i] = 2.8

			print("Max effort: ", robot_dof_props['effort'][i])
			robot_dof_props['effort'][i] = 20.0

			robot_dof_props['friction'][i] = 0.3
			robot_dof_props['stiffness'][i] = 0  # self.robot_stiffness
			robot_dof_props['armature'][i] = 0.1

			if i < 6:
				robot_dof_props['damping'][i] = 100.0
			else:
				robot_dof_props['damping'][i] = 0.0  # 0.2 Early version is 0.2
			robot_lower_qpos.append(robot_dof_props['lower'][i])
			robot_upper_qpos.append(robot_dof_props['upper'][i])
			# robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
			# if i < 6:
			# 	robot_dof_props['velocity'][i] = 1.0
			# else:
			# 	robot_dof_props['velocity'][i] = 10.0
			#
			# print("Max effort: ", robot_dof_props['effort'][i])
			# robot_dof_props['effort'][i] = 2.0
			#
			# robot_dof_props['friction'][i] = 0.02
			# robot_dof_props['stiffness'][i] = 10.0 #self.robot_stiffness
			# robot_dof_props['armature'][i] = 0.001
			#
			# if i < 6:
			# 	robot_dof_props['damping'][i] = 100.0
			# else:
			# 	robot_dof_props['damping'][i] = 0.1  # 0.2 Early version is 0.2
			# robot_lower_qpos.append(robot_dof_props['lower'][i])
			# robot_upper_qpos.append(robot_dof_props['upper'][i])

		self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
		self.arm_hand_dof_lower_limits = to_torch(robot_lower_qpos, device=self.device)
		self.arm_hand_dof_upper_limits = to_torch(robot_upper_qpos, device=self.device)
		self.arm_hand_dof_lower_qvel = to_torch(-robot_dof_props["velocity"], device=self.device)
		self.arm_hand_dof_upper_qvel = to_torch(robot_dof_props["velocity"], device=self.device)

		print("DOF_LOWER_LIMITS", robot_lower_qpos)
		print("DOF_UPPER_LIMITS", robot_upper_qpos)

		# Set up default arm position.
		# Zhaoheng: We can set this to different positions...
		self.default_arm_pos = [0.0, -0.49826458111314524, -0.01990020486871322, -2.4732269941140346, -0.01307073642274261, 2.00396583422025, 1.5480939705504309]

		# We may need to some constraint for the thumb....
		for i in range(self.num_arm_hand_dofs):
			if i < 7:
				self.arm_hand_dof_default_pos.append(self.default_arm_pos[i])
			else:
				self.arm_hand_dof_default_pos.append(0.0)
			self.arm_hand_dof_default_vel.append(10.0)

		self.arm_hand_dof_default_pos = to_torch(self.arm_hand_dof_default_pos, device=self.device)
		self.arm_hand_dof_default_vel = to_torch(self.arm_hand_dof_default_vel, device=self.device)

		# Put objects in the scene.
		arm_hand_start_pose = gymapi.Transform()
		arm_hand_start_pose.p = gymapi.Vec3(0, 0.0, 0.0)
		arm_hand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

		# compute aggregate size
		max_agg_bodies = self.num_arm_hand_bodies + 2
		max_agg_shapes = self.num_arm_hand_shapes + 2

		self.arm_hands = []
		self.envs = []
		self.hand_start_states = []
		self.hand_indices = []
		self.fingertip_indices = []

		arm_hand_rb_count = self.gym.get_asset_rigid_body_count(arm_hand_asset)
		print("Enter Loop")
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

			# Do some friction randomization
			rand_friction =  np.random.uniform(self.randomize_friction_lower, self.randomize_friction_upper)
			hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, arm_hand_actor)
			for p in hand_props:
				p.friction = rand_friction
			self.gym.set_actor_rigid_shape_properties(env_ptr, arm_hand_actor, hand_props)

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

		# override!
		self.hand_override_info = [(self.gym.find_actor_dof_handle(env_ptr, arm_hand_actor, finger_name), self.hand_qpos_init_override[finger_name]) for finger_name in self.hand_qpos_init_override]
		print("PALM", self.palm_indices, self.sensor_handle_indices)
		self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
		self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)

	def get_finger_joint_info(self):
		joint_id = [self.gym.find_actor_dof_handle(self.envs[-1], self.arm_hands[-1], sensor_name)
					for sensor_name in self.finger_joint_names]
		return zip(self.finger_joint_names, joint_id), \
			   self.arm_hand_dof_lower_limits.detach().cpu().numpy(), \
			   self.arm_hand_dof_upper_limits.detach().cpu().numpy(),

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

	def update_controller(self):
		previous_dof_pos = self.arm_hand_dof_pos.clone()
		self.gym.refresh_dof_state_tensor(self.sim)
		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_rigid_body_state_tensor(self.sim)

		self.gym.refresh_net_contact_force_tensor(self.sim)
		self.gym.refresh_force_sensor_tensor(self.sim)
		self.gym.refresh_dof_force_tensor(self.sim)

		if self.torque_control:
			dof_pos = self.arm_hand_dof_pos
			dof_vel = (dof_pos - previous_dof_pos) / self.dt
			self.dof_vel_finite_diff = dof_vel.clone()
			torques = self.p_gain * (self.cur_targets - dof_pos) - self.d_gain * dof_vel
			# print(torch.norm(torques[0]))
			self.torques = torques.clone() #torch.clip(torques, -0.5, 0.5).clone()
			self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
		else:
			print("Set!")
			self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
		return

	def refresh_gym(self):
		self.gym.refresh_dof_state_tensor(self.sim)
		self.gym.refresh_actor_root_state_tensor(self.sim)
		self.gym.refresh_rigid_body_state_tensor(self.sim)
		self.gym.refresh_net_contact_force_tensor(self.sim)

	def reset_idx(self, env_ids):
		# reset the pd-gain.
		self.randomize_p_gain_lower = self.p_gain_val - 0.03
		self.randomize_p_gain_upper = self.p_gain_val + 0.03
		self.randomize_d_gain_lower = self.d_gain_val - 0.03
		self.randomize_d_gain_upper = self.d_gain_val + 0.03

		self.p_gain[env_ids] = torch_rand_float(
			self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), self.num_actions),
			device=self.device).squeeze(1)
		self.d_gain[env_ids] = torch_rand_float(
			self.randomize_d_gain_lower, self.randomize_d_gain_upper, (len(env_ids), self.num_actions),
			device=self.device).squeeze(1)

		self.arm_hand_dof_pos[env_ids, :] = self.arm_hand_dof_default_pos
		self.arm_hand_dof_vel[env_ids, :] = self.arm_hand_dof_default_vel
		self.prev_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_dof_default_pos
		self.cur_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_dof_default_vel

		hand_indices = self.hand_indices[env_ids].to(torch.int32)

		self.gym.set_dof_position_target_tensor_indexed(self.sim,
														gymtorch.unwrap_tensor(self.prev_targets),
														gymtorch.unwrap_tensor(hand_indices), len(env_ids))

		for env_id in env_ids:
			for (idx, qpos) in self.hand_override_info:
				self.dof_state[env_id * self.num_arm_hand_dofs + idx, 0] = qpos

		self.gym.set_dof_state_tensor_indexed(self.sim,
											  gymtorch.unwrap_tensor(self.dof_state),
											  gymtorch.unwrap_tensor(hand_indices), len(env_ids))

	def pre_physics_step(self, actions):
		self.actions = actions.clone().to(self.device)

		if self.use_relative_control:
			self.actions = self.actions * self.act_moving_average + self.last_actions * (1.0 - self.act_moving_average)
			targets = self.arm_hand_dof_pos + 3.1415926 / 6 * self.actions
			self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
																		  self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
																		  self.arm_hand_dof_upper_limits[self.actuated_dof_indices])
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

	def post_physics_step(self):
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

	def simulate(self, actions):
		action_tensor = torch.clamp(actions, -1.0, 1.0) #-self.clip_actions, self.clip_actions)
		# apply actions
		self.pre_physics_step(action_tensor)

		# step physics and render each frame
		for i in range(self.control_freq_inv):
			if self.force_render:
				self.render()

			self.gym.fetch_results(self.sim, True)
			self.update_controller()
			self.gym.simulate(self.sim)

		self.gym.fetch_results(self.sim, True)
		# compute observations, rewards, resets, ...
		self.post_physics_step()

	def goto_qpos(self, qpos, control_steps):
		all_trajectory = []
		self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(qpos,
																	  self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
																	  self.arm_hand_dof_upper_limits[self.actuated_dof_indices])
		for i in range(control_steps):
			if self.force_render:
				self.render()

			self.gym.fetch_results(self.sim, True)
			self.update_controller()
			self.gym.simulate(self.sim)
			all_trajectory.append(self.arm_hand_dof_pos.detach().cpu().numpy())

		self.gym.fetch_results(self.sim, True)
		# compute observations, rewards, resets, ...
		self.post_physics_step()
		return np.array(all_trajectory)

	def goto_qpos_seq(self, qpos_seqs, control_steps):
		all_trajectory = []

		for i in range(control_steps):
			self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
				qpos_seqs[:, i, :],
				self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
				self.arm_hand_dof_upper_limits[self.actuated_dof_indices]
			)

			if self.force_render:
				self.render()

			self.gym.fetch_results(self.sim, True)
			self.update_controller()
			self.gym.simulate(self.sim)
			all_trajectory.append(self.arm_hand_dof_pos.detach().cpu().numpy())

		self.gym.fetch_results(self.sim, True)
		# compute observations, rewards, resets, ...
		self.post_physics_step()
		return np.array(all_trajectory)

	def get_valid_target(self, target_qpos):
		return tensor_clamp(target_qpos,
							self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
							self.arm_hand_dof_upper_limits[self.actuated_dof_indices])
	def get_current_qpos(self):
		return self.arm_hand_dof_pos.clone()

	def set_viewer(self):
		"""Create the viewer."""

		# todo: read from config
		self.enable_viewer_sync = True
		self.viewer = None

		# if running with a viewer, set up keyboard shortcuts and camera
		if self.headless == False:
			# subscribe to keyboard shortcuts
			self.viewer = self.gym.create_viewer(
				self.sim, gymapi.CameraProperties())
			self.gym.subscribe_viewer_keyboard_event(
				self.viewer, gymapi.KEY_ESCAPE, "QUIT")
			self.gym.subscribe_viewer_keyboard_event(
				self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

			# set the camera position based on up axis
			sim_params = self.gym.get_sim_params(self.sim)
			if sim_params.up_axis == gymapi.UP_AXIS_Z:
				cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
				cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
			else:
				cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
				cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

			self.gym.viewer_camera_look_at(
				self.viewer, None, cam_pos, cam_target)

	def render(self, mode="rgb_array"):
		"""Draw the frame to the viewer, and check for keyboard events."""
		if self.viewer:
			# check for window closed
			if self.gym.query_viewer_has_closed(self.viewer):
				sys.exit()

			# check for keyboard events
			for evt in self.gym.query_viewer_action_events(self.viewer):
				if evt.action == "QUIT" and evt.value > 0:
					sys.exit()
				elif evt.action == "toggle_viewer_sync" and evt.value > 0:
					self.enable_viewer_sync = not self.enable_viewer_sync

			# fetch results
			if self.device != 'cpu':
				self.gym.fetch_results(self.sim, True)

			# step graphics
			if self.enable_viewer_sync:
				self.gym.step_graphics(self.sim)
				self.gym.draw_viewer(self.viewer, self.sim, True)

				# Wait for dt to elapse in real time.
				# This synchronizes the physics simulation with the rendering rate.
				self.gym.sync_frame_time(self.sim)

			else:
				self.gym.poll_viewer_events(self.viewer)

			if self.virtual_display and mode == "rgb_array":
				img = self.virtual_display.grab()
				return np.array(img)

	def generate_random_valid_qpos(self, env_ids):
		pass


class ImpulseFunc:
	def __init__(self, shift, horizon):
		self.shift = shift
		self.horizon = horizon
	def __call__(self, x):
		all_time_steps = []
		for t in range(self.horizon):
			all_time_steps.append(self.shift + x)
		all_time_steps = torch.stack(all_time_steps, dim=0)
		return all_time_steps.permute(1, 0, 2)


class SinWaveFunc:
	def __init__(self, shift, T=0.1, dt=0.0167, horizon=60):
		self.shift = shift
		self.T = T
		self.dt = dt
		self.horizon = horizon
	def __call__(self, x):
		import math
		all_time_steps = []
		for t in range(self.horizon):
			shift = self.shift * math.sin(6.28 * t * self.dt / self.T)
			all_time_steps.append(shift + x)

		all_time_steps = torch.stack(all_time_steps, dim=0)
		return all_time_steps.permute(1, 0, 2)

if __name__ == '__main__':
	# Init the environment:
	env = AllegroArm(p_gain=100.0, d_gain=4.0)
	env.reset_idx(torch.tensor(list(range(0, 16))).long())

	finger_mapping, joint_lower, joint_upper = env.get_finger_joint_info()
	finger_mapping_dict = {}
	for fm in finger_mapping:
		finger_mapping_dict[fm[1]] = fm[0]

	init_qpos = env.get_current_qpos()
	all_tests = [(init_qpos, ImpulseFunc(0.3, horizon=60)),
				 (init_qpos, SinWaveFunc(0.3, T=0.5, horizon=60)), (init_qpos, SinWaveFunc(0.1, 60))]

	t = 0
	plt.ion()
	plt.show()
	plt.ylim(-2, 2)

	for test_idx, test in enumerate(all_tests):
		test_init_qpos, test_function = test[0], test[1]
		target_qpos = test_init_qpos.clone()
		target_qpos = test_function(target_qpos)
		target_qpos = env.get_valid_target(target_qpos)
		print("Goto Init Test QPos")
		env.goto_qpos(test_init_qpos, 60)

		print("Start Testing")
		results = env.goto_qpos_seq(target_qpos, 60) # 2s. Numpy array: [T, JOINT_QPOS, JOINT_QPOS]
		print("Done")

		plt.clf()
		horizon = results.shape[0]
		for i in range(16):
			plt.subplot(4, 4, i + 1)
			plt.plot(np.linspace(0, horizon - 1, horizon) * 0.0167, results[:, i, i + 6], c='b', label="tracking")
			plt.plot(np.linspace(0, horizon - 1, horizon) * 0.0167, target_qpos[i, :, i + 6].cpu().detach().numpy(), c='r', label="target")
			plt.title(finger_mapping_dict[i + 7])

		# np.save("")

		plt.legend()
		plt.draw()
		plt.pause(100)
	# for i in range(1000):
	# 	action = torch.clip(torch.randn(4, 22), -1.0, 1.0)
	# 	env.simulate(action)
	# 	print(env.get_current_qpos())
	# # Measure the movement.
	pass

