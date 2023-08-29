import numpy as np
import os
import sys
import pytorch3d.transforms as transform
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
	t = 0
	plt.ion()
	plt.show()
	plt.ylim(-2, 2)

	real_robot = np.load("/home/jmji/Downloads/real_robot_proprio_NEW(1).npy")
	sim_robot = np.load("./trajectory/real_qpos.npy")
	real_robot_1 = np.load("/home/jmji/Downloads/real_robot_proprio_NEW.npy")
	# real_robot_2 = np.load("/home/jmji/Downloads/real_robot_proprio.npy")

	target = np.load("./trajectory/qpos_targets.npy")

	# print(real_robot.shape)
	print(target.shape)
	sim_robot = sim_robot[:, 7:23]
	target = target[:, 7:23]

	plt.clf()
	# horizon = real_robot.shape[0]
	horizon = 146

	for i in range(16):
		plt.subplot(4, 4, i + 1)
		plt.plot(np.linspace(0, horizon - 1, horizon) * 0.0167, real_robot[:horizon, i], c='b', label="real_robot")
		# plt.plot(np.linspace(0, horizon - 1, horizon) * 0.0167, real_robot_1[:horizon, i], c='r', label="real_robot_1")
		plt.plot(np.linspace(0, horizon - 1, horizon) * 0.0167, real_robot_1[:horizon, i], c='y', label="real_before_robot")
		plt.plot(np.linspace(0, horizon - 1, horizon) * 0.0167, target[:horizon, i], c='g', label="targets")

		plt.title("joint_{}".format(i))

	plt.legend()
	plt.draw()
	plt.pause(100)

	pass

