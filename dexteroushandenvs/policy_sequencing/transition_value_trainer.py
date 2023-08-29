import h5py
import torch
from isaacgym import gymutil

import os
import random
import torch.optim as optim
import numpy as np

import cv2
from PIL import Image

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * _sqrt_positive_part(1 + m00 + m11 + m22)
    x = 0.5 * _sqrt_positive_part(1 + m00 - m11 - m22)
    y = 0.5 * _sqrt_positive_part(1 - m00 + m11 - m22)
    z = 0.5 * _sqrt_positive_part(1 - m00 - m11 + m22)
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o0, o1, o2, o3), -1)

def _sqrt_positive_part(x):
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def _copysign(a, b):
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)

class TValue_Trainer():
    def __init__(self, file_path) -> None:
        self.data_recorder = 1
        self.device = "cuda:0"
            
        with h5py.File(file_path, "r") as f:
            self.f = f
            self.succ_data = self.f["data"]["success_dataset"]
            self.fail_data = self.f["data"]["failure_dataset"]

            list_of_names = []
            self.f.visit(list_of_names.append)

            self.succ_dataset_idx = []
            self.fail_dataset_idx = []

            for name in list_of_names:
                name = name.split("/")
                if len(name) != 3:
                    continue
                else:
                    name = name[-1]
                name_split = name.split("_")

                if name_split[1] == "success":
                    self.succ_dataset_idx.append(int(name_split[0][:-2]))
                elif name_split[1] == "failure":
                    self.fail_dataset_idx.append(int(name_split[0][:-2]))

            self.num_success_data = len(self.succ_dataset_idx)
            self.num_failure_data = len(self.fail_dataset_idx)

            self.success_data = torch.zeros((self.num_success_data, 4), device=self.device)
            self.failure_data = torch.zeros((self.num_failure_data, 4), device=self.device)

            self.input_dim = 4

            for i in self.succ_dataset_idx:
                # print(self.succ_data["{}th_success_data".format(i)][:])
                self.success_data[i] = torch.tensor(self.succ_data["{}th_success_data".format(i)][:], device=self.device)
            for i in self.fail_dataset_idx:
                self.failure_data[i] = torch.tensor(self.fail_data["{}th_failure_data".format(i)][:], device=self.device)

            self.failure_data = self.failure_data.clone()

            self.valid_data = self.success_data[-100:].clone()
            self.success_data = self.success_data[:-100].clone()

            self.num_success_data = self.success_data.shape[0]
            self.num_failure_data = self.failure_data.shape[0]

        self.f.close()

    def init_TValue_function(self, task_name, rollout):
        from dexteroushandenvs.policy_sequencing.terminal_value_function import GraspInsertTValue

        self.t_value = GraspInsertTValue(input_dim=self.input_dim, output_dim=2).to(self.device)
        for param in self.t_value.parameters():
            param.requires_grad_(True)
    
        self.t_value_optimizer = optim.Adam(self.t_value.parameters(), lr=0.001)
        self.t_value_save_path = "./intermediate_state/{}_t_value/".format(task_name)
        os.makedirs(self.t_value_save_path, exist_ok=True)
        self.bce_logits_loss = torch.nn.BCEWithLogitsLoss()

        self.batch_size = 1024
        self.succ_batch_size = 512
        self.fail_batch_size = 512
        self.valid_batch_size = 100

        self.rollout = rollout

        self.success_buf = torch.zeros((self.batch_size, 2), dtype=torch.float32, device=self.device)
        self.success_buf[:self.succ_batch_size, 1] = torch.ones_like(self.success_buf[:self.succ_batch_size, 1])
        self.success_buf[self.succ_batch_size:, 0] = torch.ones_like(self.success_buf[self.succ_batch_size:, 0])

        self.succ_rand_range = range(0, self.num_success_data)
        self.fail_rand_range = range(0, self.num_failure_data)

        self.t_value_obs_buf = torch.zeros((self.batch_size, self.input_dim), dtype=torch.float32, device=self.device)
        self.valid_t_value_obs_buf = torch.zeros((self.valid_batch_size, self.input_dim), dtype=torch.float32, device=self.device)

    def train_rollout(self):
        for iter in range(self.rollout):
            self.rand_float = torch_rand_float(-1, 1, (self.succ_batch_size, 12), device=self.device).squeeze(-1)

            self.succ_rand = random.sample(self.succ_rand_range, int(self.batch_size / 2))
            self.fail_rand = random.sample(self.fail_rand_range, int(self.batch_size / 2))

            self.t_value_obs_buf[:self.succ_batch_size] = self.success_data[self.succ_rand] + self.rand_float[:, 0:4] * 0.05
            self.t_value_obs_buf[:self.succ_batch_size] /= torch.norm(self.t_value_obs_buf[:self.succ_batch_size], dim=-1, keepdim=True)

            self.t_value_obs_buf[self.succ_batch_size:] = self.failure_data[self.fail_rand] + self.rand_float[:, 4:8] * 0.05
            self.t_value_obs_buf[self.succ_batch_size:] /= torch.norm(self.t_value_obs_buf[self.succ_batch_size:], dim=-1, keepdim=True)

            # forward
            self.predict_success_confident = self.t_value(self.t_value_obs_buf)

            # update v value
            loss = self.bce_logits_loss(self.predict_success_confident, self.success_buf)
            self.t_value_optimizer.zero_grad()
            loss.backward()
            self.t_value_optimizer.step()

            print("loss: ", loss.item())
            print("t_value_udpate_iter: ", iter)
            if iter % 10000 == 0:
                self.valid_rand = random.sample(range(0, self.valid_batch_size), self.valid_batch_size)

                self.valid_t_value_obs_buf = self.valid_data[self.valid_rand]

                self.valid_predict_success_confident = self.t_value(self.valid_t_value_obs_buf).detach()
                
                predict_success_count = 0
                for i in range(self.valid_batch_size):
                    self.valid_predict_success_confident = torch.sigmoid(self.valid_predict_success_confident)
                    if self.valid_predict_success_confident[i, 0] < self.valid_predict_success_confident[i, 1]:
                        predict_success_count += 1

                self.valid_t_value_success_rate = predict_success_count / self.valid_batch_size
                print('valid_t_value_success_rate: ', self.valid_t_value_success_rate)
                torch.save(self.t_value.state_dict(), self.t_value_save_path + "/grasp_insert_TValue_{}_{}.pt".format(iter, self.valid_t_value_success_rate))

    def test_model(self):
        self.data = "/home/jmji/Downloads/data2.hdf5"
        from dexteroushandenvs.policy_sequencing.terminal_value_function import GraspInsertTValue

        self.t_value = GraspInsertTValue(input_dim=4, output_dim=2).to(self.device)
        self.t_value.load_state_dict(torch.load("./intermediate_state/grasping_insertion_t_value/grasp_insert_TValue_10000_0.834.pt", map_location='cuda:0'))
        self.t_value.to(self.device)
        self.t_value.eval()
        with h5py.File(self.data, "r") as f:
            self.f = f

            list_of_names = []
            self.f.visit(print)

            print(self.f["images"][:].shape)

            for i in range(self.f["images"][:].shape[0]):
                pil_image = np.uint8(self.f["images"][i])

                print("pose_input: ", self.f["pose_input"][i])
                # print("model_output: ", self.f["model_output"][i])

                input_pose = self.real_world_to_ig_pose(self.f["pose_input"][i])

                self.valid_predict_success_confident = self.t_value(input_pose).detach()
                self.valid_predict_success_confident = torch.sigmoid(self.valid_predict_success_confident)

                print(self.valid_predict_success_confident[0, 1])

                if 0.8 < self.valid_predict_success_confident[0, 1]:
                    pil_image[:, :, 2] += 64
                    np.clip(pil_image[:, :, 0], 0, 255)
                
                cv2.imshow("1", pil_image)
                cv2.waitKey(2)

            self.f.close()

    def real_world_to_ig_pose(self, input_pose):
        input_pose = torch.tensor(input_pose, dtype=torch.float32, device=self.device).clone()
        tem_pose = input_pose.clone()
        # input_pose[0:3] = tem_pose[1:4].clone()
        # input_pose[3] = tem_pose[0].clone()                
        input_matrix = quaternion_to_matrix(input_pose)                
        real_to_ig_rotation_matrix = np.array([[0, -1, 0],
                                            [0, 0, -1],
                                            [1, 0, 0]])                
        input_matrix = input_matrix.cpu().numpy()                
        input_matrix = (input_matrix.T @ real_to_ig_rotation_matrix).T                
        input_pose = matrix_to_quaternion(torch.tensor(input_matrix, dtype=torch.float32, device=self.device).unsqueeze(0).clone())                
        tem_pose = input_pose.clone()
        # input_pose[:, 1:4] = tem_pose[:, 0:3].clone()
        # input_pose[:, 0] = tem_pose[:, 3].clone()
        input_pose[:, 0:3] = tem_pose[:, 1:4].clone()
        input_pose[:, 3] = tem_pose[:, 0].clone()                
        
        return input_pose     
           
if __name__ == "__main__":
    custom_parameters = [
        {"name": "--task", "type": str, "default": "",
            "help": "Only for policy sequencing"},
        {"name": "--rollout", "type": int, "default": 100000,
            "help": "Only for policy sequencing"},
        ]
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)
    
    trainer = TValue_Trainer("./intermediate_state/{}_datasets.hdf5".format(args.task))
    trainer.init_TValue_function(args.task, args.rollout)
    trainer.train_rollout()
