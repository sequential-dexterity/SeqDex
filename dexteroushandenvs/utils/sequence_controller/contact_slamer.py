import torch
from torch import nn
import torch.nn.functional as F

class ContactSLAMer(nn.Module):
    def __init__(self, tactile, output_pose_size) :
        super(ContactSLAMer, self).__init__()
        self.linear1 = nn.Linear(tactile, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, output_pose_size)

    def forward(self, contact):
        x = F.elu(self.linear1(contact))
        x = F.elu(self.linear2(x))
        x = F.elu(self.linear5(x))
        output_pose = self.output_layer(x)
        # normalize output_quat
        output_pose[:, 0:4] = output_pose[:, 0:4] / torch.norm(output_pose[:, 0:4].detach(), dim=-1, keepdim=True)

        return output_pose, x