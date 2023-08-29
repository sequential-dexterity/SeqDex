import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def pool2x2(x):
    return nn.MaxPool2d(kernel_size=2, stride=2)(x)


def upsample2(x):
    return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class FeatureTunk(nn.Module):

    def __init__(self, pretrained=True, input_dim=6, output_dim=1):
        super(FeatureTunk, self).__init__()

        self.rgb_extractor = BasicBlock(input_dim, input_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        # self.dense121 = torchvision.models.densenet.densenet121(pretrained=pretrained).features
        # self.dense121.conv0 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.resnet18 = torchvision.models.resnet.resnet18(pretrained=pretrained)
        # self.resnet18.fc = nn.Linear(512, 64)
        self.mobilenetv3_feat = torchvision.models.mobilenet.mobilenet_v3_small(pretrained=pretrained).features
        # origin: 1
        self.mobilenetv3_avgpool = nn.AdaptiveAvgPool2d(4)
        # origin: 576
        self.mobilenetv3_classifier = nn.Sequential(
            nn.Linear(96, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        # x = self.mobilenetv3_feat(self.rgb_extractor(x))
        x = self.rgb_extractor(x)
        x = self.mobilenetv3_avgpool(x)
        x = torch.flatten(x, 1)
        x = self.mobilenetv3_classifier(x)
        
        return x


class MyNetWork(nn.Module):
    def __init__(self, output):
        super(MyNetWork, self).__init__()

        self.feature_tunk = FeatureTunk()

        self.linear1 = nn.Linear(12, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 64)
        self.selu = nn.SELU()

        self.linear_output = nn.Linear(128, output)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='selu')

    def forward(self, x):
        rgb_img = x[:, :-12]
        aux_obs = x[:, -12:]

        # 1 * 8 * 8 feat
        feat_out = self.feature_tunk(rgb_img)
        aux_obs_out = self.linear1(aux_obs)
        aux_obs_out = self.selu(aux_obs_out)
        aux_obs_out = self.linear2(aux_obs_out)
        aux_obs_out = self.selu(aux_obs_out)
        aux_obs_out = self.linear3(aux_obs_out)
        aux_obs_out = self.selu(aux_obs_out)

        output = torch.cat((aux_obs_out, feat_out), dim=1)
        output = self.linear_output(output)

        return output





        
