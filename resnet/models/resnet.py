"""Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActivation(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, kernel_size, stride=1, padding=0, activation=F.relu):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_chans_in)
        self.activation = activation
        self.conv = nn.Conv2d(n_chans_in, n_chans_out, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)

    def forward(self, x):
        return self.conv(self.activation(self.bn(x)))


class ResBlock(nn.Module):
    def __init__(self, n_chans_in, n_chans_out, stride=1):
        super().__init__()
        self.fe = nn.Sequential(PreActivation(n_chans_in, n_chans_out, kernel_size=3, stride=stride, padding=1),
                                PreActivation(n_chans_out, n_chans_out, kernel_size=3, padding=1))

        if n_chans_in != n_chans_out:
            # self.avg_pool = nn.AvgPool2d(2)
            # self.shortcut = lambda x: torch.cat([self.avg_pool(x)] * 2, dim=1)
            # Variable(torch.zeros(N, pad, H, W))
            self.shortcut = nn.Conv2d(n_chans_in, n_chans_out, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        return self.fe(x) + self.shortcut(x)


class GlobalAvgPool2d(nn.Module):
    def forward(self, x):
        batch_size, n_chans, *spatial_size = x.size()
        return F.avg_pool2d(x, kernel_size=spatial_size).view(batch_size, n_chans)


class Resnet110(nn.Module):
    def __init__(self, n_chans_out=10):
        super().__init__()
        self.n_chans = 16

        self.fe = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
                                self._make_block(ResBlock, n_chans=16, n_blocks=18, stride=1),
                                self._make_block(ResBlock, n_chans=32, n_blocks=18, stride=2),
                                self._make_block(ResBlock, n_chans=64, n_blocks=18, stride=2),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                GlobalAvgPool2d(),
                                nn.Linear(64, n_chans_out),
                                nn.BatchNorm1d(n_chans_out, n_chans_out))

    def _make_block(self, block, n_chans, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.n_chans, n_chans, stride))
            self.n_chans = n_chans
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.fe(x)
