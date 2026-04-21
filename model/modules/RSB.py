from torch import nn
import torch.nn.functional as F


class RSB(nn.Module):
    """
    Residual Block for ResNet18 Classifier
    论文公式 (21)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(RSB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class RSB_Down(nn.Module):
    """带下采样的 RSB"""

    def __init__(self, in_channels, out_channels):
        super(RSB_Down, self).__init__()
        self.rsb1 = RSB(in_channels, out_channels, stride=2)
        self.rsb2 = RSB(out_channels, out_channels, stride=1)

    def forward(self, x):
        return self.rsb2(self.rsb1(x))
