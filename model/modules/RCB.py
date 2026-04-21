from torch import nn
from .GDN import GDN_layer as GDN
class RCB(nn.Module):
    """
    Residual Convolution Block
    论文中的基础卷积残差块，使用 GDN + PReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(RCB, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = kernel_size // 2

        # 主路径: 3x3 conv + 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding)
        self.gdn1 = GDN(out_channels)
        self.prelu1 = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.gdn2 = GDN(out_channels)
        # 捷径连接，如果输入输出通道数或步幅不匹配，则使用1x1卷积调整维度，否则使用恒等映射
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                # GDN(out_channels)
            )
        self.prelu_out = nn.PReLU(out_channels)

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.gdn1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.gdn2(out)

        out = out + residual
        return self.prelu_out(out)