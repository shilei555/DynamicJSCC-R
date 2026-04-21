import torch
from torch import nn
from model.modules.GDN import GDN_layer as GDN


class RTCB(nn.Module):
    """
    Residual Transposed Convolution Block
    用于解码器上采样
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activate_func='prelu'):
        super(RTCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = kernel_size // 2
        self.deconv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                          output_padding=1 if stride > 1 else 0)
        self.igdn1 = GDN(out_channels, inverse=True)
        self.prelu1 = nn.PReLU(out_channels)
        self.deconv2 = nn.ConvTranspose2d(out_channels, out_channels, 1)
        self.igdn2 = GDN(out_channels, inverse=True)

        # 捷径连接
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 1, stride=stride, padding=0,
                                   output_padding=1 if stride > 1 else 0),
                # GDN(out_channels, inverse=True)
            )
        if activate_func == 'prelu':
            self.activation = nn.PReLU(out_channels)
        elif activate_func == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.deconv1(x)
        out = self.igdn1(out)
        out = self.prelu1(out)
        out = self.deconv2(out)
        out = self.igdn2(out)
        out = out + residual
        return self.activation(out)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    rtcb = RTCB(48, 48, stride=2).to(device)
    x = torch.randn(2, 48, 16, 16).to(device)
    out = rtcb(x)
    print(out.shape)  # 应输出: torch.Size([2, 64, 32, 32])
