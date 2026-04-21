import torch
from torch import nn
import torch.nn.functional as F


class AFB(nn.Module):
    """
    Attention Feature Block
    论文核心模块：基于 SNR 的通道注意力机制
    公式 (8)-(11)
    """

    def __init__(self, channels):
        super(AFB, self).__init__()
        self.channels = channels

        self.fc1 = nn.Linear(channels + 1, channels // 2)
        self.fc2 = nn.Linear(channels // 2, channels)

    def forward(self, x, snr):
        # x: [B, C, H, W], snr: [B] 或标量
        B, C, H, W = x.shape

        # 公式 (8): Global Average Pooling -> 通道均值
        mu_g = x.mean(dim=[2, 3])  # [B, C]

        # 公式 (9): 拼接 SNR
        if snr.dim() == 0:
            snr = snr.unsqueeze(0).expand(B)
        # snr = snr / 28.0 # 归一化 SNR，统一到 [0, 1] 范围内，28dB 是训练时SNR的上限
        mu = torch.cat([snr.unsqueeze(1), mu_g], dim=1)  # [B, C+1]

        # 公式 (10): 非线性变换
        w = F.relu(self.fc1(mu))
        w = torch.sigmoid(self.fc2(w))  # [B, C]

        # 公式 (11): 通道级重标定
        return x * w.view(B, C, 1, 1)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    afb = AFB(128).to(device)
    x = torch.randn(2, 128, 16, 16).to(device)
    snr = torch.tensor([20.0, 30.0]).to(device)  # 示例 SNR
    out = afb(x, snr)
    print(out.shape)  # 应输出: torch.Size([2, 64, 16, 16])
