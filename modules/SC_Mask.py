import math

import torch
from torch import nn


class SC_Mask(nn.Module):
    """
    Semantic Code Mask 机制
    公式 (12)-(13): 根据 CR 动态选择传输的语义码通道数
    """

    def __init__(self, K_max):
        super(SC_Mask, self).__init__()
        self.K_max = K_max

    def forward(self, x, cr):
        B, C, H, W = x.shape
        # 沿H和W展平
        x = x.view(B, C, -1)  # [B, C, H*W]
        K = torch.ceil(cr * self.K_max)
        indices = torch.arange(C, device=x.device).view(1, -1, 1)  # [1, C, 1]
        mask = indices < K.view(B, 1, 1)  # [B, C, 1]
        # 应用掩码（未保留的通道置零）
        x = x * mask.float()
        return x, mask, H, W


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sc_mask = SC_Mask(K_max=128).to(device)
    x = torch.rand(2, 128, 16, 16).to(device)
    cr = torch.rand(2).to(device)  # 示例 CR
    out = sc_mask(x, cr)
    print(out.shape)
