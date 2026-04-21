import torch
from torch import nn, tensor
import torch.nn.functional as F
from model.modules.AFB import AFB
from model.modules.RCB import RCB
from model.modules.RSB import RSB, RSB_Down
from model.modules.RTCB import RTCB
from model.modules.SC_Mask import SC_Mask
from model.modules.PowerNomalization import Power_norm, Power_norm_VLC


class Encoder(nn.Module):
    """
    DynamicJSCC-R 编码器
    架构参考论文 Table I 和 Fig. 2
    """

    def __init__(self, in_channels=3, c=128, K_max=48):
        super(Encoder, self).__init__()
        self.K_max = K_max

        # RCB Down 1: 32x32 -> 16x16, 3 -> c
        self.rcb_down1 = RCB(in_channels, c, kernel_size=3, stride=2)

        # AFB 1 (低层级语义层)
        self.afb1 = AFB(c)

        # RCB Down 2: 16x16 -> 8x8, c -> c
        self.rcb_down2 = RCB(c, c, kernel_size=3, stride=2)

        # RCB x2: 保持 8x8
        self.rcb1 = RCB(c, c, kernel_size=3, stride=1)
        self.rcb2 = RCB(c, K_max, kernel_size=3, stride=1)

        # AFB 2 (高层级语义层)
        self.afb2 = AFB(K_max)

    def forward(self, x, snr):
        # x: [B, 3, H, W], snr: 标量或 [B]
        out = self.rcb_down1(x)
        out = self.afb1(out, snr)
        out = self.rcb_down2(out)
        out = self.rcb1(out)
        out = self.rcb2(out)
        out = self.afb2(out, snr)
        return out


class Decoder(nn.Module):
    """
    DynamicJSCC-R 解码器
    """

    def __init__(self, out_channels=3, c=128, K_max=48):
        super(Decoder, self).__init__()
        self.K_max = K_max
        self.c = c

        # AFB 3
        self.afb3 = AFB(K_max)
        # RTCB Up x2: 8x8 -> 16x16 -> 32x32
        self.rtcb_up1 = RTCB(K_max, K_max, kernel_size=3, stride=2)
        self.rtcb_up2 = RTCB(K_max, c, kernel_size=3, stride=2)
        self.rtcb1 = RTCB(c, c, kernel_size=3, stride=1)
        self.rtcb2 = RTCB(c, out_channels, kernel_size=3, stride=1, activate_func='sigmoid')
        # AFB 4
        self.afb4 = AFB(c)

    def forward(self, x, snr):
        out = self.afb3(x, snr)
        out = self.rtcb_up1(out)
        out = self.rtcb_up2(out)
        out = self.rtcb1(out)
        out = self.afb4(out, snr)
        out = self.rtcb2(out)
        return out


class Classifier(nn.Module):
    """
    分类器 - 基于 ResNet18 架构
    论文 Table I: CIFAR-10 分类器
    """

    def __init__(self, in_channels=3, num_classes=10):
        super(Classifier, self).__init__()

        # 输入层: 3x3 Conv + BN + ReLU
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # RSB x2: 保持 32x32, 64 channels
        self.rsb1 = RSB(64, 64)
        self.rsb2 = RSB(64, 64)
        # (RSB Down + RSB) x3
        self.layer1 = RSB_Down(64, 128)  # 32x32 -> 16x16
        self.layer2 = RSB_Down(128, 256)  # 16x16 -> 8x8
        self.layer3 = RSB_Down(256, 512)  # 8x8 -> 4x4

        # Average Pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.rsb1(out)
        out = self.rsb2(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def AWGN_channel(x, snr_db):
    snr_linear = 10 ** (snr_db / 10.0)
    noise_std = torch.sqrt(1.0 / snr_linear)
    noise_std = noise_std.view(-1, 1, 1)  # 确保噪声标准差与输入张量的维度兼容
    noise = noise_std * torch.randn_like(x)
    y = x + noise
    return y


def mask_gen(x, cr):
    """
    生成语义码掩码 (Semantic Code Mask)

    Args:
        x: 输入信号 [B, K_max, L]
        cr: 压缩率 [B] 或标量

    Returns:
        mask: 掩码张量 [B, K_max, 1]，在通道维度上应用
    """
    B, K_max, L = x.shape

    # 处理标量 cr
    if not torch.is_tensor(cr):
        cr = torch.full((B,), cr, device=x.device)
    elif cr.dim() == 0:
        cr = cr.view(1).expand(B)

    # 计算有效通道数 [B]
    K = torch.ceil(cr * K_max).long()
    K = torch.clamp(K, min=1, max=K_max)

    # 创建掩码 [B, K_max, 1]
    indices = torch.arange(K_max, device=x.device).view(1, -1, 1)  # [1, K_max, 1]
    mask = (indices < K.view(B, 1, 1)).float()  # [B, K_max, 1]

    return mask


def AWGN_channel_VLC(x, mask, snr, cr, P=1):
    """
    可变长度编码的 AWGN 信道

    Args:
        x: 输入信号 [B, K_max, L]
        snr: 信噪比 (dB) [B] 或标量
        cr: 压缩率 [B] 或标量
        P: 信号功率，默认 1

    Returns:
        y: 加噪后的信号 [B, K_max, L]
    """
    B, K_max, L = x.shape
    device = x.device
    # 处理标量 snr
    if not torch.is_tensor(snr):
        snr = torch.full((B,), snr, device=device)
    elif snr.dim() == 0:
        snr = snr.view(1).expand(B)

    # SNR 转换 [B, 1, 1]
    gamma = 10 ** (snr / 10.0)  # [B]
    gamma = gamma.view(B, 1, 1)  # [B, 1, 1]

    # 生成噪声 [B, K_max, L]
    noise_std = torch.sqrt(P / gamma)  # [B, 1, 1]
    noise = noise_std * torch.randn(B, K_max, L, device=device)

    # 应用掩码：只对有效通道加噪
    noise = noise * mask  # mask 广播到 [B, K_max, L]
    # 加噪
    y = x + noise

    return y


class DynamicJSCCR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, c=128, K_max=48, num_classes=10):
        super(DynamicJSCCR, self).__init__()
        self.K_max = K_max
        self.c = c
        self.encoder = Encoder(in_channels, c, K_max)
        self.sc_mask = SC_Mask(K_max)
        self.decoder = Decoder(out_channels, c, K_max)
        self.classifier = Classifier(in_channels, num_classes)

    def forward(self, x, snr, cr):
        z = self.encoder(x, snr)

        z_masked, mask, H, W = self.sc_mask(z, cr)

        z_norm = Power_norm_VLC(z_masked, cr)

        z_noisy = AWGN_channel_VLC(z_norm, mask, snr, cr)

        # 零填充和fold
        z_in = z_noisy.view(z_noisy.shape[0], z_noisy.shape[1], H, W)
        # 6. 语义解码
        x_rec = self.decoder(z_in, snr)
        # 7. 分类
        logits = self.classifier(x_rec)

        return x_rec, logits


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DynamicJSCCR().to(device)
    x = torch.randn(2, 3, 64, 64).to(device)
    snr = torch.tensor([20.0, 30.0]).to(device)  # 示例 SNR
    cr = torch.tensor([0.2, 0.3]).to(device)  # 示例 CR
    x_rec, logits = model(x, snr, cr)
