import torch
from torch import tensor


def Power_norm(z):
    P = tensor(1.0)
    B, K, L = z.shape
    z = z.flatten(start_dim=1)
    batch_size, z_dim = z.shape
    z_power = torch.sqrt(torch.sum(z ** 2, 1))
    z_M = z_power.repeat(z_dim, 1)
    z = torch.sqrt(P * z_dim) * z / z_M.t()
    z = z.view(B, K, L)
    return z

def Power_norm_VLC(z, cr):
    P = tensor(1.0)
    B, K, L = z.shape
    z = z.flatten(start_dim=1)

    batch_size, z_dim = z.shape
    Kv = torch.ceil(z_dim * cr).int()
    z_power = torch.sqrt(torch.sum(z**2, 1))
    z_M = z_power.repeat(z_dim, 1)
    z = torch.sqrt(Kv*P).unsqueeze(1)*z/z_M.t()

    z = z.view(B, K, L)
    return z

if __name__ == '__main__':
    z = torch.randn(2, 3, 4)
    out = Power_norm(z)
    print(out.shape)
