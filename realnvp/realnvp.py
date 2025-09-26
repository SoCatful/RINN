import torch
import torch.nn as nn
import torch.nn.functional as F
from actnorm import ActNorm1d


def make_mlp(in_dim, out_dim, hidden=128, n_layers=2):
    layers = []
    dim = in_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(dim, hidden))
        layers.append(nn.ReLU(inplace=True))
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class Permute(nn.Module):
    """
    固定置换层，用于打乱通道顺序。
    """
    def __init__(self, num_features, perm=None):
        super().__init__()
        if perm is None:
            perm = torch.randperm(num_features)
        self.register_buffer("perm", perm.clone().long())
        self.register_buffer("inv_perm", torch.argsort(self.perm))

    def forward(self, x):
        # 适配 (N, C) 或 (N, C, L) 形式
        if x.dim() == 2:
            return x[:, self.perm], torch.zeros(x.shape[0], device=x.device)
        elif x.dim() == 3:
            return x[:, self.perm, :], torch.zeros(x.shape[0], device=x.device)
        else:
            raise ValueError(f"Unsupported input dim {x.dim()}")

    def inverse(self, y):
        if y.dim() == 2:
            return y[:, self.inv_perm]
        elif y.dim() == 3:
            return y[:, self.inv_perm, :]
        else:
            raise ValueError(f"Unsupported input dim {y.dim()}")


class AffineCoupling(nn.Module):
    """
    仿射耦合层: 输入一分为二，一半条件化另一半。
    """
    def __init__(self, num_features, hidden=128, swap=False):
        super().__init__()
        assert num_features % 2 == 0, "num_features 必须是偶数"
        self.num_features = num_features
        self.split_len = num_features // 2
        self.swap = swap
        self.s_net = make_mlp(self.split_len, self.split_len, hidden)
        self.t_net = make_mlp(self.split_len, self.split_len, hidden)

    def forward(self, x):
        if x.dim() == 3:  # (N, C, L) reshape 成 (N*L, C)
            N, C, L = x.shape
            x = x.permute(0, 2, 1).reshape(N * L, C)

        if not self.swap:
            x1, x2 = x[:, :self.split_len], x[:, self.split_len:]
            s = 0.9 * torch.tanh(self.s_net(x1))
            t = self.t_net(x1)
            y1, y2 = x1, x2 * torch.exp(s) + t
            log_det = torch.sum(s, dim=1)
        else:
            x1, x2 = x[:, :self.split_len], x[:, self.split_len:]
            s = 0.9 * torch.tanh(self.s_net(x2))
            t = self.t_net(x2)
            y2, y1 = x2, x1 * torch.exp(s) + t
            log_det = torch.sum(s, dim=1)

        y = torch.cat([y1, y2], dim=1)

        if 'L' in locals():  # 如果原始是 3D，要恢复
            y = y.view(N, L, C).permute(0, 2, 1)
            log_det = log_det.view(N, L).sum(dim=1)  # 累积时间维
        return y, log_det

    def inverse(self, y):
        if y.dim() == 3:
            N, C, L = y.shape
            y = y.permute(0, 2, 1).reshape(N * L, C)

        if not self.swap:
            y1, y2 = y[:, :self.split_len], y[:, self.split_len:]
            s = 0.9 * torch.tanh(self.s_net(y1))
            t = self.t_net(y1)
            x1, x2 = y1, (y2 - t) * torch.exp(-s)
        else:
            y1, y2 = y[:, :self.split_len], y[:, self.split_len:]
            s = 0.9 * torch.tanh(self.s_net(y2))
            t = self.t_net(y2)
            x2, x1 = y2, (y1 - t) * torch.exp(-s)
        x = torch.cat([x1, x2], dim=1)

        if 'L' in locals():
            x = x.view(N, L, C).permute(0, 2, 1)
        return x


class FlowBlock(nn.Module):
    """
    一个完整的 flow: ActNorm -> AffineCoupling -> Permute
    """
    def __init__(self, num_features, hidden=128, swap=False):
        super().__init__()
        self.actnorm = ActNorm1d(num_features)
        self.coupling = AffineCoupling(num_features, hidden=hidden, swap=swap)
        self.permute = Permute(num_features)

    def forward(self, x):
        z = self.actnorm(x)  # ActNorm1d只返回变换后的值
        z, ld_c = self.coupling(z)  # AffineCoupling返回(z, log_det)
        z, ld_p = self.permute(z)   # Permute返回(z, log_det)
        # 总log_det：ActNorm(0) + Coupling(ld_c) + Permute(ld_p)
        total_ld = ld_c + ld_p
        return z, total_ld

    def inverse(self, z):
        z = self.permute.inverse(z)
        z = self.coupling.inverse(z)
        z = self.actnorm.inverse(z)
        return z
