import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations

class JLLayer(nn.Module):
    def __init__(self, dim: int, orthogonal_init: bool = True):
        super().__init__()
        self.dim = dim
        
        # 使用Linear层并进行初始化
        self.linear = nn.Linear(dim, dim, bias=True)
        
        # 正交初始化
        if orthogonal_init:
            nn.init.orthogonal_(self.linear.weight)
        else:
            nn.init.xavier_uniform_(self.linear.weight)
        
        # 谱归一化应用于Linear层
        self.linear = parametrizations.weight_norm(self.linear, name='weight', dim=0)
        
        # 偏置初始化为0
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：x ∈ [B, dim] → x' = xW + b"""
        return self.linear(x)

    def inverse(self, x_prime: torch.Tensor) -> torch.Tensor:
        """反向传播：x' ∈ [B, dim] → x = (x' - b)W^T"""
        x_prime_minus_b = x_prime - self.linear.bias
        return F.linear(x_prime_minus_b, self.linear.weight.T, bias=None)

    def log_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """雅可比行列式对数：对于正交矩阵，det(W)=±1，因此log|det(W)|=0"""
        return torch.zeros(x.shape[0], device=x.device)