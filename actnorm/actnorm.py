import torch


__all__ = ["ActNorm1d", "ActNorm2d", "ActNorm3d"]


class ActNorm(torch.jit.ScriptModule):
    def __init__(self, num_features: int):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(num_features))
        self.bias = torch.nn.Parameter(torch.zeros(num_features))
        self.register_buffer("_initialized", torch.tensor(False))

    def reset_(self):
        self._initialized = torch.tensor(False)
        return self

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        正向变换: y = scale * x + bias
        
        Args:
            x: 输入张量
            
        Returns:
            变换后的张量
        """
        self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if not self._initialized:
            self.scale.data = 1 / x.detach().reshape(-1, x.shape[-1]).std(
                0, unbiased=False
            )
            self.bias.data = -self.scale * x.detach().reshape(
                -1, x.shape[-1]
            ).mean(0)
            self._initialized = torch.tensor(True)
        x = self.scale * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        逆向变换: x = (y - bias) / scale
        
        Args:
            y: 输出张量（需要逆变换的张量）
            
        Returns:
            逆变换后的张量
            
        Raises:
            RuntimeError: 如果ActNorm层尚未初始化
        """
        if not self._initialized:
            raise RuntimeError("ActNorm layer must be initialized before calling inverse(). "
                             "Please run forward() with some data first.")
        
        self._check_input_dim(y)
        
        # 处理多维张量的维度转换
        if y.dim() > 2:
            y = y.transpose(1, -1)
        
        # 执行逆向仿射变换: x = (y - bias) / scale
        x = (y - self.bias) / self.scale
        
        # 恢复原始维度顺序
        if x.dim() > 2:
            x = x.transpose(1, -1)
            
        return x

    def log_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算雅可比行列式的对数
        
        对于仿射变换 y = scale * x + bias，
        雅可比矩阵是对角矩阵，对角元素为 scale
        因此 log|det(J)| = sum(log|scale|)
        
        Args:
            x: 输入张量
            
        Returns:
            雅可比行列式对数的标量张量
        """
        if not self._initialized:
            raise RuntimeError("ActNorm layer must be initialized before calling log_det_jacobian(). "
                             "Please run forward() with some data first.")
        
        self._check_input_dim(x)
        
        # 计算批次大小和空间维度
        batch_size = x.shape[0]
        
        if x.dim() == 2:
            # 1D case: (N, C)
            spatial_dims = 1
        elif x.dim() == 3:
            # 1D sequence case: (N, C, L)
            spatial_dims = x.shape[2]
        elif x.dim() == 4:
            # 2D case: (N, C, H, W)
            spatial_dims = x.shape[2] * x.shape[3]
        elif x.dim() == 5:
            # 3D case: (N, C, D, H, W)
            spatial_dims = x.shape[2] * x.shape[3] * x.shape[4]
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")
        
        # log|det(J)| = spatial_dims * sum(log|scale|)
        log_det = spatial_dims * torch.sum(torch.log(torch.abs(self.scale)))
        
        # 返回批次大小的张量，每个样本的log_det相同
        return log_det.expand(batch_size)


class ActNorm1d(ActNorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


class ActNorm2d(ActNorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class ActNorm3d(ActNorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")
