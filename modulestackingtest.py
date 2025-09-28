# 基于指定文档的极简可逆神经网络（仅affine coupling + shuffle + 简单归一化）
# 解决损失爆炸问题：新增数据归一化（文档4/5预处理策略）+ 简化网络结构 + 数值范围控制
# 指定文档依据：
# 1. Affine coupling：Real NVP文档（DENSITY ESTIMATION USING REAL NVP.pdf）第3.2节
# 2. Shuffle：Real NVP文档第3.4节（信息混合）、R-INN文档第III节（随机排列）
# 3. 归一化：ActNorm（R-INN文档第III节）、数据预处理（Real NVP文档第4.1节）
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler  # 数据归一化（解决损失爆炸核心）


# ---------------------- 1. 数据生成与归一化（解决损失爆炸：数值范围压缩到[0,1]） ----------------------
def generate_and_normalize_data(
    batch_size=64, x_dim=3, y_dim=1, pad_to_dim=10, x_range=(0, 100)
):
    """
    步骤1：生成原始数据（x:3维→y:1维，y=x1+x2+x3）
    步骤2：Zero padding到10维（文档5第IV节：低维→高维，确保可逆性维度一致）
    步骤3：归一化到[0,1]（文档4第4.1节：预处理避免数值过大，解决损失爆炸）
    """
    # 1. 生成原始3维x（0-100正数）和1维y（x的和）
    x_raw = torch.rand(batch_size, x_dim) * (x_range[1] - x_range[0]) + x_range[0]  # [B,3]
    y_raw = torch.sum(x_raw, dim=1, keepdim=True)  # [B,1]，范围0-300

    # 2. Zero padding到10维（文档5：可逆模型输入输出维度必须一致）
    x_padded = torch.zeros(batch_size, pad_to_dim)
    x_padded[:, :x_dim] = x_raw  # 前3维有效，后7维0
    y_padded = torch.zeros(batch_size, pad_to_dim)
    y_padded[:, :y_dim] = y_raw  # 第1维有效，后9维0

    # 3. 归一化到[0,1]（解决损失爆炸：原始y范围0-300→压缩到0-1，MSE损失不会超1）
    # 适配sklearn的2D输入要求（[样本数, 特征数]）
    scaler_x = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    
    x_norm = torch.tensor(
        scaler_x.fit_transform(x_padded.numpy()), dtype=torch.float32
    )
    y_norm = torch.tensor(
        scaler_y.fit_transform(y_padded.numpy()), dtype=torch.float32
    )

    return x_norm, y_norm, scaler_x, scaler_y  # 返回归一化器，用于后续反归一化


# ---------------------- 2. 极简可逆组件（仅affine coupling + shuffle + ActNorm） ----------------------
class ActNorm(nn.Module):
    """简单归一化层（文档5第III节：ActNorm，无批统计，适合可逆模型）
    功能：x → (x - mean) / std * scale + shift（mean/std从初始批次计算，后续固定）
    优势：比BatchNorm更简单，无移动平均，不破坏可逆性（文档5推荐）
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))  # 可训练缩放
        self.shift = nn.Parameter(torch.zeros(dim))  # 可训练平移
        self.register_buffer("mean", torch.zeros(dim))  # 非训练参数：初始均值
        self.register_buffer("std", torch.ones(dim))    # 非训练参数：初始标准差
        self.initialized = False  # 标记是否已计算初始mean/std

    def forward(self, x):
        if not self.initialized and self.training:
            # 初始批次计算mean/std（仅一次，后续固定，避免破坏可逆性）
            self.mean.data = x.mean(dim=0)
            self.std.data = x.std(dim=0) + 1e-6  # 避免除0
            self.initialized = True
        # 归一化：(x - mean)/std → 缩放+平移
        return (x - self.mean) / self.std * self.scale + self.shift

    def inverse(self, y):
        # 逆归一化：y → (y - shift)/scale * std + mean（可逆操作）
        return (y - self.shift) / self.scale * self.std + self.mean


class ShuffleLayer(nn.Module):
    """Shuffle层（文档4第3.4节：信息混合，确保所有维度被affine coupling处理）
    功能：随机打乱输入维度（正向），用相同索引恢复（反向）
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.register_buffer("shuffle_idx", torch.randperm(dim))  # 固定打乱索引（确保可逆）
        # 预计算逆索引：用于反向恢复
        self.register_buffer("inv_shuffle_idx", torch.argsort(self.shuffle_idx))

    def forward(self, x):
        # 正向：按shuffle_idx打乱维度
        return x[:, self.shuffle_idx]

    def inverse(self, y):
        # 反向：按inv_shuffle_idx恢复原始维度
        return y[:, self.inv_shuffle_idx]


class MinimalAffineCoupling(nn.Module):
    """极简affine coupling层（文档4第3.2节核心结构，仅线性网络预测s/t）
    分块策略：前dim//2维为x1（不变），后dim//2维为x2（用x1预测s/t变换）
    """
    def __init__(self, dim):
        super().__init__()
        self.split_dim = dim // 2  # 分块维度（10维→前5维x1，后5维x2）
        # 极简预测网络：仅1层线性（避免复杂结构导致数值不稳定）
        self.scale_net = nn.Linear(self.split_dim, dim - self.split_dim)
        self.shift_net = nn.Linear(self.split_dim, dim - self.split_dim)
        # 限制scale范围：用tanh压缩到[-1,1]，避免exp(s)过大导致数值爆炸
        self.tanh = nn.Tanh()

    def forward(self, x):
        # 正向：x = [x1, x2] → y = [x1, x2*exp(s(x1)) + t(x1)]（文档4公式3）
        x1, x2 = x.chunk(2, dim=1)
        s = self.tanh(self.scale_net(x1))  # s∈[-1,1]，exp(s)∈[1/e, e]（数值稳定）
        t = self.shift_net(x1)
        y2 = x2 * torch.exp(s) + t
        return torch.cat([x1, y2], dim=1)

    def inverse(self, y):
        # 反向：y = [y1, y2] → x = [y1, (y2 - t(y1))*exp(-s(y1))]（文档4公式4）
        y1, y2 = y.chunk(2, dim=1)
        s = self.tanh(self.scale_net(y1))  # 复用x1=y1的s/t，无需存储（文档4高效性）
        t = self.shift_net(y1)
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([y1, x2], dim=1)


# ---------------------- 3. 主可逆模型（堆叠3个极简可逆块：ActNorm → Shuffle → AffineCoupling） ----------------------
class MinimalInvertibleModel(nn.Module):
    def __init__(self, dim=10, num_blocks=3):
        super().__init__()
        self.dim = dim
        # 堆叠3个可逆块（每个块：归一化→信息混合→耦合变换，文档1/4的多块堆叠策略）
        self.blocks = nn.ModuleList([
            nn.Sequential(
                ActNorm(dim),          # 1. 归一化（稳定数值）
                ShuffleLayer(dim),      # 2. Shuffle（混合信息）
                MinimalAffineCoupling(dim)  # 3. Affine coupling（可逆变换）
            ) for _ in range(num_blocks)
        ])
        # 为每个块注册逆函数（因Sequential无inverse，需手动实现）
        self.block_inverses = [self._block_inverse(block) for block in self.blocks]

    def _block_inverse(self, block):
        """为单个块生成逆函数：按正向逆顺序调用子层的inverse"""
        def inverse(y):
            # 正向顺序：ActNorm → Shuffle → AffineCoupling → 逆顺序：AffineCoupling→Shuffle→ActNorm
            y = block[2].inverse(y)  # AffineCoupling逆
            y = block[1].inverse(y)  # Shuffle逆
            y = block[0].inverse(y)  # ActNorm逆
            return y
        return inverse

    def forward(self, x):
        """正向传播：x(10维归一化) → y(10维归一化)（文档5“正向设计”）"""
        for block in self.blocks:
            x = block(x)
        return x

    def inverse(self, y):
        """反向传播：y(10维归一化) → x(10维归一化)（文档5“反向设计”）"""
        # 可逆模型必须按块倒序调用逆函数（文档1定理1：可逆变换顺序可逆）
        for inv_func in reversed(self.block_inverses):
            y = inv_func(y)
        return y


# ---------------------- 4. 训练逻辑（正向/反向损失1:1，带反归一化验证） ----------------------
def train_model(
    model, train_loader, val_loader, scaler_x, scaler_y, epochs=150, lr=1e-3
):
    criterion = nn.MSELoss()  # 归一化后MSE损失范围[0,1]，不会爆炸
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # 加L2正则稳定训练
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("=== 训练开始（正向/反向损失1:1，数据已归一化到[0,1]） ===")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        # 训练批次
        for x_norm_batch, y_norm_batch in train_loader:
            x_norm_batch, y_norm_batch = x_norm_batch.to(device), y_norm_batch.to(device)
            optimizer.zero_grad()

            # 1. 正向传播：x_norm → y_norm_pred（计算正向损失）
            y_norm_pred = model(x_norm_batch)
            loss_forward = criterion(y_norm_pred, y_norm_batch)

            # 2. 反向传播：y_norm_pred → x_norm_recon（计算反向损失）
            x_norm_recon = model.inverse(y_norm_pred)
            loss_backward = criterion(x_norm_recon, x_norm_batch)

            # 3. 总损失：1:1加权（文档5“双向联合训练”策略）
            total_loss = loss_forward * 1.0 + loss_backward * 1.0

            # 优化更新（梯度不会爆炸，因数值范围小）
            total_loss.backward()
            optimizer.step()
            total_train_loss += total_loss.item() * x_norm_batch.size(0)

        # 每30轮验证（反归一化到原始范围，验证实际误差）
        if (epoch + 1) % 30 == 0:
            model.eval()
            with torch.no_grad():
                x_norm_val, y_norm_val = next(iter(val_loader))
                x_norm_val, y_norm_val = x_norm_val.to(device), y_norm_val.to(device)

                # 正向验证：预测y并反归一化
                y_norm_val_pred = model(x_norm_val)
                # 反归一化：从[0,1]→原始范围（sklearn需要2D输入）
                y_val_pred = torch.tensor(
                    scaler_y.inverse_transform(y_norm_val_pred.cpu().numpy()),
                    dtype=torch.float32
                )
                y_val_true = torch.tensor(
                    scaler_y.inverse_transform(y_norm_val.cpu().numpy()),
                    dtype=torch.float32
                )
                val_forward_loss_raw = criterion(y_val_pred, y_val_true)

                # 反向验证：恢复x并反归一化
                x_norm_val_recon = model.inverse(y_norm_val_pred)
                x_val_recon = torch.tensor(
                    scaler_x.inverse_transform(x_norm_val_recon.cpu().numpy()),
                    dtype=torch.float32
                )
                x_val_true = torch.tensor(
                    scaler_x.inverse_transform(x_norm_val.cpu().numpy()),
                    dtype=torch.float32
                )
                val_backward_loss_raw = criterion(x_val_recon, x_val_true)

            print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"训练总损失（归一化）: {total_train_loss/len(train_loader.dataset):.6f} | "
                f"验证正向损失（原始）: {val_forward_loss_raw.item():.2f} | "
                f"验证反向损失（原始）: {val_backward_loss_raw.item():.2f}"
            )

    return model


# ---------------------- 5. 预测逻辑（输入1,2,3和y=10，反归一化输出原始结果） ----------------------
def predict_targets(model, scaler_x, scaler_y, pad_dim=10, x_dim=3, y_dim=1):
    """
    测试1：输入x=[1,2,3]（原始3维）→ 预测y（原始1维，理论值6）
    测试2：输入y=10（原始1维）→ 恢复x（原始3维，理论和为10）
    """
    model.eval()
    device = next(model.parameters()).device
    print("\n=== 目标预测（结果已反归一化到原始范围） ===")

    # 测试1：x=[1,2,3] → 预测y
    x_raw_test1 = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)  # 原始x（和为6）
    # Zero padding到10维
    x_padded_test1 = torch.zeros(1, pad_dim)
    x_padded_test1[:, :x_dim] = x_raw_test1
    # 归一化到[0,1]
    x_norm_test1 = torch.tensor(
        scaler_x.transform(x_padded_test1.numpy()), dtype=torch.float32
    ).to(device)
    # 正向预测+反归一化
    with torch.no_grad():
        y_norm_pred1 = model(x_norm_test1)
        y_raw_pred1 = scaler_y.inverse_transform(y_norm_pred1.cpu().numpy())[0, 0]  # 提取有效y
    print(f"输入原始x = [1, 2, 3] → 预测原始y = {y_raw_pred1:.2f}（理论值6）")

    # 测试2：y=10 → 恢复x
    y_raw_test2 = torch.tensor([[10.0]], dtype=torch.float32)  # 原始y（和为10）
    # Zero padding到10维
    y_padded_test2 = torch.zeros(1, pad_dim)
    y_padded_test2[:, :y_dim] = y_raw_test2
    # 归一化到[0,1]
    y_norm_test2 = torch.tensor(
        scaler_y.transform(y_padded_test2.numpy()), dtype=torch.float32
    ).to(device)
    # 反向恢复+反归一化
    with torch.no_grad():
        x_norm_recon2 = model.inverse(y_norm_test2)
        x_raw_recon2 = scaler_x.inverse_transform(x_norm_recon2.cpu().numpy())[0, :x_dim]  # 提取有效x
        x_recon_sum2 = x_raw_recon2.sum()
    print(f"输入原始y = 10 → 恢复原始x = {x_raw_recon2.round(2)}（和为{x_recon_sum2:.2f}，理论值10）")


# ---------------------- 6. 主函数（数据→训练→预测） ----------------------
if __name__ == "__main__":
    # 超参数（基于指定文档，确保极简+稳定）
    pad_dim = 10    # Zero padding后维度（文档5）
    x_dim = 3       # 原始x维度
    y_dim = 1       # 原始y维度
    batch_size = 32
    epochs = 150    # 足够收敛且不过拟合
    lr = 1e-3       # 小学习率避免梯度波动

    # 设备配置（自动适配CPU/GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 生成并归一化数据（解决损失爆炸的核心步骤）
    train_x_norm, train_y_norm, scaler_x, scaler_y = generate_and_normalize_data(
        batch_size=1000, pad_to_dim=pad_dim, x_range=(0, 100)
    )
    val_x_norm, val_y_norm, _, _ = generate_and_normalize_data(
        batch_size=100, pad_to_dim=pad_dim, x_range=(0, 100)
    )

    # 2. 构建数据加载器
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x_norm, train_y_norm),
        batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(val_x_norm, val_y_norm),
        batch_size=batch_size, shuffle=False
    )

    # 3. 初始化极简可逆模型（3个块）
    model = MinimalInvertibleModel(dim=pad_dim, num_blocks=3)
    print(f"模型结构: {model}")

    # 4. 训练模型
    model = train_model(
        model, train_loader, val_loader, scaler_x, scaler_y, epochs=epochs, lr=lr
    )

    # 5. 预测目标值
    predict_targets(model, scaler_x, scaler_y, pad_dim=pad_dim, x_dim=x_dim, y_dim=y_dim)