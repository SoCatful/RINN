import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from actnorm.actnorm import ActNorm1d
from realnvp.realnvp import FlowBlock
from JL.layer import JLLayer
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. 离散档位定义
X_choices = np.array([100, 150, 200])  # 电阻档位
Y_choices = np.array([[-10, -12, -15], [-3, -4, -5]])  # S参数档位（S11, S21）

# 2. 数据生成函数
def sample_XY(n_samples):
    X = np.random.choice(X_choices, size=(n_samples, 3))
    # Y映射规则：简单线性映射+离散噪声
    Y = np.zeros((n_samples, 2))
    for i in range(n_samples):
        # 例如：X=[100,150,200] → Y=[-10,-3]，可自定义映射
        Y[i, 0] = -10 + (X[i, 0] - 100) // 50 * (-2)  # S11
        Y[i, 1] = -3 + (X[i, 1] - 100) // 50 * (-1)   # S21
        # 加入±1dB离散噪声
        Y[i, 0] += np.random.choice([-1, 0, 1])
        Y[i, 1] += np.random.choice([-1, 0, 1])
        # 四舍五入到离散档位
        Y[i, 0] = Y_choices[0][np.argmin(np.abs(Y_choices[0] - Y[i, 0]))]
        Y[i, 1] = Y_choices[1][np.argmin(np.abs(Y_choices[1] - Y[i, 1]))]
    return X, Y

# 3. 训练/测试数据
X_train, Y_train = sample_XY(100)
X_test, Y_test = sample_XY(20)

# 4. 零填充X到2维（与Y/z一致）
def pad_X(X):
    # 只取前2维，或均值填充
    if X.shape[1] > 2:
        return X[:, :2]
    else:
        return X
X_train_pad = pad_X(X_train)
X_test_pad = pad_X(X_test)

# 5. 转为Tensor
X_train_t = torch.tensor(X_train_pad, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test_pad, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

# 6. 标准高斯z样本
def sample_z(n):
    return torch.randn(n, 2)

# 7. 模型定义
actnorm = ActNorm1d(2)
realnvp = FlowBlock(num_features=2, hidden=32, swap=False)
jl = JLLayer(dim=2)

# 8. 损失函数定义
def nmse(pred, true):
    return ((pred - true) ** 2 / (true ** 2 + 1e-8)).mean()

def mmd(x, y):
    # 简单MMD实现（高斯核）
    xx = x @ x.t()
    yy = y @ y.t()
    xy = x @ y.t()
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    dxx = rx.t() + rx - 2 * xx
    dyy = ry.t() + ry - 2 * yy
    dxy = rx.t() + ry - 2 * xy
    kxx = torch.exp(-dxx / 2)
    kyy = torch.exp(-dyy / 2)
    kxy = torch.exp(-dxy / 2)
    return kxx.mean() + kyy.mean() - 2 * kxy.mean()

# 9. 优化器
params = list(actnorm.parameters()) + list(realnvp.parameters()) + list(jl.parameters())
optimizer = optim.Adam(params, lr=0.01)

# 10. 训练循环
loss_curve = []
for epoch in range(101):
    optimizer.zero_grad()
    # 正向传播
    x = X_train_t
    y_true = Y_train_t
    z_true = sample_z(x.shape[0])
    x_norm = actnorm(x)
    y_pred, _ = realnvp(x_norm)
    z_pred = jl(y_pred)
    # 逆向传播（用于Lx）
    y_inv = y_true
    z_inv = sample_z(y_inv.shape[0])
    yz_inv = jl.inverse(z_inv)
    x_inv = realnvp.inverse(yz_inv)
    x_inv = actnorm.inverse(x_inv)
    # 损失
    Ly = nmse(y_pred, y_true)
    Lz = mmd(z_pred, z_true)
    Lx = mmd(x_inv, x)
    L_total = 0.3 * Lx + 0.5 * Ly + 0.2 * Lz
    L_total.backward()
    optimizer.step()
    loss_curve.append(L_total.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, L_total={L_total.item():.4f}, Ly={Ly.item():.4f}, Lx={Lx.item():.4f}, Lz={Lz.item():.4f}")

# 11. 正向推理
print("\n--- 正向推理 ---")
Y_pred_list = []
for i in range(X_test_t.shape[0]):
    x = X_test_t[i:i+1]
    y_samples = []
    for _ in range(10):
        x_norm = actnorm(x)
        y_pred, _ = realnvp(x_norm)
        y_pred = jl(y_pred)
        y_samples.append(y_pred.detach().cpu().numpy())
    y_samples = np.array(y_samples).reshape(10, 2)
    y_mean = y_samples.mean(axis=0)
    y_std = y_samples.std(axis=0)
    # 四舍五入到离散档位
    y_mean_round = [Y_choices[0][np.argmin(np.abs(Y_choices[0] - y_mean[0]))], Y_choices[1][np.argmin(np.abs(Y_choices[1] - y_mean[1]))]]
    Y_pred_list.append(y_mean_round)
    print(f"X_test[{i}]: {X_test[i]}, Y_pred均值: {y_mean_round}, 95%区间: [{y_mean-2*y_std}, {y_mean+2*y_std}]")

# 12. 逆向推理
print("\n--- 逆向推理 ---")
for i in range(Y_test_t.shape[0]):
    y = Y_test_t[i:i+1]
    x_candidates = []
    nmse_list = []
    for _ in range(10):
        z_sample = sample_z(1)
        yz = jl.inverse(y)
        x_pred = realnvp.inverse(yz)
        x_pred = actnorm.inverse(x_pred)
        # 裁剪回离散档位
        x_pred_np = x_pred.detach().cpu().numpy().flatten()
        x_pred_round = [X_choices[np.argmin(np.abs(X_choices - x_pred_np[0]))], X_choices[np.argmin(np.abs(X_choices - x_pred_np[1]))], X_choices[0]]
        x_candidates.append(x_pred_round)
        nmse_val = ((np.array(x_pred_round) - X_test[i]) ** 2 / (X_test[i] ** 2 + 1e-8)).mean()
        nmse_list.append(nmse_val)
    # Top3候选
    top3_idx = np.argsort(nmse_list)[:3]
    print(f"Y_test[{i}]: {Y_test[i]}, Top3 X_pred: {[x_candidates[j] for j in top3_idx]}, NMSE: {[nmse_list[j] for j in top3_idx]}")

# 13. 损失曲线可视化
import matplotlib.pyplot as plt
plt.plot(loss_curve)
plt.xlabel('Epoch')
plt.ylabel('L_total')
plt.title('训练损失曲线')
plt.show()


# 14. 实际预测模块（新增部分）
def predict_with_error(model, X_input, Y_ground_truth=None):
    # 预处理输入
    X_padded = pad_X(np.array([X_input]))
    X_tensor = torch.tensor(X_padded, dtype=torch.float32)
    
    # 正向推理
    with torch.no_grad():
        x_norm = actnorm(X_tensor)
        y_pred, _ = realnvp(x_norm)
        z_pred = jl(y_pred)
    
    # 转换为numpy数组
    y_pred_np = y_pred.numpy().flatten()
    
    # 离散化处理
    y_pred_discrete = [
        Y_choices[0][np.argmin(np.abs(Y_choices[0] - y_pred_np[0]))],
        Y_choices[1][np.argmin(np.abs(Y_choices[1] - y_pred_np[1]))]
    ]
    
    # 理论值计算（根据用户定义的映射规则）
    theoretical_S11 = -10 + (X_input[0] - 100) // 50 * (-2)
    theoretical_S21 = -3 + (X_input[1] - 100) // 50 * (-1)
    
    # 误差计算
    s11_error = y_pred_discrete[0] - theoretical_S11
    s21_error = y_pred_discrete[1] - theoretical_S21
    
    print(f"输入X: {X_input} => 预测Y: {y_pred_discrete}")
    print(f"理论值应: S11={theoretical_S11}dB, S21={theoretical_S21}dB")
    print(f"预测误差: ΔS11={s11_error}dB, ΔS21={s21_error}dB\n")

# 15. 实际使用示例（新增部分）
print("\n=== 实际预测案例演示 ===")
# 案例1：标准输入
predict_with_error(model=None, X_input=[100, 150, 200])  # 使用训练完成的模型

# 案例2：边界值测试
predict_with_error(model=None, X_input=[200, 200, 100])

# 案例3：异常值处理
predict_with_error(model=None, X_input=[300, 50, 150])


# 16. 逆向预测模块（新增部分）
def predict_X_from_Y(model, Y_input, X_ground_truth=None):
    # 预处理输入
    Y_tensor = torch.tensor([Y_input], dtype=torch.float32)
    
    # 逆向推理
    x_candidates = []
    for _ in range(10):
        with torch.no_grad():
            z_sample = sample_z(1)
            yz = jl.inverse(Y_tensor)
            x_pred = realnvp.inverse(yz)
            x_pred = actnorm.inverse(x_pred)
            
            # 后处理
            x_pred_np = x_pred.numpy().flatten()
            # 离散化到电阻档位
            x_discrete = [X_choices[np.argmin(np.abs(X_choices - val))] for val in x_pred_np[:2]]
            x_discrete.append(X_choices[0])  # 第三维默认值
            x_candidates.append(x_discrete)
    
    # 计算理论X（根据原有映射规则逆向推导）
    theoretical_X = [
        (abs(Y_input[0]) - 10) * 50 / 2 + 100,
        (abs(Y_input[1]) - 3) * 50 / 1 + 100,
        X_choices[0]
    ]
    
    # 计算模式预测结果
    x_mode = np.round(np.mean(x_candidates, axis=0))
    
    # 误差计算
    error = [
        (x_mode[0] - theoretical_X[0]) / theoretical_X[0] * 100,
        (x_mode[1] - theoretical_X[1]) / theoretical_X[1] * 100
    ]
    
    print(f"输入Y: {Y_input} => 预测X: {x_mode}")
    print(f"理论值应: X={theoretical_X}")
    print(f"相对误差: 通道1={error[0]:.1f}%，通道2={error[1]:.1f}%\n")
    return x_mode

# 更新实际使用示例（新增部分）
print("\n=== 逆向预测案例演示 ===")
# 案例4：标准Y输入
predict_X_from_Y(model=None, Y_input=[-12, -4])

# 案例5：边界值测试
predict_X_from_Y(model=None, Y_input=[-15, -5])

# 案例6：异常值处理
predict_X_from_Y(model=None, Y_input=[-20, 0])