import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import os

# 设置环境变量来避免OpenMP错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from actnorm import ActNorm1d
from realnvp import FlowBlock
from JL import JLLayer


# ---- toy dataset: mixture of 8 Gaussians in 2D ----
def sample_mixture_gaussians(batch_size=256, dim=2, num_modes=8, radius=5.0, std=0.2):
    assert dim == 2, "This toy dataset is for 2D only."
    centers = []
    for i in range(num_modes):
        angle = 2 * math.pi * i / num_modes
        centers.append([radius * math.cos(angle), radius * math.sin(angle)])
    centers = torch.tensor(centers)

    # 随机选择中心
    indices = torch.randint(0, num_modes, (batch_size,))
    chosen = centers[indices]

    # 在中心点附近加高斯噪声
    noise = std * torch.randn(batch_size, dim)
    return chosen + noise


# ---- Flow model ----
class ToyFlow(nn.Module):
    def __init__(self, dim=2, hidden=128, num_blocks=6):
        super().__init__()
        layers = []
        layers.append(ActNorm1d(dim))
        for i in range(num_blocks):
            layers.append(FlowBlock(num_features=dim, hidden=hidden, swap=(i % 2 == 0)))
        # 先不要 JL
        # layers.append(JLLayer(dim))
        self.layers = nn.ModuleList(layers)


    def forward(self, x):
        log_det_total = torch.zeros(x.size(0), device=x.device)
        z = x
        for layer in self.layers:
            if isinstance(layer, FlowBlock):
                z, ld = layer(z)
                log_det_total = log_det_total + ld
            elif hasattr(layer, "log_det_jacobian"):
                z = layer(z)
                log_det_total = log_det_total + layer.log_det_jacobian(z)
            else:
                z = layer(z)
        return z, log_det_total

    def inverse(self, z):
        # 逆向时按相反顺序走
        x = z
        for layer in reversed(self.layers):
            if hasattr(layer, "inverse"):
                x = layer.inverse(x)
            else:
                raise RuntimeError(f"Layer {layer} does not support inverse()")
        return x


# ---- log probability under standard Gaussian prior ----
def standard_normal_logprob(z):
    return -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * z.size(1) * math.log(2 * math.pi)


# ---- training loop ----
def train_toy_flow(epochs=2000, batch_size=512, lr=1e-3, device="cpu"):
    dim = 2
    model = ToyFlow(dim=dim, hidden=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []
    for step in range(epochs):
        x = sample_mixture_gaussians(batch_size=batch_size).to(device)

        z, log_det = model(x)
        prior_logprob = standard_normal_logprob(z)
        log_likelihood = prior_logprob + log_det
        loss = -log_likelihood.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 200 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")
            # 调试：打印各个组件的log-det值
            with torch.no_grad():
                z_temp, log_det_total = model(x)
                # 重新计算各个组件的log-det
                log_det_actnorm = torch.zeros(x.size(0), device=x.device)
                log_det_flow = torch.zeros(x.size(0), device=x.device) 
                log_det_jl = torch.zeros(x.size(0), device=x.device)
                
                temp_x = x
                for layer in model.layers:
                    if isinstance(layer, ActNorm1d):
                        temp_x = layer(temp_x)
                        # ActNorm的log-det是scale的对数和
                        if hasattr(layer, 'scale'):
                            log_det_actnorm += torch.log(torch.abs(layer.scale)).sum()
                    elif isinstance(layer, FlowBlock):
                        temp_x, ld = layer(temp_x)
                        log_det_flow += ld
                    elif isinstance(layer, JLLayer):
                        temp_x = layer(temp_x)
                        log_det_jl += layer.log_det_jacobian(temp_x)
                
                print("ld_actnorm.mean:", log_det_actnorm.mean().item(), 
                      "ld_flow.mean:", log_det_flow.mean().item(), 
                      "ld_jl.mean:", log_det_jl.mean().item())


    # plot training loss
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss (NLL)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    print("正在保存训练损失图像...")
    plt.savefig("train_toy_loss.png")  # 先保存
    print("训练损失图像已保存为 train_toy_loss.png")
    plt.show()  # 再显示

    # plot data vs generated samples
    with torch.no_grad():
        x_real = sample_mixture_gaussians(batch_size=1000).to(device)
        z_real, _ = model(x_real)

        # 从标准高斯采样，再逆变换
        z_fake = torch.randn(1000, dim).to(device)
        x_fake = model.inverse(z_fake)

    print("正在保存结果对比图像...")
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(x_real[:, 0].cpu(), x_real[:, 1].cpu(), s=5, alpha=0.5)
    plt.title("Real data")
    plt.subplot(1, 2, 2)
    plt.scatter(x_fake[:, 0].cpu(), x_fake[:, 1].cpu(), s=5, alpha=0.5, color="orange")
    plt.title("Generated data")
    plt.savefig("train_toy_result.png")  # 先保存
    print("结果对比图像已保存为 train_toy_result.png")
    plt.show()  # 再显示


if __name__ == "__main__":
    train_toy_flow()
