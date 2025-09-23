import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import normflows as nf
# 这里需要装一下normflows，只是目前的方案，之后可能会改
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(0)

# 模型参数
dim = 2
K = 6  # flow 层数
hidden_units = 64
n_iter = 2000
batch_size = 256

# base distribution
base = nf.distributions.base.DiagGaussian(dim)

# RealNVP flows
flows = []
for i in range(K):
    param_map = nf.nets.MLP([dim // 2, hidden_units, hidden_units, dim])
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    flows.append(nf.flows.Permute(dim))

model = nf.NormalizingFlow(base, flows).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 目标分布：双高斯混合
loc = torch.tensor([[-2., 0.], [2., 0.]], device=device)       # (2,2)
scale = torch.tensor([[0.5, 0.5], [0.5, 0.5]], device=device)  # (2,2)

component_dist = torch.distributions.Independent(
    torch.distributions.Normal(loc, scale), 1
)

target = torch.distributions.MixtureSameFamily(
    torch.distributions.Categorical(torch.tensor([0.5, 0.5], device=device)),
    component_dist
)

# 训练
loss_history = []

for i in range(n_iter):
    x = target.sample((batch_size,))
    loss = model.forward_kld(x).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if (i + 1) % 200 == 0:
        print(f"Iter {i+1}, Loss: {loss.item():.4f}")

# 可视化 loss 曲线
plt.figure()
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.savefig("training_loss.png", dpi=300)
plt.show()

# 可视化样本
x_samples = model.sample(1000).detach().cpu().numpy()
x_target = target.sample((1000,)).detach().cpu().numpy()

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Samples from target")
plt.scatter(x_target[:,0], x_target[:,1], alpha=0.5)
plt.subplot(1,2,2)
plt.title("Samples from RealNVP")
plt.scatter(x_samples[:,0], x_samples[:,1], alpha=0.5)
plt.savefig("generated_samples.png", dpi=300)
plt.show()
