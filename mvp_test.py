import torch
from actnorm import ActNorm1d
from realnvp import FlowBlock
from JL import JLLayer   # 修正导入路径

def main():
    torch.manual_seed(0)
    batch_size, dim = 8, 16

    # 构造输入
    x = torch.randn(batch_size, dim)

    # 定义模块
    actnorm = ActNorm1d(dim)
    realnvp = FlowBlock(num_features=dim, hidden=64, swap=False)
    jl = JLLayer(dim)

    # forward
    z1 = actnorm(x)
    log_det1 = actnorm.log_det_jacobian(x)

    z2, log_det2 = realnvp(z1)
    z3 = jl(z2)
    log_det3 = jl.log_det_jacobian(z2)

    total_log_det = log_det1 + log_det2 + log_det3

    # inverse
    z2_recon = jl.inverse(z3)
    z1_recon = realnvp.inverse(z2_recon)
    x_recon = actnorm.inverse(z1_recon)

    # 检查结果
    recon_error = (x - x_recon).abs().max().item()
    print(f"Input shape: {x.shape}")
    print(f"Final latent shape: {z3.shape}")
    print(f"Total log-det shape: {total_log_det.shape}")
    print(f"Reconstruction error (max abs): {recon_error:.6f}")

    assert recon_error < 1e-5, "Reconstruction failed!"

if __name__ == "__main__":
    main()
