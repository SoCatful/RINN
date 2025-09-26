import torch
from realnvp import FlowBlock

def main():
    torch.manual_seed(0)
    x = torch.randn(8, 16)  # batch=8, feature=16
    flow = FlowBlock(num_features=16, hidden=64)

    # forward
    z, log_det = flow(x)
    # inverse
    x_recon = flow.inverse(z)

    print("Input shape:", x.shape)
    print("Latent shape:", z.shape)
    print("Log-det shape:", log_det.shape)
    print("Reconstruction error:", (x - x_recon).abs().max().item())

if __name__ == "__main__":
    main()
