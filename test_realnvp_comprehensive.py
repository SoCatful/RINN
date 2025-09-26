import torch
import numpy as np
from realnvp import FlowBlock

def test_inverse_accuracy():
    """测试逆变换的准确性"""
    torch.manual_seed(42)
    
    # 测试不同配置
    configs = [
        {"num_features": 16, "hidden": 64, "swap": False},
        {"num_features": 32, "hidden": 128, "swap": True},
        {"num_features": 8, "hidden": 32, "swap": False},
    ]
    
    for i, config in enumerate(configs):
        print(f"\n=== 测试配置 {i+1}: {config} ===")
        
        # 创建flow
        flow = FlowBlock(**config)
        
        # 测试不同batch size
        for batch_size in [1, 4, 8]:
            # 随机输入
            x = torch.randn(batch_size, config["num_features"])
            
            # 前向
            z, log_det = flow(x)
            
            # 逆向
            x_recon = flow.inverse(z)
            
            # 计算误差
            error = torch.mean((x - x_recon) ** 2).item()
            max_error = torch.max(torch.abs(x - x_recon)).item()
            
            print(f"  Batch size {batch_size}: MSE = {error:.2e}, Max error = {max_error:.2e}")
            
            # 检查误差是否在可接受范围内
            assert error < 1e-5, f"MSE too large: {error}"
            assert max_error < 1e-4, f"Max error too large: {max_error}"

def test_log_determinant():
    """测试log行列式的性质"""
    torch.manual_seed(123)
    
    flow = FlowBlock(num_features=16, hidden=64)
    
    # 生成两个不同的输入
    x1 = torch.randn(4, 16)
    x2 = torch.randn(4, 16)
    
    # 前向变换
    z1, log_det1 = flow(x1)
    z2, log_det2 = flow(x2)
    
    print(f"\n=== Log行列式测试 ===")
    print(f"输入1的log|det| = {log_det1.mean().item():.4f}")
    print(f"输入2的log|det| = {log_det2.mean().item():.4f}")
    
    # 验证log_det的形状
    assert log_det1.shape == (4,), f"Log det shape error: {log_det1.shape}"
    assert log_det2.shape == (4,), f"Log det shape error: {log_det2.shape}"
    
    print("Log行列式测试通过！")

def test_parameter_count():
    """测试模型参数数量是否合理"""
    configs = [
        {"num_features": 16, "hidden": 64},
        {"num_features": 32, "hidden": 128},
    ]
    
    print(f"\n=== 参数数量测试 ===")
    for config in configs:
        flow = FlowBlock(**config)
        param_count = sum(p.numel() for p in flow.parameters())
        print(f"配置 {config}: 参数数量 = {param_count}")
        
        # 验证参数数量不为零
        assert param_count > 0, "模型没有参数！"

def main():
    print("开始Real NVP综合测试...")
    
    test_inverse_accuracy()
    test_log_determinant()
    test_parameter_count()
    
    print("\n✅ 所有测试通过！Real NVP逆变换功能正常。")

if __name__ == "__main__":
    main()