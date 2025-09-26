import torch
import torch.nn as nn
import torch.optim as optim
from JL.layer import JLLayer

def test_dimensions():
    """测试输入输出维度是否匹配"""
    dim = 10
    batch_size = 32
    jl_layer = JLLayer(dim=dim)
    x = torch.randn(batch_size, dim)
    
    # 测试前向传播
    y = jl_layer(x)
    assert y.shape == (batch_size, dim), f"前向传播维度错误：期望 {(batch_size, dim)}，得到 {y.shape}"
    
    # 测试反向传播
    x_recovered = jl_layer.inverse(y)
    assert x_recovered.shape == (batch_size, dim), f"反向传播维度错误：期望 {(batch_size, dim)}，得到 {x_recovered.shape}"
    print("✓ 维度测试通过")

def test_invertibility():
    """测试可逆性：x → y → x_recovered 应该近似等于原始输入"""
    dim = 10
    batch_size = 32
    jl_layer = JLLayer(dim=dim)
    x = torch.randn(batch_size, dim)
    
    # 前向传播后再反向传播
    y = jl_layer(x)
    x_recovered = jl_layer.inverse(y)
    
    # 计算重构误差
    error = torch.norm(x - x_recovered) / torch.norm(x)
    assert error < 1e-5, f"可逆性测试失败：相对误差 {error:.6f}"
    print(f"✓ 可逆性测试通过（相对误差：{error:.6f}）")

def test_training():
    """测试训练过程"""
    dim = 10
    batch_size = 32
    epochs = 100
    
    # 创建模型和优化器
    jl_layer = JLLayer(dim=dim)
    optimizer = optim.Adam(jl_layer.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 生成训练数据
    x = torch.randn(batch_size, dim)
    target = torch.sin(x)  # 一个非线性变换作为目标
    
    # 训练循环
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向传播
        y = jl_layer(x)
        loss = criterion(y, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
    
    # 验证训练效果
    jl_layer.eval()
    with torch.no_grad():
        y = jl_layer(x)
        final_loss = criterion(y, target)
        print(f"✓ 训练完成（最终损失：{final_loss:.6f}）")

def main():
    print("=== 开始测试JLLayer ===\n")
    
    # 运行所有测试
    test_dimensions()
    print()
    test_invertibility()
    print()
    test_training()
    
    print("\n=== 所有测试通过 ===")

if __name__ == "__main__":
    main()