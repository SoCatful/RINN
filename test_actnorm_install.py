#!/usr/bin/env python
"""
测试actnorm安装是否成功
"""

try:
    import torch
    print(f"✓ PyTorch导入成功，版本: {torch.__version__}")
    
    from actnorm import ActNorm2d
    print("✓ actnorm导入成功！")
    
    # 创建一个简单的测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建ActNorm层
    actnorm = ActNorm2d(3).to(device)
    print("✓ ActNorm2d实例化成功！")
    
    # 创建测试数据
    x = torch.randn(1, 3, 32, 32).to(device)
    print(f"输入形状: {x.shape}")
    
    # 应用归一化
    x_normalized = actnorm(x)
    print(f"输出形状: {x_normalized.shape}")
    print("✓ ActNorm前向传播成功！")
    
    print("\n🎉 actnorm安装和测试全部通过！")
    
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保在正确的conda环境中运行")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")