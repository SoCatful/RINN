# ActNorm 安装技术文档

## 概述
本文档描述了如何在 PyTorch 2.8.0+cu128 环境下安装和配置 ActNorm 库。

## 环境要求
- Python 3.8+
- PyTorch 2.8.0+cu128
- CUDA 12.8 支持

## 安装步骤

### 1. 环境准备
确保你的 conda 环境已激活：
```bash
# 激活 r-inn-env 环境（这里记得换成你自己的环境）
C:\Users\wcq\anaconda3\Scripts\activate.bat r-inn-env
```

### 2. 获取源码
由于官方版本对 PyTorch 版本有限制，我们需要使用修改后的版本：
```bash
# 克隆源码到临时目录
git clone https://github.com/ludvb/actnorm.git temp_actnorm
```

### 3. 修改版本限制
编辑 `temp_actnorm/pyproject.toml` 文件，将 torch 版本限制从 `^1.3` 修改为 `>=1.3,<3.0`：

```toml
[tool.poetry.dependencies]
torch = ">=1.3,<3.0"  # 修改后的版本限制
```

### 4. 本地安装
在 r-inn-env 环境中安装修改后的版本：
```bash
C:\Users\wcq\anaconda3\envs\r-inn-env\python.exe -m pip install temp_actnorm/
```

### 5. 验证安装
运行测试脚本验证安装是否成功：
```bash
C:\Users\wcq\anaconda3\envs\r-inn-env\python.exe test_actnorm_install.py
```

## 预期输出
成功安装后，你应该看到以下输出：
```
✓ PyTorch导入成功，版本: 2.8.0+cu128
✓ actnorm导入成功！
使用设备: cuda
✓ ActNorm2d实例化成功！
输入形状: torch.Size([1, 3, 32, 32])
输出形状: torch.Size([1, 3, 32, 32])
✓ ActNorm前向传播成功！

🎉 actnorm安装和测试全部通过！
```

## 使用方法
在你的代码中导入 ActNorm：
```python
import torch
from actnorm import ActNorm2d

# 创建 ActNorm 层
actnorm = ActNorm2d(num_features=3)

# 前向传播
x = torch.randn(1, 3, 32, 32)
output = actnorm(x)
```

## 常见问题

### Q1: 安装时出现版本冲突错误
**解决方案**: 确保使用的是修改后的 pyproject.toml 文件，版本限制已调整为 `>=1.3,<3.0`。

### Q2: 导入时出现 "No module named 'actnorm'"
**解决方案**: 确认使用的是 r-inn-env 环境的 Python 解释器：
```bash
C:\Users\wcq\anaconda3\envs\r-inn-env\python.exe your_script.py
```

### Q3: CUDA 设备不可用
**解决方案**: 确保 PyTorch 已正确安装 CUDA 版本：
```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
```

## 技术细节

### 版本兼容性修改
- **原始限制**: `torch = "^1.3"` (等价于 >=1.3,<2.0)
- **修改后**: `torch = ">=1.3,<3.0"` 
- **兼容性**: 支持 PyTorch 1.3 到 2.8.0+

### 安装原理
通过本地源码安装方式，绕过 PyPI 的版本限制检查，直接使用修改后的依赖配置。

