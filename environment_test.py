# 打开Python终端（在激活的r-inn-env环境中输入python）
import torch
import numpy as np
import sklearn
import matplotlib.pyplot as plt

# 验证PyTorch与CUDA
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("GPU设备名:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无GPU")

# 验证基础库版本
print("NumPy版本:", np.__version__)
print("Scikit-learn版本:", sklearn.__version__)

#PyTorch版本: 2.8.0+cu128
#CUDA是否可用: True
##GPU设备名: NVIDIA GeForce RTX 5080
#NumPy版本: 2.3.3
#Scikit-learn版本: 1.7.1