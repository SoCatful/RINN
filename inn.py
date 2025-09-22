import torch
import torch.nn as nn
import torch.optim as optim

# 定义简单全连接神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 5)
        self.fc4 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 数据生成
num_samples = 1000
input_data = torch.rand((num_samples, 3))  # 随机生成输入数据
output_data = input_data.sum(dim=1, keepdim=True)  # 输出为输入数据的和

# 模型训练
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(input_data)
    loss = criterion(predictions, output_data)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

print("训练完成！")

# 添加测试代码
with torch.no_grad():
    # 测试预测y值
    test_input = torch.tensor([[0.1, 0.3, 0.5]])
    predicted_y = model(test_input)
    expected_y = test_input.sum(dim=1, keepdim=True)
    error_y = torch.abs(predicted_y - expected_y) / expected_y * 100
    print(f"输入: {test_input.numpy()}, 期望y: {expected_y.numpy()}, 预测y: {predicted_y.numpy()}, 误差: {error_y.numpy()}%")

    # 测试预测x值
    test_output = torch.tensor([[1.1]])
    # 由于模型不可逆，无法直接预测x值，这里仅输出提示
    print("模型不可逆，无法直接预测x值。")