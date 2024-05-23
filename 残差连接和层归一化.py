import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(output_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual  # 残差连接
        out = self.relu(out)
        return out

class LayerNormalization(nn.Module):
    def __init__(self, features):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# 示例用法
input_size = 10
output_size = 10

# 创建残差连接块
residual_block = ResidualBlock(input_size, output_size)

# 创建输入张量
input_tensor = torch.randn(5, input_size)  # 假设批次大小为5

# 应用残差连接块
output_tensor = residual_block(input_tensor)

print(input_tensor)
print(output_tensor)

# 创建层归一化层
layer_norm = LayerNormalization(input_size)

# 应用层归一化层
normalized_output = layer_norm(output_tensor)

print(normalized_output)
