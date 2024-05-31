import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.0):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()

    # 分别将输入特征映射到隐藏层，再将隐藏层映射到输出层。
    # 我们还定义了一个 dropout 层和一个 ReLU 激活函数，
    # 它们用于在模型训练过程中引入非线性性和防止过拟合。
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x) #dropout 可以看作是一种网络层，当应用于输入或隐藏层时，它会在前向传播期间将一些神经元的输出置零。在测试阶段，dropout 不会对神经元的输出进行修改，而是将其按比例缩放以保持期望的输出。这样可以确保测试阶段的预测结果稳定。
        x = self.fc2(x)
        return x

# 示例用法
input_size = 10  # 输入特征的维度
hidden_size = 20  # 隐藏层神经元的数量
output_size = 1  # 输出的维度（例如，二分类任务）
dropout_prob = 0.1  # dropout 层的丢弃概率

# 创建前馈神经网络
ffnn = FeedForwardNN(input_size, hidden_size, output_size, dropout_prob)

# 创建输入张量（假设输入特征的维度为10）
input_tensor = torch.randn(5, input_size)  # 假设批次大小为5

# 将输入张量传递给前馈神经网络
output_tensor = ffnn(input_tensor)

print(input_tensor.shape)
print(output_tensor.shape)
