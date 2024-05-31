import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 设置随机种子以确保结果可重复性
torch.manual_seed(42)

# 定义数据转换，将图像数据转换为张量，并进行标准化
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化图像数据
])

# 下载并加载训练集和测试集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 创建数据加载器，用于批量加载数据
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层到第一个隐藏层
        self.fc2 = nn.Linear(128, 64)       # 第一个隐藏层到第二个隐藏层
        self.fc3 = nn.Linear(64, 10)        # 第二个隐藏层到输出层

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 将输入展平成一维向量
        x = torch.relu(self.fc1(x))  # 第一个隐藏层，使用ReLU激活函数
        x = torch.relu(self.fc2(x))  # 第二个隐藏层，使用ReLU激活函数
        x = self.fc3(x)  # 输出层，不使用激活函数
        return x

# 创建神经网络实例
model = SimpleNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，学习率为0.001

# 训练模型
for epoch in range(5):  # 迭代5个epoch
    running_loss = 0.0
    for inputs, labels in trainloader:
        optimizer.zero_grad()  # 清零梯度

        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

print('完成训练')

# 在测试集上评估模型
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'准确度: {100 * correct / total}%')


'''
ReLU是一种简单而有效的非线性激活函数，其定义为：
f(x)=max(0,x)
负数全部化零，正数保持不变。可能会导致一部分神经元永远无法激化，导致网络性能下降。
SGD (Stochastic Gradient Descent): 随机梯度下降是最简单的优化器之一，它在每次迭代中使用单个样本的梯度来更新参数。
θ=θ−α⋅∇L(θ)
θ 是参数，
α 是学习率
∇L(θ) 是损失函数关于参数的梯度。
Adam (Adaptive Moment Estimation): Adam是一种自适应学习率的优化算法，结合了动量和自适应学习率调整的优点。
Adagrad (Adaptive Gradient Algorithm): Adagrad是一种自适应学习率的优化算法，根据参数的历史梯度调整学习率。
RMSprop (Root Mean Square Propagation): RMSprop也是一种自适应学习率的优化算法，它类似于Adagrad，但使用了梯度的移动平均来调整学习率。
Adadelta: Adadelta是对Adagrad的改进，它解决了Adagrad学习率急剧下降的问题，并且不需要手动设置学习率。
AdamW: AdamW是Adam的一种变体，通过引入权重衰减来解决Adam的参数权重衰减问题。
SparseAdam: SparseAdam是Adam的一个变体，专门用于处理稀疏梯度。
Adamax: Adamax是Adam的另一种变体，用于处理无穷范数。
ASGD (Averaged Stochastic Gradient Descent): ASGD是随机梯度下降的一种变体，它使用历史梯度的加权平均来更新参数。
LBFGS (Limited-memory BFGS): LBFGS是一种准牛顿方法，通常用于优化非常大的参数空间。
Sigmoid函数: 将输入映射到0到1之间的连续输出，常用于输出层或二元分类问题。
Tanh函数: 将输入映射到-1到1之间的连续输出，常用于隐藏层。
Leaky ReLU: 与ReLU类似，但在负数部分引入一个小的斜率，以解决ReLU可能出现的神经元死亡问题。
ELU (Exponential Linear Unit): 在负数部分引入指数增长，可以让神经元具有一定的负数输入时的输出，有助于缓解梯度消失问题。
Softmax函数: 将向量映射到概率分布，常用于多类别分类问题的输出层。
'''

import re
from datetime import datetime

# 假设有一个简单的事件模式识别
def test():
    text = "2021年1月1日 张三 在北京进行了体育锻炼"
    pattern = re.compile(r"(\d{4}年\d{1,2}月\d{1,2}日)\s*(.*)\s*在\s*(.*)\s*进行了\s*(.*)")
    matches = pattern.findall(text)
    events = []
    for match in matches:
        date_str, subject, location, action = match
        date = datetime.strptime(date_str, "%Y年%m月%d日")
        events.append({"date": date, "subject": subject, "location": location, "action": action})
    print(events)
test()

