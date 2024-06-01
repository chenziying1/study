import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

# 加载IMDB数据集的子集
dataset = load_dataset("imdb", split={'train': 'train[:1%]', 'test': 'test[:1%]'})

# 加载BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# 冻结模型的所有层
for param in model.parameters():
    param.requires_grad = False

# 只训练分类器层
model.classifier = nn.Linear(model.config.hidden_size, 2)
model.classifier.weight.requires_grad = True
model.classifier.bias.requires_grad = True

# 数据预处理函数
def preprocess(data):
    return tokenizer(data['text'], padding=True, truncation=True, max_length=256)

# 处理训练和测试数据
train_data = dataset['train'].map(preprocess, batched=True)
test_data = dataset['test'].map(preprocess, batched=True)

# 设置格式转换函数
def format_dataset(data):
    data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return data

train_data = format_dataset(train_data)
test_data = format_dataset(test_data)

# 数据加载器
train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=1)
test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=1)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in train_loader:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=inputs, attention_mask=attention_mask).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / total
    return total_loss / len(train_loader), accuracy

# 评估函数
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=inputs, attention_mask=attention_mask).logits
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return total_loss / len(test_loader), accuracy

# 训练和评估模型
num_epochs = 3
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
