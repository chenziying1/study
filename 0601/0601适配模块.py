import torch
from torch import nn
from transformers import BertModel, BertTokenizer, BertConfig

class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)
        return x

class BertWithAdapters(nn.Module):
    def __init__(self, model_name, adapter_hidden_dim):
        super(BertWithAdapters, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.adapters = nn.ModuleList([Adapter(self.bert.config.hidden_size, adapter_hidden_dim) for _ in range(self.bert.config.num_hidden_layers)])
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        for i, adapter in enumerate(self.adapters):
            hidden_states[i] = hidden_states[i] + adapter(hidden_states[i])
        
        return hidden_states[-1]

# 实例化模型
model_name = 'bert-base-uncased'
adapter_hidden_dim = 64
model = BertWithAdapters(model_name, adapter_hidden_dim)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 打印模型结构
print(model)
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset

# 创建自定义数据集
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 创建数据加载器
def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = CustomDataset(
        texts=df['text'].to_numpy(),
        labels=df['label'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)

# 假设我们有一个DataFrame数据集
import pandas as pd

data = {
    'text': ['I love programming', 'I hate bugs', 'Debugging is fun', 'I enjoy learning new things'],
    'label': [1, 0, 1, 1]
}
df = pd.DataFrame(data)

MAX_LEN = 128
BATCH_SIZE = 8

train_data_loader = create_data_loader(df, tokenizer, MAX_LEN, BATCH_SIZE)

# 训练模型
EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=2e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(EPOCHS):
    model.train()
    for batch in train_data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        loss = nn.CrossEntropyLoss()(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
