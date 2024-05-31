import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import os
from datasets import load_dataset


# 自定义数据集类
class TextDataset(Dataset):
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
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        print(text)
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 创建一个小的数据集
texts = ["I love this movie!", "This movie was terrible.", "An excellent film with a great cast.", "Not my cup of tea."]
labels = [1, 0, 1, 0]  # 1表示正面评价，0表示负面评价

# 初始化BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 创建数据集对象
max_len = 16
dataset = TextDataset(texts, labels, tokenizer, max_len)

# 获取一个样本
sample = dataset[0]

# 打印样本内容
'''
print("Text:", sample['text'])
print("Input IDs:", sample['input_ids'])
print("Attention Mask:", sample['attention_mask'])
print("Label:", sample['label'])
'''

# 假设有训练集和测试集的文本和标签
train_texts = ["This is a good movie", "I don't like this film"]
train_labels = [1, 0]  # 标签，例如 1 表示积极，0 表示消极

test_texts = ["Amazing movie, highly recommend it", "Waste of time"]
test_labels = [1, 0]

# 实例化 tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 创建数据集实例
train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=128)
test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len=128)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# 查看数据加载器的一个批次数据
for batch in train_loader:
    print(batch)
    break

