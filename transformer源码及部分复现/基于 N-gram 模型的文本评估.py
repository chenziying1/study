import torch
from collections import Counter
import random
import numpy as np

# 训练数据 (一个简单的例子)
corpus = [
    "I love natural language processing",
    "natural language processing is fun",
    "I love deep learning",
    "deep learning is a key part of natural language processing"
]

# 预处理数据
def preprocess(corpus):
    tokens = [sentence.split() for sentence in corpus]
    vocab = Counter([word for sentence in tokens for word in sentence])
    word_to_idx = {word: idx for idx, (word, _) in enumerate(vocab.items())}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return tokens, word_to_idx, idx_to_word

tokens, word_to_idx, idx_to_word = preprocess(corpus)
vocab_size = len(word_to_idx)
print(f"Vocab Size: {vocab_size}")
class NGramModel(torch.nn.Module):
    def __init__(self, vocab_size, n):
        super(NGramModel, self).__init__()
        self.n = n
        self.embedding = torch.nn.Embedding(vocab_size, 10)
        self.linear1 = torch.nn.Linear(10 * (n - 1), 128)
        self.linear2 = torch.nn.Linear(128, vocab_size)
    
    def forward(self, x):
        embeds = self.embedding(x).view(1, -1)
        out = torch.nn.functional.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = torch.nn.functional.log_softmax(out, dim=1)
        return log_probs

# 初始化模型
n = 3  # 使用三元模型
model = NGramModel(vocab_size, n)
# 准备训练数据
def make_context_target(tokens, n):
    context_target_pairs = []
    for sentence in tokens:
        for i in range(len(sentence) - n + 1):
            context = sentence[i:i+n-1]
            target = sentence[i+n-1]
            context_target_pairs.append((context, target))
    return context_target_pairs

context_target_pairs = make_context_target(tokens, n)
print(context_target_pairs)

# 转换为索引
def context_target_to_idx(context_target_pairs, word_to_idx):
    context_idx_target_idx = []
    for context, target in context_target_pairs:
        context_idx = torch.tensor([word_to_idx[word] for word in context], dtype=torch.long)
        target_idx = torch.tensor([word_to_idx[target]], dtype=torch.long)
        context_idx_target_idx.append((context_idx, target_idx))
    return context_idx_target_idx

context_idx_target_idx = context_target_to_idx(context_target_pairs, word_to_idx)

# 定义损失函数和优化器
loss_function = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(100):
    total_loss = 0
    for context, target in context_idx_target_idx:
        model.zero_grad()
        log_probs = model(context)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(context_idx_target_idx)}')
        
def evaluate_text_perplexity(model, tokens, word_to_idx, n):
    model.eval()
    context_target_pairs = make_context_target(tokens, n)
    context_idx_target_idx = context_target_to_idx(context_target_pairs, word_to_idx)
    total_loss = 0
    with torch.no_grad():
        for context, target in context_idx_target_idx:
            log_probs = model(context)
            loss = loss_function(log_probs, target)
            total_loss += loss.item()
    perplexity = np.exp(total_loss / len(context_idx_target_idx))
    return perplexity

perplexity = evaluate_text_perplexity(model, tokens, word_to_idx, n)
print(f'困惑度: {perplexity}')
