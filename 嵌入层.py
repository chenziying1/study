import torch
import torch.nn as nn

# 假设词汇表大小为10000，每个单词用一个整数来表示
vocab_size = 10000
embedding_dim = 512  # 嵌入向量的维度

# 创建嵌入层
embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# 假设有一个输入序列，每个单词用整数表示
input_sequence = torch.tensor([1, 5, 20, 3, 9])

# 将输入序列传递给嵌入层，得到嵌入后的表示
embedded_sequence = embedding_layer(input_sequence)

print("输入序列:", input_sequence)
print("嵌入层:", embedding_layer)
print("输入:", input_sequence.shape)
print("嵌入之后:", embedded_sequence.shape)
print("嵌入:", embedded_sequence)
