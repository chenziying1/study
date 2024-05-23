import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 示例用法
d_model = 512  # 嵌入向量的维度
max_len = 1000  # 输入序列的最大长度
positional_encoder = PositionalEncoding(d_model, max_len)

# 创建一个输入序列（假设序列长度为10，嵌入维度为512）
input_sequence = torch.randn(10, d_model)

# 对输入序列应用位置编码
output_sequence = positional_encoder(input_sequence)

print("输入:", input_sequence.shape)
print("输出:", output_sequence.shape)
