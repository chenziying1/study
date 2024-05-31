import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        # 投影到查询、键和值空间的线性变换
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        
        # 最后的输出投影
        self.output_projection = nn.Linear(d_model, d_model)
        
    def split_heads(self, tensor, batch_size):
        tensor = tensor.view(batch_size, -1, self.num_heads, self.head_dim)
        return tensor.permute(0, 2, 1, 3)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 通过线性变换将输入投影到查询、键和值空间
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        
        # 将投影后的输入序列分割成多个头,可以简单的理解为分割成多个部分方便并行处理
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        
        # 应用可选的遮挡（mask）将未来位置的权重设置为负无穷大，这样 softmax 函数会使这些位置的权重趋于零。
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # 使用 softmax 函数获得归一化的注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 对值进行加权求和
        attention_output = torch.matmul(attention_weights, value)
        
        # 将多个头的输出连接起来
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)
        
        # 最后的输出投影
        output = self.output_projection(attention_output)
        
        return output

# 示例用法
d_model = 512  # 输入和输出的维度
num_heads = 8  # 多头注意力的头数
seq_length = 10  # 输入序列的长度

# 创建一个多头自注意力层
multihead_attention = MultiHeadSelfAttention(d_model, num_heads)

# 创建输入序列（假设输入序列长度为10，维度为512）
query = torch.randn(seq_length, d_model)
key = torch.randn(seq_length, d_model)
value = torch.randn(seq_length, d_model)

# 将输入序列传递给多头自注意力层
output = multihead_attention(query, key, value)

print( query.shape)
print(output.shape)
