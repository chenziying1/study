text = "i like dog i like cat i like animal dog cat animal apple cat dog like dog fish milk like dog \
cat eyes like i like apple apple i hate apple i movie book music like cat dog hate cat dog like"

# 基本的参数设置
EMBEDDING_DIM = 2  # 词向量维度
PRINT_EVERY = 1000  # 每训练多少步可视化一次
EPOCHS = 100000  # 训练的轮数
BATCH_SIZE = 5  # 每一批训练数据中输入词的个数
N_SAMPLES = 3  # 负样本大小
WINDOW_SIZE = 5  # 周边词窗口大小
FREQ = 0  # 去除低频词的阈值
DELETE_WORDS = False  # 是否进行高频词抽样处理

from collections import Counter


# 文本预处理
def preprocess(text, FREQ):
    text = text.lower()  # 转小写
    words = text.split()  # 分词
    # 去除低频词
    word_counts = Counter(words)  # 计算词频
    trimmed_words = [word for word in words if word_counts[word] > FREQ]
    return trimmed_words

words = preprocess(text, FREQ)

vocab2id, id2vocab = {}, {}
for w in words:
    if w not in vocab2id:
        vocab2id[w] = len(vocab2id)
        id2vocab[len(id2vocab)] = w
# 将文本转化为数值
id_words = [vocab2id[w] for w in words]

import numpy as np
import torch

int_word_counts = Counter(id_words)
total_count = len(id_words)  # 总频次
word_freqs = {w: c/total_count for w,
              c in int_word_counts.items()}  # 计算每个单词的出现频率

# 负样本抽样中，计算某个单词被选中的概率分布
unigram_dist = np.array(list(word_freqs.values()))
noise_dist = torch.from_numpy(
    unigram_dist ** (0.75) / np.sum(unigram_dist ** (0.75)))

import random

if DELETE_WORDS:
    t = 1e-5
    prob_drop = {w: 1 - np.sqrt(t/word_freqs[w])
                 for w in int_word_counts}  # 每个单词被丢弃的概率，与词频相关
    train_words = [w for w in id_words if random.random() >
                   prob_drop[w]]  # 根据丢弃概率保留训练单词
else:
    train_words = id_words
    
def get_target(words, idx, window_size):
    # idx 表示输入词在单词序列中的下标
    # 确定目标词的起止范围
    start_point = idx - window_size if (idx - window_size) > 0 else 0
    end_point = idx + window_size
    targets = words[start_point:idx] + words[idx + 1:end_point + 1]  # 获取目标词
    return targets

def get_batch(words, batch_size, window_size):
    n_batches = len(words)//batch_size  # 有多少个批次
    words = words[:n_batches * batch_size]  # 取可以整除批次的训练数据
    # 隔一个 BATCH_SIZE 大小遍历数据
    # 注意，这里的 BATCH_SIZE 并非训练过程中一批数据的大小
    # 而是一批训练数据中输入词汇的多少，因此真正的训练数据是 BATCH_SIZE * 2 * WINDOW_SIZE
    for idx in range(0, len(words), batch_size):
        batch_x, batch_y = [], []  # 输入词及目标词
        batch = words[idx:idx + batch_size]  # 一个 batch 中所取的输入词
        for j in range(batch_size):
            x = batch[j]  # 获取输入词
            y = get_target(batch, j, window_size)  # 获取目标词
            # 一个输入词对应多个目标词，因此复制输入词个数，使两者一一对应，再放入批数据中
            batch_x.extend([x] * len(y))
            batch_y.extend(y)  # 将目标词放入批数据中
        yield batch_x, batch_y

from torch import nn, optim


# Skip-gram 模型的构建
class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist):
        super(SkipGramNeg, self).__init__()
        self.n_vocab = n_vocab  # 词典大小
        self.n_embed = n_embed  # 词向量维度
        self.noise_dist = noise_dist  # 负样本抽样中，计算某个单词被选中的概率分布
        # 定义词向量层，包括输入词及目标词分别对应的词向量层，参数矩阵大小一样
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)
        # 词向量层参数初始化
        #self.in_embed.weight.data.uniform_(-1, 1)
        #self.out_embed.weight.data.uniform_(-1, 1)
    # 输入词的前向过程，即获取输入词的词向量

    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors
    # 目标词的前向过程，即获取目标词的词向量

    def forward_output(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors
    # 负样本词的前向过程，即获取噪声词（负样本）的词向量

    def forward_noise(self, batch_size, n_samples):
        # 从词汇分布中采样负样本
        noise_words = torch.multinomial(self.noise_dist,
                                        batch_size * n_samples,
                                        replacement=True)
        noise_vectors = self.out_embed(noise_words).view(
            batch_size, N_SAMPLES, self.n_embed)
        return noise_vectors

class NegativeSamplingLoss(nn.Module):
    def __init__(self, n_embed):
        super(NegativeSamplingLoss, self).__init__()
        self.n_embed = n_embed

    def forward(self, input_vectors, output_vectors, noise_vectors):
        batch_size, _ = input_vectors.shape
        # 将输入词向量与目标词向量作维度转化处理
        input_vectors = input_vectors.view(batch_size, self.n_embed, 1)
        output_vectors = output_vectors.view(batch_size, 1, self.n_embed)
        # 目标词损失
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()
        # 负样本损失
        noise_loss = torch.bmm(noise_vectors.neg(),
                               input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)
        # 综合计算两类损失
        return - (out_loss + noise_loss).mean()

model = SkipGramNeg(n_vocab=len(vocab2id),
                    n_embed=EMBEDDING_DIM, noise_dist=noise_dist)
criterion = NegativeSamplingLoss(n_embed=EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

steps = 0
for e in range(EPOCHS):
    # 获取输入词以及目标词
    for input_words, target_words in get_batch(train_words, BATCH_SIZE, WINDOW_SIZE):
        steps += 1
        inputs, targets = torch.LongTensor(
            input_words), torch.LongTensor(target_words)  # 转化为 tensor 格式
        # 输入词、目标词以及噪声词向量
        input_vectors = model.forward_input(inputs)
        output_vectors = model.forward_output(targets)
        batch_size, _ = input_vectors.shape
        noise_vectors = model.forward_noise(batch_size, N_SAMPLES)
        # 计算损失
        loss = criterion(input_vectors, output_vectors, noise_vectors)
        # 打印损失
        if steps % PRINT_EVERY == 0:
            print("loss：", loss)
        # 梯度回传
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
