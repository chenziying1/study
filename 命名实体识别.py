'''
1.命名实体识别任务的定义与理论解决方法
2.命名实体识别任务中的数据集处理方法
3.命名实体识别模型——BiLstm-CRF 的理论知识
4.用 Keras 构建 BiLstm-CRF 模型
5.BiLstm-CRF 的训练与预测方法
'''

'''
命名实体识别是自然语言处理中的一项基础任务，命名实体指文本中具有特定意义的实体，通常包括人名，地名，专有名词等。
如在文本
“张无忌，金庸武侠小说《倚天屠龙记》人物角色，中土明教第三十四代教主。
武当七侠之一张翠山与天鹰教紫微堂主殷素素之子，明教四大护教法王之一金毛狮王谢逊义子。” 中，
人名实体有：张无忌，张翠山，殷素素，谢逊
书名实体有：倚天屠龙记
门派实体有：明教，武当，天鹰教
因此，命名实体识别任务通常包括两部分
实体的边界识别：如正确识别“张翠山”，而不是“张翠”或“张翠山与”等
确定实体的类型：如张无忌为人名实体，而不是门派实体或者书名实体
'''

'''
BMEO 标注方法
B 实体词首
M 实体词中
E 实体词尾
O 非实体
结合实体类型，
“武当七侠之一张翠山与天鹰教紫微堂主殷素素之子” 
这份文本就会被标注为
“武/门派_B 当/门派_E 七/O 侠/O 之/O 一/O 
张/人名_B 翠/人名_M 山/人名_E 与/O 
天/门派_B 鹰/门派_M 教/门派_E 紫/O 
微/O 堂/O 主/O 殷/人名_B 素/人名_M 素/人名_E 之/O 子/O”
'''

'''
命名实体识别任务本质上是一个序列标注问题，即给序列中的每一帧进行分类，在这里每一帧代表一个字。
如”武当七侠之一张翠山“，不考虑实体类型，则共有四个标签 BMEO。
既然是分类问题，就很自然地想到逐帧分类：即训练一个判别器，输入一个字，输出该字的类别
但是实际上，并不是说“张”这个字一定代表实体词首，有可能是“张开”这个词的起始，但“张开”并非实体。
因此，每一帧都是上下文关联的，如“张”后面跟着“翠山”，那么“张”就是实体词首，反之则不一定

例子：
你在听一个故事：“今天 天气 真 好。”
我们要为每个字做标注。按照我们的标签规则，可能的标注序列有：
"今天"：开始词（今） + 结尾词（天）
"天气"：开始词（天） + 结尾词（气）
"真"：单独词（真）
"好"：单独词（好）
在这个例子中，“今天”这个词的两个字是有关系的，“今”是开始词，而“天”是结尾词。你不可能把“天”标记为开始词，因为这与上下文不符。
关联和路径：
当你标注的时候，每个字的标签不仅要考虑这个字本身，还要考虑前后的字。例如：
如果一个字被标记为开始词，下一个字应该是中间词或结尾词。
如果一个字被标记为中间词，前一个字应该是开始词或中间词，后一个字应该是中间词或结尾词。
如果一个字被标记为结尾词，前一个字应该是开始词或中间词。
单独词没有前后关联，因为它独立存在。
这就形成了一个路径问题：你要找出一种合理的标注路径，这条路径从开始词到结尾词或单独词，符合所有的规则。
对于每一个字（帧），你有4种可能的标注（k种可能性），如果整个句子有n个字（n帧），那么理论上有k^n种不同的标注方式。

条件随机场
条件随机场（conditional random field，简称 CRF），是是一种鉴别式机率模型。在序列标注问题中，我们要计算的是条件概率 
结合上一小节序列标注的内容，可以理解为给予图中的边以权重，并找到权重最高的一条路径作为输出。
相当于选择

长短期记忆网络（Long Short-Term Memory，简称 LSTM），是循环神经网络（Recurrent Neural Network，简称 RNN）的一种，
BiLSTM 是由前向 LSTM 与后向 LSTM 组合而成，由于其设计的特点，在自然语言处理任务中都常被用来建模上下文信息。
与 CRF 不同的是，BiLSTM 依靠神经网络超强的非线性拟合能力，在训练时将数据变换到高维度的非线性空间中去，从而学习出一个模型。
虽然 BiLSTM 的精度非常的高，但是在预测时，会出现一些明显的错误，如实体词尾后一帧依然预测为实体词尾等，
而在 CRF 中，因为特征函数的存在，限定了标签之间的关系。
因此，将 CRF 接到 BiLSTM 上，就可以将两者的特点结合，取长补短，通过 BiLSTM 提取高效的特征，让 CRF 的学习更加有效。
'''
raw_text = '''张无忌，金庸武侠小说《倚天屠龙记》人物角色，中土明教第三十四代教主。武当七侠之一张翠山与天鹰教紫微堂主殷素素之子，明教四大护教法王之一金毛狮王谢逊义子。
              张翠山，《倚天屠龙记》第一卷的男主角，在武当七侠之中排行第五，人称张五侠。与天鹰教殷素素结为夫妇，生下张无忌，后流落到北极冰海上的冰火岛，与谢逊相识并结为兄弟。
              殷素素，金庸武侠小说《倚天屠龙记》第一卷的女主人公。天鹰教紫薇堂堂主，容貌娇艳无伦，智计百出，亦正亦邪。与武当五侠张翠山同赴王盘山，结果被金毛狮王谢逊强行带走，三人辗转抵达冰火岛。殷素素与张翠山在岛上结为夫妇，并诞下一子张无忌。
              谢逊，是金庸武侠小说《倚天屠龙记》中的人物，字退思，在明教四大护教法王中排行第三，因其满头金发，故绰号“金毛狮王”。
           '''
annotations = {'name':['张无忌','张翠山','殷素素','谢逊'], 'book':['倚天屠龙记'],'org':['明教','武当','天鹰教']}
raw_text, annotations

import re

# 先去掉原始文本中的换行和空格符
raw_text = raw_text.replace('\n', '').replace(' ', '')
# 初始化 label：将其全部初始化为 O
labels = len(raw_text)*['O']

# 通过 key-value 的方式遍历 annotations 字典，进行转换
for ann, entities in annotations.items():
    for entity in entities:
        # 先生成实体对应的 BME 标注类型
        B, M, E = [['{}_{}'.format(ann,i)] for i in ['B','M','E']]
        '''
            print(B, M, E)
            ['name_B'] ['name_M'] ['name_E']，['name_B'] ['name_M'] ['name_E']，['name_B'] ['name_M'] ['name_E']，['name_B'] ['name_M'] ['name_E']
            ['book_B'] ['book_M'] ['book_E']
            ['org_B'] ['org_M'] ['org_E']，['org_B'] ['org_M'] ['org_E']，['org_B'] ['org_M'] ['org_E']
        '''
        # 计算实体词中的数量
        M_len = len(entity) - 2
        # 生成 label，如果词中数为0，则直接为 BE，不然按数量添加 M
        #-2 是因为实体词中包含了开始和结束
        label = B + M * M_len + E if M_len else B + E
        '''
        生成与实体对应的 label数量
        print(label)
        ['name_B', 'name_M', 'name_E']，['name_B', 'name_M', 'name_E']，['name_B', 'name_M', 'name_E']，['name_B', 'name_E']
        ['book_B', 'book_M', 'book_M', 'book_M', 'book_E']
        ['org_B', 'org_E']，['org_B', 'org_E']，['org_B', 'org_M', 'org_E']
        '''
        # 从原始文本中找到实体对应出现的所有位置
        idxs = [r.start() for r in re.finditer(entity, raw_text)]
        #print(idxs)
        for idx in idxs:
        # 替换原 label 中的 O 为实际 label
            labels[idx:idx+len(entity)] = label

'''
# 打印原始文本和对应转换后的 label
for ann,label in zip(raw_text,labels):
    print(ann, label)
'''

'''
在自然语言处理中，需要将文本数据特征提取为向量数据，才能被模型使用。
在本实验中使用的是词袋模型，忽略文本的语法和语序要素，将其仅仅看做是若干个词汇的集合。
'''
from collections import Counter
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences

# 原始文本和标签
raw_text = '''张无忌，金庸武侠小说《倚天屠龙记》人物角色，中土明教第三十四代教主。武当七侠之一张翠山与天鹰教紫微堂主殷素素之子，明教四大护教法王之一金毛狮王谢逊义子。'''
annotations = {'name':['张无忌','张翠山','殷素素','谢逊'], 'book':['倚天屠龙记'],'org':['明教','武当','天鹰教']}

# 先去掉原始文本中的换行和空格符
raw_text = raw_text.replace('\n', '').replace(' ', '')
# 初始化 label：将其全部初始化为 O
labels = len(raw_text)*['O']

# 通过 key-value 的方式遍历 annotations 字典，进行转换
for ann, entities in annotations.items():
    for entity in entities:
        # 先生成实体对应的 BME 标注类型
        B, M, E = [['{}_{}'.format(ann,i)] for i in ['B','M','E']]
        # 计算实体词中的数量
        M_len = len(entity) - 2
        # 生成 label，如果词中数为0，则直接为 BE，不然按数量添加 M
        label = B + M * M_len + E if M_len else B + E
        # 从原始文本中找到实体对应出现的所有位置
        idxs = [r.start() for r in re.finditer(entity, raw_text)]

        for idx in idxs:
        # 替换原 label 中的 O 为实际 label
            labels[idx:idx+len(entity)] = label

# 统计每个字出现的次数
word_counts = Counter(raw_text)
# 建立字典表，只记录出现次数不小于 2 的字
vocab = [w for w, f in word_counts.items() if f >= 2]
label_set = list(set(labels))

# 拆分训练集，每一句话作为一个样本，先找到每个句号的位置
sentence_len = [r.start()+1 for r in re.finditer('。', raw_text)]

# 进行拆分，这里要注意最后一个句号后面不需要拆分，所以最后一个位置不需要取到
split_text = np.split(list(raw_text), sentence_len[:-1])
split_label = np.split(labels, sentence_len[:-1])

# 构建词袋模型，这里要将字典从 2 开始编号，把 0 和 1 空出来，0 作为填充元素，1 作为不在字典中的字的编号
word2idx = {w: i+2 for i, w in enumerate(vocab)}
label2idx = [[label_set.index(w) for w in s] for s in split_label]

# 构建输入，即对于样本中每一个字，从词袋模型中找到这个字对应的 idx，出现频率过低的字，并没有出现在词袋模型中，此时将这些字的 idx 取为 1
train_x = [[word2idx.get(w, 1) for w in s] for s in split_text]

max_len = 64

# 在输入的左边填充 0，在输出的左端填充 -1
train_x = pad_sequences(train_x, max_len, value=0)
train_y = pad_sequences(label2idx, max_len, value=-1)
train_y = np.expand_dims(train_y, 2)

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss

# 定义模型的超参
EMBED_DIM = 200
BiRNN_UNITS = 200

# 初始化模型
model = Sequential()
# 添加 Embedding 层，将输入转换成向量
model.add(Embedding(len(vocab)+2, EMBED_DIM, mask_zero=True))
# 添加 BiLstm 层
model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
# 初始化 crf
crf = CRF(len(train_y), sparse_target=True)
# 将 crf 添加到模型中
model.add(crf)
model.summary()
# 编译模型
model.compile('adam', loss=crf_loss, metrics=[crf.accuracy])

model.fit(train_x, train_y, batch_size=9, epochs=120)
model.save('model.h5')

text = '谢逊，是金庸武侠小说《倚天屠龙记》中的人物，字退思，在明教四大护教法王中排行第三，因其满头金发，故绰号“金毛狮王"。'

# 将预测数据转换为特征向量
pred_x = [word2idx.get(w, 1) for w in text]
pred_x = pad_sequences([pred_x], max_len)

# 使用模型进行预测
pred = model.predict(pred_x)

# 去除多余的维度
pred = np.squeeze(pred)[-len(text):]

# 把输出向量转换为 label 对应的 idx
result = [np.argmax(r) for r in pred]

# 打印输出结果
reslut_labels = [label_set[i] for i in result]
for w, l in zip(text, reslut_labels):
    print(w, l)