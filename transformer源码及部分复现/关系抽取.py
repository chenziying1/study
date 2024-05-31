# 对于 lists 中每一个子列表，第一个元素为实体1，第二个元素为实体2，第三个元素为实体1对实体2的关系，第四个元素为文本。
lists = [['杨康','杨铁心','子女','杨康，杨铁心与包惜弱之子，金国六王爷完颜洪烈的养子。'],
         ['杨康','杨铁心','子女','丘处机与杨铁心、郭啸天结识后，以勿忘“靖康之耻”替杨铁心的儿子杨康取名。'],
         ['杨铁心','包惜弱','配偶','金国六王爷完颜洪烈因为贪图杨铁心的妻子包惜弱的美色，杀害了郭靖的父亲郭啸天。'],
         ['杨铁心','包惜弱','配偶','杨康，杨铁心与包惜弱之子，金国六王爷完颜洪烈的养子。'],
         ['张翠山','殷素素','配偶','张无忌,武当七侠之一张翠山与天鹰教紫微堂主殷素素之子。'],
         ['小龙女','杨过','师傅','小龙女是杨过的师父，与杨过互生情愫，但因师生恋不容于世。'],
         ['黄药师','黄蓉','父','黄药师，黄蓉之父，对其妻冯氏（小字阿衡）一往情深。'],
         ['郭啸天','郭靖','父','郭靖之父郭啸天和其义弟杨铁心因被段天德陷害，死于临安牛家村。']]

relation2idx = {'子女':0,'配偶':1,'师傅':2,'父':3}

datas, labels, pos_list1, pos_list2 = [], [], [], []
translation = 32
for entity1, entity2, relation, text in lists:
    # 找到第一个实体出现的下标
    idx1 = text.index(entity1)
    # 找到第二个实体出现的下标
    idx2 = text.index(entity2)
    sentence, pos1, pos2 = [], [], []
    for i, w in enumerate(text):
        sentence.append(w)
        # 计算句子中每个字与实体1首字的距离
        pos1.append(i-idx1+translation)
        # 计算句子中每个字与实体2首字的距离
        pos2.append(i-idx2+translation)
    datas.append(sentence)
    labels.append(relation2idx[relation])
    pos_list1.append(pos1)
    pos_list2.append(pos2)


from collections import Counter
# 统计每个字出现的次数, sum(datas,[]) 的功能是将列表铺平
word_counts = Counter(sum(datas, []))
# 建立字典表，只记录出现次数不小于 2 的字
vocab = [w for w, f in iter(word_counts.items()) if f >= 2]

# 构建词袋模型，和上一节实验相同，将字典从 2 开始编号，把 0 和 1 空出来，0 作为填充元素，1 作为不在字典中的字的编号
word2idx = dict((w,i+2) for i,w in enumerate(vocab))

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

# 构建输入，即对于样本中每一个字，从词袋模型中找到这个字对应的 idx，出现频率过低的字，并没有出现在词袋模型中，此时将这些字的 idx 取为 1
train_x = [[word2idx.get(w, 1) for w in s] for s in datas]

max_len = 64

# 在输入的左边填充 0
train_x = pad_sequences(train_x, max_len, value=0)
## 填充位置编码
train_pos1 = pad_sequences(pos_list1, max_len, value=0)
train_pos2 = pad_sequences(pos_list2, max_len, value=0)
# one_hot 编码 label
train_y = to_categorical(labels, num_classes=len(relation2idx))

from keras.layers import Input, Embedding, concatenate, Conv1D, GlobalMaxPool1D, Dense, LSTM
from keras.models import Model

# 定义输入层
words = Input(shape=(max_len,),dtype='int32')
position1 = Input(shape=(max_len,),dtype='int32')
position2 = Input(shape=(max_len,),dtype='int32')
#  Embedding 层将输入进行编码
pos_emb1 = Embedding(output_dim=16, input_dim=256)(position1)
pos_emb2 = Embedding(output_dim=16, input_dim=256)(position2)
word_emb = Embedding(output_dim=16, input_dim=256)(words)
# 分别拼接 文本编码与位置1 和文本编码与位置2
concat1 = concatenate([word_emb, pos_emb1])
concat2 = concatenate([word_emb, pos_emb2])
# 卷积池化层
conv1 = Conv1D(filters=128, kernel_size=3)(concat1)
pool1 = GlobalMaxPool1D()(conv1)
conv2 = Conv1D(filters=128, kernel_size=3)(concat2)
pool2 = GlobalMaxPool1D()(conv2)
# 拼接，最后接全连接层，激活函数为 softmax
concat = concatenate([pool1, pool2])
out = Dense(units=len(relation2idx),activation='softmax')(concat)

model = Model(inputs=[words, position1, position2],outputs=out)
# 编译模型
model.compile(optimizer='ADAM', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 训练 50 次
model.fit([train_x, train_pos1, train_pos2], train_y, batch_size=8, epochs=50)
model.save('model.h5')

test_instance = ['张翠山','殷素素','张无忌,武当七侠之一张翠山与天鹰教紫微堂主殷素素之子。']
test_ne1, test_ne2, test_text = test_instance
# 将预测数据转换为向量
pred_x = [word2idx.get(w, 1) for w in test_text]
idx1 = test_text.index(test_ne1)
idx2 = test_text.index(test_ne2)
pos1 = [i-idx1+translation for i in range(len(test_text))]
pos2 = [i-idx2+translation for i in range(len(test_text))]
pred_x = pad_sequences([pred_x], max_len, value=0)
test_pos1 = pad_sequences([pos1], max_len, value=0)
test_pos2 = pad_sequences([pos2], max_len, value=0)
# 翻转 relation2idx 字典
idx2relation = dict(zip(relation2idx.values(),relation2idx.keys()))
# 使用模型进行预测
pred = model.predict([pred_x, test_pos1, test_pos2])
# 模型预测最大值的位置作为预测值
output_idx = np.argmax(pred)
# 找到 idx2relation 中实际的标签
output_label = idx2relation[output_idx]


