# 基于词频
import jieba
import collections
import jieba.analyse
from jieba import posseg
from gensim import corpora, models

text = "关键词是能够表达文档中心内容的词语，常用于计算机系统标引论文内容特征、信息检索、系统汇集以供读者检阅。关键词提取是\
文本挖掘领域的一个分支，是文本检索、文档比较、摘要生成、文档分类和聚类等文本挖掘研究的基础性工作。"
count = collections.Counter(jieba.lcut(text))  # 分词及计算词频
count = sorted(count.items(), key=lambda x: x[1], reverse=True)  # 按词频排序

with open("hit_stopwords.txt", "r", encoding="utf8") as f:    
    # 获取停用词（其中也包含了标点符号）
    stopwords = [w.strip() for w in f.readlines()]
words = [word for word in jieba.lcut(
    text) if word not in set(stopwords)]  # 分词及过滤停用词
count = collections.Counter(words)  # 计算词频
count = sorted(count.items(), key=lambda x: x[1], reverse=True)

# 基于词频,去除停用词以及标点符号，并且指定词性
with open("hit_stopwords.txt", "r", encoding="utf8") as f:  # 获取停用词（其中也包含了标点符号）
    stopwords = [w.strip() for w in f.readlines()]
v_types = {'v', 'vd', 'vn', 'vf', 'vx', 'vi',
           'vl', 'vg'}  # jieba 词性标注中所有动词相关的词性符号
# print(posseg(text))
words = [i.word for i in posseg.cut(text) if i.word not in set(
    stopwords) and i.flag in v_types]  # 分词及过滤停用词，并且指定词性
count = collections.Counter(words)  # 计算词频
count = sorted(count.items(), key=lambda x: x[1], reverse=True)  # 按词频排序


keywords = jieba.analyse.extract_tags(
    text, topK=5, withWeight=False, allowPOS=())

keywords = jieba.analyse.extract_tags(
    text, topK=5, withWeight=False, allowPOS=('ns', 'vn', 'n'))

# 首先进行文本处理
texts = ["理论上，NLP是一种很吸引人的人机交互方式。",
         "早期的语言处理系统如SHRDLU，当它们处于一个有限的“积木世界”，运用有限的词汇表会话时，工作得相当好。",
         "这使得研究员们对此系统相当乐观，然而，当把这个系统拓展到充满了现实世界的含糊与不确定性的环境中时，他们很快丧失了信心。",
         "由于理解（understanding）自然语言，需要关于外在世界的广泛知识以及运用这些知识的能力。",
         "自然语言认知，同时也被视为一个人工智能完备（AI-complete）的问题。",
         "同时，在自然语言处理中，理解的定义也变成一个主要的问题。",
         "有关理解定义问题的研究已经引发关注。"]

# 将文本进行分词
words = [jieba.lcut(text) for text in texts]
dictionary = corpora.Dictionary(words)  # 构建词典
# doc2bow()方法将 dictionary 转化为一个词袋。
corpus = [dictionary.doc2bow(w) for w in words]
# 得到的结果corpus是一个向量的列表，向量的个数就是文档数。在每个文档向量中都包含一系列元组,元组的形式是（单词 ID，词频）
lda_model = models.ldamodel.LdaModel(
    corpus=corpus, num_topics=2, id2word=dictionary, passes=10)
topic_words = lda_model.print_topics(num_topics=2, num_words=5)
print('主题及其 top5 的单词:\n', topic_words)  # * 前的数值表示单词权重
keywords = jieba.analyse.textrank(
    text, topK=5, withWeight=True, allowPOS=('ns', 'n', 'vn', 'n'))
print('关键词:\n', keywords)  # 输出关键词及其权重