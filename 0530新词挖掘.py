# 首先对如下语料进行分词
import jieba
import re
import math
from collections import Counter

text = "支持向量机是一类按监督学习方式对数据进行二元分类的广义线性分类器，其决策边界是对学习样本求解的最大边距超平面。\
支持向量机使用铰链损失函数计算经验风险并在求解系统中加入了正则化项以优化结构风险，是一个具有稀疏性和稳健性的分类器。\
铰链损失函数的思想就是让那些未能正确分类的和正确分类的之间的距离要足够的远。\
支持向量机可以通过核方法进行非线性分类，是常见的核学习方法之一。\
支持向量机被提出于1964年，在二十世纪90年代后得到快速发展并衍生出一系列改进和扩展算法，\
在人像识别、文本分类等模式识别问题中有得到应用。"
words = jieba.lcut(text)

def get_chinese_words(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.split()[0] for line in f.readlines()]

CH_DICT = set(get_chinese_words("chinese_words.txt"))

unigram_freq, bigram_freq = {}, {}
for i in range(len(words)-1):
    # 一阶计数，即单一单词，为未登录词且由中文构成
    if words[i] not in CH_DICT and not re.search("[^\u4e00-\u9fa5]", words[i]):
        if words[i] in unigram_freq:  # 一阶计数
            unigram_freq[words[i]] += 1
        else:
            unigram_freq[words[i]] = 1
    bigram = words[i]+words[i+1]
    # 二阶计数，即两个单词的组合形式，为未登录词且由中文构成
    if bigram not in CH_DICT and not re.search("[^\u4e00-\u9fa5]", bigram):
        if bigram in bigram_freq:
            bigram_freq[bigram] += 1
        else:
            bigram_freq[bigram] = 1

unigram_freq_sorted = sorted(
    unigram_freq.items(), key=lambda d: d[1], reverse=True)
bigram_freq_sorted = sorted(
    bigram_freq.items(), key=lambda d: d[1], reverse=True)

print("unigram:\n", unigram_freq_sorted)
print("bigram:\n", bigram_freq_sorted)

def preprocess_data(file_path):
    texts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for text in f.readlines():
            text = re.sub("[^\u4e00-\u9fa5。？．，！：]", "",
                          text.strip())  # 只保留中文以及基本的标点符号
            text_splited = re.split("[。？．，！：]", text)  # 按照基本的标点符号进行分块
            texts += text_splited
    texts = [text for text in texts if text is not ""]  # 去除空字符
    return texts

texts = preprocess_data("hongloumeng.txt")
def get_chinese_words(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.split()[0] for line in f.readlines()]
CH_DICT = set(get_chinese_words("chinese_words.txt"))
def get_candidate_wordsinfo(texts, max_word_len):
    # texts 表示输入的所有文本，max_word_len 表示最长的词长
    # 四个词典均以单词为 key，分别以词频、词频、左字集合、右字集合为 value
    words_freq, candidate_words_freq, candidate_words_left_characters, candidate_words_right_characters = {}, {}, {}, {}
    WORD_NUM = 0  # 统计所有可能的字符串频次
    for text in texts:  # 遍历每个文本
        # word_indexes 中存储了所有可能的词汇的切分下标 (i,j) ，i 表示词汇的起始下标，j 表示结束下标，注意这里有包括了所有的字
        # word_indexes 的生成需要两层循环，第一层循环，遍历所有可能的起始下标 i；第二层循环，在给定 i 的情况下，遍历所有可能的结束下标 j
        word_indexes = [(i, j) for i in range(len(text))
                        for j in range(i + 1, i + 1 + max_word_len)]
        WORD_NUM += len(word_indexes)
        for index in word_indexes:  # 遍历所有词汇的下标
            word = text[index[0]:index[1]]  # 获取单词
            # 更新所有切分出的字符串的频次信息
            if word in words_freq:
                words_freq[word] += 1
            else:
                words_freq[word] = 1
            if len(word) >= 2 and word not in CH_DICT:  # 长度大于等于 2 的词以及不是词典中的词作为候选新词
                # 更新候选新词词频
                if word in candidate_words_freq:
                    candidate_words_freq[word] += 1
                else:
                    candidate_words_freq[word] = 1
                # 更新候选新词左字集合
                if index[0] != 0:  # 当为文本中首个单词时无左字
                    if word in candidate_words_left_characters:
                        candidate_words_left_characters[word].append(
                            text[index[0]-1])
                    else:
                        candidate_words_left_characters[word] = [
                            text[index[0]-1]]
                # 更新候选新词右字集合
                if index[1] < len(text)-1:  # 当为文本中末个单词时无右字
                    if word in candidate_words_right_characters:
                        candidate_words_right_characters[word].append(
                            text[index[1]+1])
                    else:
                        candidate_words_right_characters[word] = [
                            text[index[1]+1]]
    return WORD_NUM, words_freq, candidate_words_freq, candidate_words_left_characters, candidate_words_right_characters

WORD_NUM, words_freq, candidate_words_freq, candidate_words_left_characters, candidate_words_right_characters = \
    get_candidate_wordsinfo(texts=texts, max_word_len=3)  # 字符串最长为 3
    
# 计算候选单词的 PMI 值
def compute_pmi(words_freq, candidate_words):
    words_pmi = {}
    for word in candidate_words:
        # 首先，将某个候选单词按照不同的切分位置切分成两项，比如“电影院”可切分为“电”和“影院”以及“电影”和“院”
        bi_grams = [(word[0:i], word[i:]) for i in range(1, len(word))]
        # 对所有切分情况计算 PMI 值，取最大值作为当前候选词的最终 PMI 值
        # words_freq[bi_gram[0]]，words_freq[bi_gram[1]] 分别表示一个候选儿童村新词的前后两部分的出现频次
        words_pmi[word] = max(map(lambda bi_gram: math.log(
            words_freq[word]/(words_freq[bi_gram[0]]*words_freq[bi_gram[1]]/WORD_NUM)), bi_grams))
        """
        通俗版本
        pmis = []
        for bi_gram in bigrams: # 遍历所有切分情况
            pmis.append(math.log(words_freq[word]/(words_freq[bi_gram[0]]*words_freq[bi_gram[1]]/WORD_NUM))) # 计算 pmi 值
        words_pmi[word] = max(pmis) # 取最大值
        """
    return words_pmi

words_pmi = compute_pmi(words_freq, candidate_words_freq)

def compute_entropy(candidate_words_characters):
    words_entropy = {}
    for word, characters in candidate_words_characters.items():
        character_freq = Counter(characters)  # 统计邻字的出现分布
        # 根据出现分布计算邻字熵
        words_entropy[word] = sum(map(
            lambda x: - x/len(characters) * math.log(x/len(characters)), character_freq.values()))
    return words_entropy

words_left_entropy = compute_entropy(candidate_words_left_characters)
words_right_entropy = compute_entropy(candidate_words_right_characters)

def get_newwords(candidate_words_freq, words_pmi, words_left_entropy, words_right_entropy,
                 words_freq_limit=15, pmi_limit=6, entropy_limit=1):
    # 在每一项指标中根据阈值进行筛选
    candidate_words = [
        k for k, v in candidate_words_freq.items() if v >= words_freq_limit]
    candidate_words_pmi = [k for k, v in words_pmi.items() if v >= pmi_limit]
    candidate_words_left_entropy = [
        k for k, v in words_left_entropy.items() if v >= entropy_limit]
    candidate_words_right_entropy = [
        k for k, v in words_right_entropy.items() if v >= entropy_limit]
    # 对筛选结果进行合并
    return list(set(candidate_words).intersection(candidate_words_pmi, candidate_words_left_entropy, candidate_words_right_entropy))

get_newwords(candidate_words_freq, words_pmi,
             words_left_entropy, words_right_entropy)

get_newwords(candidate_words_freq, words_pmi, words_left_entropy,
             words_right_entropy, words_freq_limit=100, pmi_limit=3, entropy_limit=3)



