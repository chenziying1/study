import re
# 英文字母，用于替换及插入操作
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
def get_similar_words(word):
    n = len(word)
    # 在各个位置删除某一字母而得的词
    s1 = [word[0:i]+word[i+1:] for i in range(n)]
    # 在各个位置相邻字母调换位置
    s2 = [word[0:i]+word[i+1]+word[i]+word[i+2:] for i in range(n-1)]
    # 在各个位置替换
    s3 = [word[0:i]+c+word[i+1:] for i in range(n) for c in ALPHABET]
    # 在各个位置插入
    s4 = [word[0:i]+c+word[i:] for i in range(n+1) for c in ALPHABET]
    similar_words = set(s1+s2+s3+s4)  # 去重
    return similar_words

def get_words(text):
    return re.findall("[a-z]+", text.lower())

def get_unigram(words):
    unigram = {}
    for w in words:
        if w in unigram:
            unigram[w] += 1  # 增加词频
        else:
            unigram[w] = 1  # 初次计数为 1
    return unigram

UNIGRAM = get_unigram(get_words(open("english_data.txt").read()))

# 过滤非词典中的单词
def known(words):
    return set(w for w in words if w in UNIGRAM)

def correct(word):
    if word not in UNIGRAM:  # 如果单词不在词典中，说明是错误单词
        candidates = known(get_similar_words(word))  # 获取相似词并过滤
        if candidates:
            # 在这里假设所有相似词的相似程度一样，只根据候选项频次大小作为指标进行推荐，即取频次最高的单词
            candidate = max(candidates, key=lambda w: UNIGRAM[w])
            print("‘{}’的推荐纠正项为‘{}’".format(word, candidate))
        else:
            print("‘{}’疑似错误".format(word))
    else:
        print("正确单词")
        return

'''
测试
'''
# 例：过滤后均为正确单词
known(get_similar_words("aand"))

# 例：
correct("word")