import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import multiprocessing

# 提取句子特征
def sentence_features(st, ix):
    d_ft = {
        'word': st[ix],
        'dist_from_first': ix,
        'dist_from_last': len(st) - ix,
        'capitalized': st[ix][0].upper() == st[ix][0],
        'prefix1': st[ix][0],
        'prefix2': st[ix][:2],
        'prefix3': st[ix][:3],
        'suffix1': st[ix][-1],
        'suffix2': st[ix][-2:],
        'suffix3': st[ix][-3:],
        'prev_word': '' if ix == 0 else st[ix - 1],
        'next_word': '' if ix == (len(st) - 1) else st[ix + 1],
        'numeric': st[ix].isdigit()
    }
    return d_ft

# 获取未标记的句子
def get_untagged_sentence(tagged_sentence):
    return [s for s, t in tagged_sentence]

# 提取特征和标签
def ext_ft(tg_sent):
    sent, tag = [], []
    for index in range(len(tg_sent)):
        sent.append(sentence_features(get_untagged_sentence(tg_sent), index))
        tag.append(tg_sent[index][1])
    return sent, tag

tagged_sentences = nltk.corpus.treebank.tagged_sents(tagset='universal')

# 并行提取特征
with multiprocessing.Pool() as pool:
    results = pool.map(ext_ft, tagged_sentences)

# 将结果展平
X, y = zip(*results)
X = [item for sublist in X for item in sublist]
y = [item for sublist in y for item in sublist]

n_sample = 50000
dict_vectorizer = DictVectorizer(sparse=True)  # 使用稀疏矩阵
X_transformed = dict_vectorizer.fit_transform(X[:n_sample])
y_sampled = y[:n_sample]

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_sampled, test_size=0.2, random_state=123)

rf = RandomForestClassifier(n_jobs=-1)  # 使用所有可用的核
rf.fit(X_train, y_train)

def predict_pos_tags(sentence):
    features = [sentence_features(sentence, index) for index in range(len(sentence))]
    features = dict_vectorizer.transform(features)
    tags = rf.predict(features)
    return zip(sentence, tags)

test_sentence = "This is a simple POS tagger"
for tagged in predict_pos_tags(test_sentence.split()):
    print(tagged)

predictions = rf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(10, 10))
plt.xticks(np.arange(len(rf.classes_)), rf.classes_)
plt.yticks(np.arange(len(rf.classes_)), rf.classes_)
plt.imshow(conf_matrix, cmap=plt.cm.Blues)
plt.colorbar()

feature_list = zip(dict_vectorizer.get_feature_names_out(), rf.feature_importances_)
sorted_features = sorted(feature_list, key=lambda x: x[1], reverse=True)
print(sorted_features[:20])
