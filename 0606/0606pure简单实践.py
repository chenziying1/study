import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 加载spaCy的预训练模型
nlp = spacy.load("zh_core_web_lg")

# 示例文本
text = "2024年6月6日，微软公司宣布与OpenAI建立战略合作伙伴关系，双方将共同推进人工智能技术的发展。微软CEO萨提亚·纳德拉表示，这一合作将为全球客户带来更多创新和价值。"

# 预处理和实体识别
doc = nlp(text)

entities = [(ent.text, ent.label_) for ent in doc.ents]
print("Entities:", entities)

# 示例实体对和对应的上下文（用于训练关系分类模型）
train_data = [
    ("微软公司", "OpenAI", "建立战略合作伙伴关系"),
    ("微软公司", "萨提亚·纳德拉", "微软CEO"),
]

# 提取特征：我们将简单地使用实体对之间的上下文作为特征
def extract_features(entity1, entity2, context):
    return context

X_train = [extract_features(e1, e2, ctx) for e1, e2, ctx in train_data]
y_train = ["合作伙伴", "CEO"]  # 假设我们有标注的关系类型

# 使用CountVectorizer和LogisticRegression进行关系分类
vectorizer = CountVectorizer()
classifier = LogisticRegression()

pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("classifier", classifier),
])

pipeline.fit(X_train, y_train)

# 从新文本中抽取关系
def extract_relations(doc):
    relations = []
    for ent1 in doc.ents:
        for ent2 in doc.ents:
            if ent1.start < ent2.start:
                context = doc[ent1.end:ent2.start].text
                feature = extract_features(ent1.text, ent2.text, context)
                relation = pipeline.predict([feature])
                relations.append((ent1.text, ent2.text, relation[0]))
    return relations

relations = extract_relations(doc)
print("Relations:", relations)
