import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric

# 加载数据集，这里使用IMDB电影评论数据集
dataset = load_dataset("imdb")

# 加载BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# 定义分词函数
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# 对数据集进行分词
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 数据集格式转换
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 划分训练和验证集
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 定义评估指标
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return metric.compute(predictions=predictions, references=labels)

# 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()

# 评估模型
trainer.evaluate()

#1. 超参数调优（Hyperparameter Tuning）  
training_args = TrainingArguments(  
    output_dir="./results",  
    evaluation_strategy="epoch",  
    learning_rate=3e-5,  # 可以尝试不同的学习率  
    per_device_train_batch_size=16,  # 适当调整批次大小  
    per_device_eval_batch_size=16,  
    num_train_epochs=4,  # 增加或减少训练轮数  
    weight_decay=0.01,  
)  
  
#2.数据增强（Data Augmentation）  
''''' 
通过生成新的训练样本或对现有数据进行变换，可以使模型更好地泛化。例如，使用同义词替换、随机删除、随机插入等方法进行数据增强。 
'''  
  
#3. 梯度累积（Gradient Accumulation）  
training_args = TrainingArguments(  
    output_dir="./results",  
    evaluation_strategy="epoch",  
    learning_rate=2e-5,  
    per_device_train_batch_size=8,  
    gradient_accumulation_steps=4,  # 累积4个小批次的梯度，相当于总批次大小为32  
    num_train_epochs=3,  
    weight_decay=0.01,  
)  
  
#4. 冻结层（Layer Freezing）  
for param in model.bert.parameters():  
    param.requires_grad = False  
  
# 只训练顶部的分类头  
training_args = TrainingArguments(  
    output_dir="./results",  
    evaluation_strategy="epoch",  
    learning_rate=2e-5,  
    per_device_train_batch_size=16,  
    num_train_epochs=2,  
    weight_decay=0.01,  
)  
  
# 解冻所有层进行进一步训练  
for param in model.bert.parameters():  
    param.requires_grad = True  
  
# 进行更多轮训练  
training_args.num_train_epochs = 3  
  
#5. 提前停止（Early Stopping）  
from transformers import EarlyStoppingCallback  
  
trainer = Trainer(  
    model=model,  
    args=training_args,  
    train_dataset=small_train_dataset,  
    eval_dataset=small_eval_dataset,  
    compute_metrics=compute_metrics,  
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # 设定提前停止的耐心值  
)  
  
#6. 使用学习率调度器（Learning Rate Schedulers）  
'''
from transformers import get_linear_schedule_with_warmup  
  
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)  
num_training_steps = len(train_dataloader) * training_args.num_train_epochs  
scheduler = get_linear_schedule_with_warmup(  
    optimizer, num_warmup_steps=0, num_training_steps=num_training_steps  
)  
  
trainer = Trainer(  
    model=model,  
    args=training_args,  
    train_dataset=small_train_dataset,  
    eval_dataset=small_eval_dataset,  
    compute_metrics=compute_metrics,  
    optimizers=(optimizer, scheduler)  # 使用优化器和学习率调度器  
)  
'''