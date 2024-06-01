import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载GPT-2模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 添加参数块作为软提示
soft_prompt = "Please ask me a question:"
input_text = "Can you tell me about artificial intelligence?"

# 将输入文本和软提示连接起来
input_text_with_prompt = soft_prompt + input_text

# 分词并编码输入文本
input_ids = tokenizer.encode(input_text_with_prompt, return_tensors='pt')

# 生成文本
output = model.generate(input_ids, max_length=100, temperature=0.7, top_k=50, top_p=0.95, num_return_sequences=1)

# 解码生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:", generated_text)
