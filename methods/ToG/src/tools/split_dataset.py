import json
import random

with open('A.json', 'r', encoding='utf-8') as f:
    original_data = json.load(f)

# 读取A.json数据集后随机打乱，然后取前1000条，保存为B.json
random.shuffle(original_data)
new_data = original_data[:1000]

with open('B.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f)