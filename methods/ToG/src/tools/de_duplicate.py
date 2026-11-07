import json
from collections import OrderedDict

with open("A.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# 根据quesion字段对数据集进行去重
result = list(OrderedDict((item['question'], item) for item in data).values())

with open("B.json", "w", encoding="utf-8") as file:
    json.dump(result, file, ensure_ascii=False)
