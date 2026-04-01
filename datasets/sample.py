import json
from random import sample

with open("train_lora.json", "r", encoding="utf-8") as f:
    data = json.load(f)

train_data = data["train"]

# 随机选 50 条作为 eval
eval_data = sample(train_data, 1000)

with open("eval.json", "w", encoding="utf-8") as f:
    json.dump(eval_data, f, ensure_ascii=False, indent=2)