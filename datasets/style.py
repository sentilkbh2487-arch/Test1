import json
import re

# ===== 1. 读取数据 =====
with open("toy_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)["valid"]


# ===== 2. 清洗函数 =====
def clean_text(text):
    # 去掉多余空格（关键）
    text = text.replace(" ", "")

    # 去掉奇怪符号（可选）
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9，。！？]", "", text)

    return text.strip()


# ===== 3. 构建风格语料 =====
style_texts = []

for dialog in data:
    for sentence in dialog:
        cleaned = clean_text(sentence)

        # 过滤太短的句子（可选）
        if len(cleaned) >= 3:
            style_texts.append(cleaned)


# ===== 4. 去重 =====
style_texts = list(set(style_texts))

# ===== 5. 保存 =====
with open("style_corpus.txt", "w", encoding="utf-8") as f:
    for line in style_texts:
        f.write(line + "\n")

print(f"共生成风格语料：{len(style_texts)} 条")