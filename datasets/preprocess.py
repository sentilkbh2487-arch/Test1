import json

# -----------------------------
# 1️⃣ 输入输出路径
# -----------------------------
input_path = r"/datasets/toy_data.json"
output_path = r"/datasets/train_lora.json"

# -----------------------------
# 2️⃣ 读取原始数据
# -----------------------------
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

train_dialogs = data.get("train", [])
valid_dialogs = data.get("valid", [])

# -----------------------------
# 3️⃣ 数据处理函数
# -----------------------------
def process_dialogs(dialogs):
    processed = []
    for dialog in dialogs:
        # dialog 是多轮列表
        for i in range(len(dialog) - 1):
            input_text = dialog[i].replace(" ", "").strip()
            output_text = dialog[i+1].replace(" ", "").strip()

            # 过滤太短或无意义的文本
            if len(input_text) < 2 or len(output_text) < 2:
                continue
            if input_text in ["嗯", "哦", "哈哈", "嗯嗯"] or output_text in ["嗯", "哦", "哈哈", "嗯嗯"]:
                continue

            processed.append({
                "instruction": input_text,
                "output": output_text
            })
    return processed

# -----------------------------
# 4️⃣ 处理训练集和验证集
# -----------------------------
train_data = process_dialogs(train_dialogs)
valid_data = process_dialogs(valid_dialogs)

# -----------------------------
# 5️⃣ 保存处理后的数据
# -----------------------------
# 合并成一个文件（可以按需要分开保存）
final_data = {
    "train": train_data,
    "valid": valid_data
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print(f"处理完成！训练样本数：{len(train_data)}，验证样本数：{len(valid_data)}")
print(f"保存路径：{output_path}")