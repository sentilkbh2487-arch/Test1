# train/train_lora.py
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

# -----------------------------
# 配置路径
# -----------------------------
dataset_path = Path(r"D:\JetBrains\PycharmProjects\reserch\datasets\train_lora.json")
model_cache_path = r"D:\\JetBrains\\PycharmProjects\\reserch\\model_cache\\Qwen--Qwen2.5-7B-Instruct\\snapshots\\a09a35458c702b33eeacc393d103063234e8bc28"
output_model_path = Path(r"D:\\JetBrains\\PycharmProjects\\reserch\\model\\lora-model")  # 保存微调模型

# -----------------------------
# 加载数据
# -----------------------------
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

train_data = data["train"]
valid_data = data["valid"]

# -----------------------------
# 加载 tokenizer 和模型
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_cache_path, trust_remote_code=True, local_files_only=True)

# ⚠️ 强制全模型到 GPU，避免 meta-device 错误
model = AutoModelForCausalLM.from_pretrained(
    model_cache_path,
    device_map=None,         # 不使用 auto，手动搬到 GPU
    trust_remote_code=True,
    torch_dtype=torch.float16
)
model = model.to("cuda")  # 🔑 所有层放到 GPU

# -----------------------------
# LoRA 配置
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],  # 注意力层注入
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# -----------------------------
# 数据 Tokenize（Instruction-Response 格式）
# -----------------------------
MAX_LENGTH = 128  # 显存受限，128 足够大部分指令

def preprocess(example):
    instruction = example["instruction"]
    output = example["output"]
    text = f"用户：{instruction}\n助手：{output}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH)

train_encodings = [preprocess(x) for x in train_data]
valid_encodings = [preprocess(x) for x in valid_data]

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        # ⚠️ labels 需要和 input_ids 一致，用于 CausalLM loss 计算
        item = {k: torch.tensor(v) for k, v in self.encodings[idx].items()}
        item['labels'] = item['input_ids'].clone()
        return item

train_dataset = Dataset(train_encodings)
valid_dataset = Dataset(valid_encodings)

# -----------------------------# 训练参数
# -----------------------------
training_args = TrainingArguments(
    output_dir=output_model_path,
    per_device_train_batch_size=1,   # batch size = 1
    gradient_accumulation_steps=8,   # 累积 8 步等效 batch 8
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    eval_strategy="epoch",            # 每轮结束评估一次
    save_total_limit=2,
    fp16=True,                        # 混合精度
    optim="adamw_torch",
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)
# -----------------------------
# 开始训练
# -----------------------------
trainer.train()

# -----------------------------
# 保存微调模型
# -----------------------------
output_model_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(output_model_path)
tokenizer.save_pretrained(output_model_path)
print("LoRA 微调完成，模型已保存到 model/lora-model")