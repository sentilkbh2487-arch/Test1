from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path
import torch
import os

# -------------------------------
# 0. 配置
# -------------------------------
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# -------------------------------
# 1. 指定本地模型路径（含 tokenizer.json 的目录）
# -------------------------------
local_model_path = Path(
    r"D:\JetBrains\PycharmProjects\reserch\model_cache\Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28"
)

# -------------------------------
# 2. 4bit量化配置（节省显存）
# -------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# -------------------------------
# 3. GPU 检测
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# 4. 加载 tokenizer
# -------------------------------
print("Loading tokenizer from local path...")
tokenizer = AutoTokenizer.from_pretrained(
    local_model_path,
    trust_remote_code=True,
    local_files_only=True
)

# -------------------------------
# 5. 加载模型（自动映射到 GPU）
# -------------------------------
print("Loading model from local path...")
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    device_map="auto",                 # 自动分配 GPU
    quantization_config=bnb_config,    # 4bit 量化
    trust_remote_code=True
)

print("Model loaded successfully!")

# -------------------------------
# 6. 简单对话循环
# -------------------------------
while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    messages = [{"role": "user", "content": user_input}]

    # 将 messages 转成模型可读文本
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 编码输入到 GPU
    inputs = tokenizer([text], return_tensors="pt").to(device)

    # 生成回答
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id
    )

    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 只取 assistant 回答部分
    print("\nAI:", response.split("assistant")[-1].strip())