import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -------------------------------
# 1. 指定缓存目录（D盘）
# -------------------------------
cache_dir = r"D:\JetBrains\PycharmProjects\reserch\model_cache"

# 国内镜像加速
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"

# -------------------------------
# 2. 模型名称和量化配置
# -------------------------------
model_name = "Qwen/Qwen2.5-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# -------------------------------
# 3. 下载 tokenizer
# -------------------------------
print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True
)

# -------------------------------
# 4. 下载模型
# -------------------------------
print("Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",             # 自动映射显卡
    quantization_config=bnb_config,
    cache_dir=cache_dir,           # 指定缓存目录
    trust_remote_code=True
)

print("Model downloaded and cached successfully!")
print(f"Cache path: {cache_dir}")