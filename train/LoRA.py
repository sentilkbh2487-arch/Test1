import argparse
import json
from pathlib import Path
import torch
import jieba
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# 参数解析
# -------------------------------
def choose_mode():
    # 如果命令行传了参数，就优先用（兼容论文复现）
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ["base", "lora"]:
            return arg

    # 否则进入交互模式
    while True:
        print("\n请选择评估模式：")
        print("1. Base Model")
        print("2. LoRA Model")

        choice = input("请输入编号 (1/2)：").strip()

        if choice == "1":
            return "base"
        elif choice == "2":
            return "lora"
        else:
            print("输入错误，请重新输入！")

mode = choose_mode()

# -------------------------------
# 路径
# -------------------------------
base_model_path = Path(r"D:\JetBrains\PycharmProjects\reserch\model_cache\Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28")
lora_model_path = Path(r"D:\JetBrains\PycharmProjects\reserch\model\lora-model\checkpoint-348")  # ⚠️ 改成最新checkpoint
eval_data_path = Path(r"D:\JetBrains\PycharmProjects\reserch\datasets\eval.json")

# -------------------------------
# 设备
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# -------------------------------
# tokenizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    local_files_only=True
)

# -------------------------------
# 加载模型（关键）
# -------------------------------
def load_model(mode):
    print(f"Loading {mode} model...")

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    if mode == "base":
        return base

    elif mode == "lora":
        model = PeftModel.from_pretrained(
            base,
            lora_model_path,
            device_map="auto"
        )
        return model

model = load_model(mode)

# -------------------------------
# 数据
# -------------------------------
with open(eval_data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

inputs = [x["instruction"] for x in data]
references = [x["output"] for x in data]

# -------------------------------
# 语义模型
# -------------------------------
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def tokenize_cn(text):
    return " ".join(jieba.cut(text))

# -------------------------------
# 评估函数
# -------------------------------
def evaluate(model):
    model.eval()

    ppl_list, bleu_list, rouge_list, sim_list = [], [], [], []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1

    for inp, ref in zip(inputs, references):

        prompt = f"用户：{inp}\n助手："

        inputs_tensor = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs_tensor,
                max_new_tokens=128,
                do_sample=False
            )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = pred.split("助手：")[-1].strip()

        # ---------- PPL ----------
        with torch.no_grad():
            enc = tokenizer(prompt + ref, return_tensors="pt").to(device)
            labels = enc["input_ids"]

            out = model(**enc, labels=labels)
            loss = out.loss
            ppl = torch.exp(loss).item()

        ppl_list.append(ppl)

        # ---------- BLEU ----------
        bleu = sentence_bleu(
            [tokenize_cn(ref).split()],
            tokenize_cn(pred).split(),
            smoothing_function=smooth
        )
        bleu_list.append(bleu)

        # ---------- ROUGE ----------
        rouge = scorer.score(tokenize_cn(ref), tokenize_cn(pred))
        rouge_list.append(rouge)

        # ---------- 语义 ----------
        emb1 = sbert_model.encode(ref, convert_to_tensor=True)
        emb2 = sbert_model.encode(pred, convert_to_tensor=True)

        sim = util.cos_sim(emb1, emb2).item()
        sim_list.append(sim)

    return {
        "PPL": sum(ppl_list) / len(ppl_list),
        "BLEU": sum(bleu_list) / len(bleu_list),
        "ROUGE-1": sum(r['rouge1'].fmeasure for r in rouge_list) / len(rouge_list),
        "ROUGE-L": sum(r['rougeL'].fmeasure for r in rouge_list) / len(rouge_list),
        "Semantic Similarity": sum(sim_list) / len(sim_list)
    }

# -------------------------------
# 执行
# -------------------------------
metrics = evaluate(model)

print(f"\n=== {mode.upper()} RESULT ===")
print(metrics)