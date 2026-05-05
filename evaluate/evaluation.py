import argparse
import json
from pathlib import Path
import torch
import jieba
import sys
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

# -------------------------------
# 参数解析
# -------------------------------
def choose_mode():
    # 如果命令行传了参数，就优先用
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
lora_model_path = Path(r"D:\JetBrains\PycharmProjects\reserch\model\lora-model\checkpoint-549")  # 最新 checkpoint
eval_data_path = Path(r"D:\JetBrains\PycharmProjects\reserch\datasets\eval.json")
style_path = Path(r"D:\JetBrains\PycharmProjects\reserch\datasets\style_corpus.txt")

# -------------------------------
# 输出文件路径（根据模式区分）
# -------------------------------
output_dir = Path(r"D:\JetBrains\PycharmProjects\reserch\results")
output_dir.mkdir(parents=True, exist_ok=True)
output_file_path = output_dir / f"eval_results_{mode}.json"

# -------------------------------
# 设备 & 量化配置
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

# -------------------------------
# Tokenizer（统一使用 base + 手动扩展）
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    local_files_only=True
)

# 🔥 你的自定义 EOS token（必须和训练一致）
EOS_TOKEN = "<|endofanswer|>"

if EOS_TOKEN not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"additional_special_tokens": [EOS_TOKEN]})

# 避免 warning
tokenizer.pad_token = tokenizer.eos_token


# -------------------------------
# 加载模型（关键修改）
# -------------------------------
def load_model(mode):
    print(f"Loading {mode} model...")

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    # 🔥 关键：让 base 模型适配 tokenizer（解决 size mismatch）
    base.resize_token_embeddings(len(tokenizer))

    if mode == "base":
        return base

    elif mode == "lora":
        model = PeftModel.from_pretrained(
            base,
            str(lora_model_path),
            device_map="auto"
        )
        return model


model = load_model(mode)
model.to(device)

# -------------------------------
# 数据
# -------------------------------
with open(eval_data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

inputs = [x["instruction"] for x in data]
references = [x["output"] for x in data]

# -------------------------------
# SBERT
# -------------------------------
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# 风格语料
# -------------------------------
with open(style_path, "r", encoding="utf-8") as f:
    style_texts = [line.strip() for line in f if line.strip()]

style_emb = sbert_model.encode(style_texts, convert_to_tensor=True)

# -------------------------------
# ROUGE / BLEU
# -------------------------------
scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
smooth = SmoothingFunction().method1

# -------------------------------
# 工具函数
# -------------------------------
def clean_pred(text):
    if "回答：" in text:
        text = text.split("回答：")[-1]
    return text.strip()


# -------------------------------
# 评估函数
# -------------------------------
def evaluate_and_save(model, inputs, references, tokenizer, device, output_file):

    model.eval()
    ppl_list = []
    bleu_list = []
    rouge_list = []
    style_list = []
    content_list = []

    results = []

    unwanted_prefixes = ["用户：", "助手：", "Human:", "问题："]

    for inp, ref in zip(inputs, references):
        prompt = f"你是一个温柔细腻的人，直接回答下面的问题，不编造信息，也不要复述问题：\n问题：{inp}\n回答："
        inputs_tensor = tokenizer(prompt, return_tensors="pt").to(device)

        # 根据输入长度自适应生成长度
        input_len = len(tokenizer.encode(prompt))
        min_gen_len = 8
        max_gen_len = 48
        gen_len = min(max(int(input_len * 0.8), min_gen_len), max_gen_len)

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs_tensor["input_ids"],
                attention_mask=inputs_tensor.get("attention_mask"),
                max_new_tokens=gen_len,
                do_sample=True,       # 启用采样避免重复
                top_p=0.8,
                top_k=30,
                temperature=0.6,      # 限制随机性
                repetition_penalty=1.8,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,  # 达到 EOS 时提前结束
                length_penalty=0.1    # <1.0更倾向短句, >1.0更长
            )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = clean_pred(pred)

        for p in unwanted_prefixes:
            if pred.startswith(p):
                pred = pred[len(p):].strip()

        if len(pred) == 0:
            pred = "无回答"

        # ===============================
        # PPL
        # ===============================
        with torch.no_grad():
            enc = tokenizer(prompt + ref, return_tensors="pt").to(device)
            out = model(**enc, labels=enc["input_ids"])
            ppl = torch.exp(out.loss).item()
            ppl_list.append(ppl)

        # ===============================
        # BLEU（字符级）
        # ===============================
        bleu = sentence_bleu(
            [list(ref)],
            list(pred),
            smoothing_function=smooth
        )
        bleu_list.append(bleu)

        # ===============================
        # ROUGE
        # ===============================
        rouge = scorer.score(ref, pred)
        rouge_list.append(rouge)

        # ===============================
        # 风格相似度（核心）
        # ===============================
        emb_pred = sbert_model.encode(pred, convert_to_tensor=True)
        style_score = util.cos_sim(emb_pred, style_emb).mean().item()
        style_list.append(style_score)

        # ===============================
        # 内容相似度
        # ===============================
        emb_ref = sbert_model.encode(ref, convert_to_tensor=True)
        content_score = util.cos_sim(emb_ref, emb_pred).item()
        content_list.append(content_score)

        # ===============================
        # 保存样本
        # ===============================
        results.append({
            "instruction": inp,
            "reference": ref,
            "prediction": pred,
            "PPL": ppl,
            "BLEU": bleu,
            "ROUGE-1": rouge["rouge1"].fmeasure,
            "ROUGE-L": rouge["rougeL"].fmeasure,
            "Style Similarity": style_score,
            "Content Similarity": content_score,
            "gen_len": gen_len
        })

    # ===============================
    # 平均指标（必须在循环外）
    # ===============================
    avg_metrics = {
        "PPL": sum(ppl_list) / len(ppl_list),
        "BLEU": sum(bleu_list) / len(bleu_list),
        "ROUGE-1": sum(r["rouge1"].fmeasure for r in rouge_list) / len(rouge_list),
        "ROUGE-L": sum(r["rougeL"].fmeasure for r in rouge_list) / len(rouge_list),
        "Style Similarity": sum(style_list) / len(style_list),
        "Content Similarity": sum(content_list) / len(content_list)
    }

    # ===============================
    # 保存
    # ===============================
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "mode": mode,
            "average_metrics": avg_metrics,
            "samples": results
        }, f, ensure_ascii=False, indent=2)

    return avg_metrics


# -------------------------------
# run
# -------------------------------
metrics = evaluate_and_save(
    model,
    inputs,
    references,
    tokenizer,
    device,
    output_file_path
)

print("\n=== FINAL METRICS ===")
print(metrics)
print("\nSaved to:", output_file_path)