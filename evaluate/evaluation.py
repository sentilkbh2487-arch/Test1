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
lora_model_path = Path(r"D:\JetBrains\PycharmProjects\reserch\model\lora-model\checkpoint-348")  # 最新 checkpoint
eval_data_path = Path(r"D:\JetBrains\PycharmProjects\reserch\datasets\eval.json")

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
# Tokenizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    local_files_only=True
)

# -------------------------------
# 加载模型
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
# 中文分词 & 语义模型
# -------------------------------
def tokenize_cn(text):
    return " ".join(jieba.cut(text))

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# 评估函数
# -------------------------------
def evaluate_and_save(model, inputs, references, tokenizer, device, output_file):
    model.eval()
    results = []

    ppl_list, bleu_list, rouge_list, sim_list = [], [], [], []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    unwanted_prefixes = ["用户：", "助手：", "Human:", "问题："]

    for inp, ref in zip(inputs, references):
        prompt = f"你是一个温柔细腻的人，直接回答下面的问题，不编造信息，也不要复述问题：\n问题：{inp}\n回答："
        inputs_tensor = tokenizer(prompt, return_tensors="pt").to(device)

        # 根据输入长度自适应生成长度
        input_len = len(tokenizer.encode(prompt))
        min_gen_len = 8
        max_gen_len = 64
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
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,  # 达到 EOS 时提前结束
                length_penalty=0.1    # <1.0更倾向短句, >1.0更长
            )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 去掉 prompt 前缀
        if "回答：" in pred:
            pred = pred.split("回答：")[-1].strip()

        # 检测到无关前缀就删除从该前缀开始的整段内容
        for prefix in unwanted_prefixes:
            if prefix in pred:
                pattern = re.escape(prefix) + ".*"
                pred = re.sub(pattern, "", pred, flags=re.DOTALL).strip()

        # ---------- PPL ----------
        with torch.no_grad():
            enc = tokenizer(prompt + ref, return_tensors="pt").to(device)
            labels = enc["input_ids"]
            out = model(**enc, labels=labels)
            loss = out.loss
            ppl = torch.exp(loss).item()
            ppl_list.append(ppl)

        # ---------- BLEU ----------
        bleu = sentence_bleu([tokenize_cn(ref).split()], tokenize_cn(pred).split(), smoothing_function=smooth)
        bleu_list.append(bleu)

        # ---------- ROUGE ----------
        rouge_score_val = scorer.score(tokenize_cn(ref), tokenize_cn(pred))
        rouge_list.append(rouge_score_val)

        # ---------- 语义相似度 ----------
        emb_ref = sbert_model.encode(ref, convert_to_tensor=True)
        emb_pred = sbert_model.encode(pred, convert_to_tensor=True)
        sim = util.cos_sim(emb_ref, emb_pred).item()
        sim_list.append(sim)

        # ---------- 保存每条样本 ----------
        results.append({
            "instruction": inp,
            "reference": ref,
            "prediction": pred,
            "PPL": ppl,
            "BLEU": bleu,
            "ROUGE-1": rouge_score_val['rouge1'].fmeasure,
            "ROUGE-L": rouge_score_val['rougeL'].fmeasure,
            "Semantic Similarity": sim
        })

    # ---------- 平均指标 ----------
    avg_metrics = {
        "PPL": sum(ppl_list)/len(ppl_list),
        "BLEU": sum(bleu_list)/len(bleu_list),
        "ROUGE-1": sum(r['rouge1'].fmeasure for r in rouge_list)/len(rouge_list),
        "ROUGE-L": sum(r['rougeL'].fmeasure for r in rouge_list)/len(rouge_list),
        "Semantic Similarity": sum(sim_list)/len(sim_list)
    }

    # ---------- 保存 JSON ----------
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "mode": mode,
            "average_metrics": avg_metrics,
            "samples": results
        }, f, ensure_ascii=False, indent=2)

    return avg_metrics, results

# -------------------------------
# 执行评估
metrics, all_results = evaluate_and_save(model, inputs, references, tokenizer, device, output_file_path)

print(f"\n=== {mode.upper()} METRICS ===")
print(metrics)
print(f"\n评估完成！结果已保存到 {output_file_path}")