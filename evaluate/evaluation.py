from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
import torch
import json
import jieba
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import SmoothingFunction

# -------------------------------
# 0. 配置路径
# -------------------------------
base_model_path = Path(r"D:\JetBrains\PycharmProjects\reserch\model_cache\Qwen--Qwen2.5-7B-Instruct\snapshots\a09a35458c702b33eeacc393d103063234e8bc28")
lora_model_path = Path(r"D:\JetBrains\PycharmProjects\reserch\model\lora-model")
eval_data_path = Path(r"D:\JetBrains\PycharmProjects\reserch\datasets\eval.json")

# -------------------------------
# 1. GPU & 4bit 配置
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
print(f"Using device: {device}")

# -------------------------------
# 2. 加载 tokenizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, local_files_only=True)

# -------------------------------
# 3. 加载原始模型
# -------------------------------
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# -------------------------------
# 4. 加载 LoRA 模型
# -------------------------------
print("Loading LoRA model...")
lora_model = PeftModel.from_pretrained(base_model, lora_model_path, device_map="auto")
lora_model.to(device)

# -------------------------------
# 5. 加载评估数据
# -------------------------------
with open(eval_data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

inputs = [x["instruction"] for x in data]
references = [x["output"] for x in data]

# -------------------------------
# 6. 语义相似度模型
# -------------------------------
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# 7. 中文分词函数
# -------------------------------
def tokenize_cn(text):
    return " ".join(jieba.cut(text))

# -------------------------------
# 8. 评估函数
# -------------------------------
def evaluate_model(model, inputs, references, max_new_tokens=128):
    model.eval()
    ppl_list, bleu_list, rouge_list, semantic_sim_list = [], [], [], []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    for inp, ref in zip(inputs, references):
        prompt = f"用户：{inp}\n助手："

        # 编码输入
        inputs_tensor = tokenizer([prompt], return_tensors="pt").to(device)

        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs_tensor,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id
            )

        # 解码
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).split("助手：")[-1].strip()

        # ---------- PPL ----------
        with torch.no_grad():
            enc = tokenizer(prompt + pred, return_tensors="pt").to(device)
            labels = enc["input_ids"]
            outputs_logits = model(**enc).logits
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
            shift_logits = outputs_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            ppl = torch.exp(loss.mean()).item()
            ppl_list.append(ppl)



        # ---------- BLEU ----------
        smooth_fn = SmoothingFunction().method1

        bleu_score = sentence_bleu(
            [tokenize_cn(ref).split()],
            tokenize_cn(pred).split(),
            weights=(0.5, 0.5),
            smoothing_function=smooth_fn
        )
        bleu_list.append(bleu_score)

        # ---------- ROUGE ----------
        rouge_score_dict = scorer.score(tokenize_cn(ref), tokenize_cn(pred))
        rouge_list.append(rouge_score_dict)

        # ---------- Semantic Similarity ----------
        emb_ref = sbert_model.encode(ref, convert_to_tensor=True)
        emb_pred = sbert_model.encode(pred, convert_to_tensor=True)
        sim = util.cos_sim(emb_ref, emb_pred).item()
        semantic_sim_list.append(sim)

    # 平均指标
    avg_ppl = sum(ppl_list) / len(ppl_list)
    avg_bleu = sum(bleu_list) / len(bleu_list)
    avg_rouge1 = sum([r['rouge1'].fmeasure for r in rouge_list]) / len(rouge_list)
    avg_rougeL = sum([r['rougeL'].fmeasure for r in rouge_list]) / len(rouge_list)
    avg_sem_sim = sum(semantic_sim_list) / len(semantic_sim_list)

    return {
        "PPL": avg_ppl,
        "BLEU": avg_bleu,
        "ROUGE-1": avg_rouge1,
        "ROUGE-L": avg_rougeL,
        "Semantic Similarity": avg_sem_sim
    }

# -------------------------------
# 9. 执行评估
# -------------------------------
print("Evaluating base model...")
base_metrics = evaluate_model(base_model, inputs, references)
print("Base model metrics:", base_metrics)

print("\nEvaluating LoRA model...")
lora_metrics = evaluate_model(lora_model, inputs, references)
print("LoRA model metrics:", lora_metrics)