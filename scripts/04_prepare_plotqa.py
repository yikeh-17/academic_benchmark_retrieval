import json
import random
import os
import re
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# ==========================================
# 1. 目录配置 (对齐项目架构)
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ★ 你的本地原始 JSON 文件路径 (只保留这一个外部依赖)
INPUT_JSON_PATH = PROJECT_ROOT / "data/raw/plotqa/plot_qa_all.json" 

# 最终我们要统一保存的物理绝对路径
IMAGE_SAVE_DIR = PROJECT_ROOT / "data/raw/plotqa/images"
# 最终生成的标准 JSONL 索引文件路径
OUTPUT_JSONL_PATH = PROJECT_ROOT / "data/processed/plotqa_sample_index.jsonl"

def clean_text(value) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())

def build_text_for_embedding(dataset_name: str, task_type: str, domain: str, abilities: list[str], question: str) -> str:
    return " ".join([
        f"Dataset: {dataset_name}.",
        f"Task: {task_type}.",
        f"Domain: {domain}.",
        "Image type: plot.",
        f"Abilities: {', '.join(abilities)}.",
        f"Question: {question}",
        "Answer type: structured bounding boxes and values."
    ])

def load_json_robustly(filepath: Path):
    print(f"📂 正在读取本地数据集: {filepath.name}")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"\n⚠️ [警告] 发现 JSON 文件损坏 ({e})")
        print("⏳ 正在启动紧急挽救机制...")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        last_brace_idx = content.rfind('}')
        if last_brace_idx != -1:
            fixed_content = content[:last_brace_idx+1] + "\n]"
            try:
                records = json.loads(fixed_content)
                print(f"✅ 成功挽救了 {len(records)} 条完好的数据！")
                return records
            except json.JSONDecodeError:
                print("❌ 文件损坏过于严重，无法完成自动修复。")
                return None
        else:
            print("❌ 在文件中找不到任何完整的 JSON 对象。")
            return None

def main():
    IMAGE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT_JSON_PATH.exists():
        print(f"❌ 错误: 找不到本地原始 JSON 文件: {INPUT_JSON_PATH}")
        return

    # 1. 加载本地 JSON 并抽样
    all_records = load_json_robustly(INPUT_JSON_PATH)
    if not all_records: return
    
    TARGET_COUNT = 500
    print(f"🧩 当前可用本地数据总数: {len(all_records)}")
    
    random.seed(42)
    sampled_records = random.sample(all_records, min(TARGET_COUNT, len(all_records)))

    # 2. 挂载 Hugging Face 远程数据集
    print("🚀 正在连接 Hugging Face，准备精准抓取云端图片...")
    # 这里只会读取元数据，不会把15万张图片塞进内存
    hf_dataset = load_dataset("achang/plot_qa", split="train")

    kept = 0
    missing_image_count = 0

    print(f"🎯 开始从云端提取对应的 {len(sampled_records)} 张图片并构建 JSONL...")
    
    with open(OUTPUT_JSONL_PATH, "w", encoding="utf-8") as output_file:
        for idx, item in enumerate(tqdm(sampled_records, desc="Processing PlotQA")):
            
            conversations = item.get("conversations", [])
            question = ""
            answer = ""
            for msg in conversations:
                if msg.get("role") == "user":
                    question = clean_text(msg.get("content", "").replace("<image>", "").replace("\n", " "))
                elif msg.get("role") == "assistant":
                    answer = msg.get("content", "").strip()

            if not question or not answer:
                continue

            # ==========================================
            # ★ 核心黑科技：提取行号，从 HF 强行抽取图片
            # ==========================================
            # 从 id (如 "plot_qa_22790") 中提取出数字 22790
            item_id = item.get("id", "")
            match = re.search(r'\d+', item_id)
            
            if not match:
                # 如果 id 没数字，尝试从 Windows 路径提取数字
                match = re.search(r'sample_(\d+)', item.get("image", ""))
            
            if not match:
                missing_image_count += 1
                continue
                
            # 获取对应的 Hugging Face 索引
            hf_index = int(match.group())
            
            try:
                # 拿着索引，像查字典一样瞬间从云端/本地缓存抽出对应的图片对象
                hf_record = hf_dataset[hf_index]
                pil_image = hf_record.get("image")
                
                if pil_image is None:
                    raise ValueError("未找到图像特征")
                    
                # 转换格式并保存到 Linux 硬盘
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                    
                image_filename = f"plotqa_{kept:06d}.jpg"
                new_image_filepath = IMAGE_SAVE_DIR / image_filename
                pil_image.save(new_image_filepath)
                
            except Exception as e:
                missing_image_count += 1
                continue # 云端找不到或者越界，跳过
            
            # ==========================================
            # 构造 JSON
            # ==========================================
            abilities = ["plot_understanding", "data_extraction", "structural_parsing", "visual_grounding"]
            task_type = "plot_data_extraction"
            
            text_for_embedding = build_text_for_embedding(
                dataset_name="PlotQA",
                task_type="plot structural data extraction",
                domain="plot reasoning",
                abilities=[a.replace("_", " ") for a in abilities],
                question=question
            )

            output_item = {
                "id": f"plotqa_{kept:06d}",
                "dataset_name": "PlotQA",
                "image": f"data/raw/plotqa/images/{image_filename}", 
                "question": question,
                "answer": answer,
                "modality": "image+text",
                "task_type": task_type,
                "domain": "plot_reasoning",
                "ability": abilities,
                "difficulty": "hard", 
                "text_for_embedding": text_for_embedding
            }

            output_file.write(json.dumps(output_item, ensure_ascii=False) + "\n")
            kept += 1

    print("\n" + "=" * 60)
    print("✅ PlotQA sample-level 云端直抽处理完毕！")
    print(f"📁 成功从云端抽取了 {kept} 张图片至: {IMAGE_SAVE_DIR}")
    print(f"📄 JSONL 索引文件已生成: {OUTPUT_JSONL_PATH}")
    print(f"📊 记录: 期间跳过了 {missing_image_count} 条无法在云端匹配的异常记录。")
    print("=" * 60)

if __name__ == "__main__":
    main()