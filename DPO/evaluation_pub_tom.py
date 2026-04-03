import os
import csv
import json
import pandas as pd
from tqdm import tqdm
import zipfile
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
from sklearn.metrics import accuracy_score, recall_score, f1_score
from huggingface_hub import hf_hub_download


# ===============================
# CONFIG
# ===============================
custom_cache_dir = "/extra/agnese.lombardi/PUB/cfilt___PUB"
base_model_id = "meta-llama/Meta-Llama-3-8B"
lora_path = "/extra/agnese.lombardi/ISA_ToM/ToM_dpo"
RESULTS_CSV_PATH = "results/pub_results_tom.csv"
TASKS = [str(i) for i in range(1, 15)]

# ===============================
# LOAD TASK FROM ZIP FILES
# ===============================
def load_task_manually(task_id):
    """
    Load a specific task from its ZIP file in the HuggingFace dataset
    """
    try:
        print(f"Loading task {task_id}...")
        
        # Download the specific task ZIP file
        zip_file = hf_hub_download(
            repo_id="cfilt/PUB",
            filename=f"data/task_{task_id}.zip",
            repo_type="dataset",
            cache_dir=custom_cache_dir
        )
        
        print(f"Found ZIP file: {zip_file}")
        
        # Extract and parse the JSONL from the ZIP
        data = []
        with zipfile.ZipFile(zip_file, 'r') as z:
            jsonl_filename = f"task_{task_id}.jsonl"
            
            if jsonl_filename not in z.namelist():
                print(f"Could not find {jsonl_filename} in the zip file")
                print(f"Available files: {z.namelist()}")
                return None
            
            with z.open(jsonl_filename) as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        # Skip known problematic line in task 14 (row 260, 0-indexed = 259)
                        if task_id == "14" and i == 259:
                            print(f"Skipping known problematic line 260 in task 14")
                            continue
                        
                        item = json.loads(line.decode('utf-8'))
                        
                        # Convert any non-string "correct answer" to string
                        if "correct answer" in item:
                            if not isinstance(item["correct answer"], str):
                                item["correct answer"] = str(item["correct answer"])
                        
                        data.append(item)
                        
                    except json.JSONDecodeError as e:
                        print(f"Skipping malformed JSON at line {i+1}: {str(e)}")
                    except Exception as e:
                        print(f"Skipping line {i+1} due to error: {str(e)}")
        
        if not data:
            print(f"No valid data found for task {task_id}")
            return None
        
        # Create dataset
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        print(f"✓ Successfully loaded task {task_id} with {len(dataset)} examples")
        return dataset
        
    except Exception as e:
        print(f"Error loading task {task_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ===============================
# LOAD MODEL + TOKENIZER
# ===============================
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, lora_path)
model.to("cuda:0")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
model.eval()


# ===============================
# EVALUATION
# ===============================
def evaluate_pub():
    results = {}
    for task_id in TASKS:
        print(f"\n🔍 Task {task_id}")
        task_ds = load_task_manually(task_id)

        if task_ds is None or len(task_ds) == 0:
            results[f"PUB Task {task_id}"] = {"accuracy": 0, "recall": 0, "f1": 0}
            continue

        y_true, y_pred = [], []

        for ex in tqdm(task_ds, desc=f"Evaluating Task {task_id}"):
            question = ex.get("pretext")
            options = ex.get("options")
            gold = ex.get("correct answer")

            if not question or not options or not gold:
                continue

            prompt = (
                f"{question}\nOptions:\n"
                + "\n".join(f"- {o}" for o in options)
                + "\nAnswer:"
            )

            out = generator(
                prompt,
                max_new_tokens=10,
                do_sample=False,
                return_full_text=False,
            )[0]["generated_text"].lower()

            pred = next((o.lower() for o in options if o.lower() in out), "unknown")

            if pred != "unknown":
                y_true.append(gold.lower())
                y_pred.append(pred)

        if not y_true:
            results[f"PUB Task {task_id}"] = {"accuracy": 0, "recall": 0, "f1": 0}
            continue

        results[f"PUB Task {task_id}"] = {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        }

    return results

# ===============================
# SAVE RESULTS
# ===============================
def save_results(results):
    os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
    with open(RESULTS_CSV_PATH, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Accuracy", "Recall", "F1"])
        for k, v in results.items():
            writer.writerow([k, v["accuracy"], v["recall"], v["f1"]])

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    print("🚀 Evaluating PUB benchmark")
    results = evaluate_pub()
    save_results(results)
    print("✅ Done")
