# from peft import PeftModel
# from transformers import AutoModelForCausalLM, AutoTokenizer

# base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
# base_model.save_pretrained("/extra/agnese.lombardi/ISA_ToM/pragmatics_fine_tuned")

# model = PeftModel.from_pretrained(
#     base_model,
#     "/extra/agnese.lombardi/ISA_ToM/pragmatics_fine_tuned",
#     local_files_only=True
# )

import os
import json
from datasets import load_dataset, Dataset, concatenate_datasets

data_dir = "/home/agnese.lombardi/.cache/huggingface/datasets/cfilt___PUB/65a6a87359fe4aa5278952741d5ed7e0ecb0f0ff/data"
task_files = [f for f in os.listdir(data_dir) if f.startswith("task_") and f.endswith(".jsonl")]
all_datasets = []

def clean_jsonl(path):
    cleaned_lines = []
    with open(path, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if not isinstance(obj.get("correct answer", ""), str):
                    obj["correct answer"] = str(obj["correct answer"])
                cleaned_lines.append(json.dumps(obj))
            except Exception as e:
                print(f"Skipping malformed line in {path}: {e}")
    cleaned_path = path.replace(".jsonl", "_cleaned.jsonl")
    with open(cleaned_path, "w") as f:
        f.write("\n".join(cleaned_lines))
    return cleaned_path

for file_name in task_files:
    file_path = os.path.join(data_dir, file_name)
    try:
        ds = load_dataset("json", data_files=file_path, split="train")
        all_datasets.append(ds)
        print(f"Loaded {file_name}")
    except Exception as e:
        print(f"Failed to load {file_name}: {e}")
        try:
            cleaned = clean_jsonl(file_path)
            ds = load_dataset("json", data_files=cleaned, split="train")
            all_datasets.append(ds)
            print(f"Loaded cleaned version of {file_name}")
        except Exception as inner_e:
            print(f"Still failed after cleaning {file_name}: {inner_e}")

