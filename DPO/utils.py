import os
import json
from datasets import load_dataset, Dataset, concatenate_datasets

# Your custom cache folder
custom_cache_dir = "/extra/agnese.lombardi/PUB"

# Find the correct subfolder for the dataset
dataset_root = os.path.join(custom_cache_dir, "cfilt___PUB")
# The actual data files are usually under a folder named "data" or a hash folder
data_dir = None
for root, dirs, files in os.walk(dataset_root):
    jsonl_files = [f for f in files if f.startswith("task_") and f.endswith(".jsonl")]
    if jsonl_files:
        data_dir = root
        break

if not data_dir:
    raise FileNotFoundError(f"No task_*.jsonl files found under {dataset_root}")

print(f"Using data directory: {data_dir}")

task_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("task_") and f.endswith(".jsonl")]
all_datasets = []

def clean_jsonl(path):
    cleaned_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
                if "correct answer" in obj and not isinstance(obj["correct answer"], str):
                    obj["correct answer"] = str(obj["correct answer"])
                cleaned_lines.append(obj)
            except Exception as e:
                print(f"Skipping malformed line {i+1} in {path}: {e}")
    cleaned_path = path.replace(".jsonl", "_cleaned.jsonl")
    with open(cleaned_path, "w", encoding="utf-8") as f:
        for item in cleaned_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return cleaned_path

for file_path in task_files:
    file_name = os.path.basename(file_path)
    try:
        ds = load_dataset("json", data_files=file_path, split="train")
        all_datasets.append(ds)
        print(f"Loaded {file_name}")
    except Exception as e:
        print(f"Failed to load {file_name}: {e}")
        try:
            cleaned_path = clean_jsonl(file_path)
            ds = load_dataset("json", data_files=cleaned_path, split="train")
            all_datasets.append(ds)
            print(f"Loaded cleaned version of {file_name}")
        except Exception as inner_e:
            print(f"Still failed after cleaning {file_name}: {inner_e}")

if all_datasets:
    full_dataset = concatenate_datasets(all_datasets)
    print(f"✅ Successfully concatenated {len(all_datasets)} tasks into a dataset with {len(full_dataset)} examples")
else:
    print("❌ No datasets were successfully loaded")
