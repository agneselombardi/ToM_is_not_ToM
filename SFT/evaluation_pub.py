import os
import csv
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json
from tqdm import tqdm
import sklearn.metrics
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pandas as pd
import zipfile
import io

MODEL_PATH = "/extra/agnese.lombardi/ISA_ToM"
model_id = "ToM_fine_tuned"
TASKS = [str(i) for i in range(1, 15)]
RESULTS_CSV_PATH = "results/pub_results_tom.csv"

LOCAL_MODEL_PATH = os.path.join(MODEL_PATH, model_id)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

def load_task_manually(task_id):
    """
    Load a specific task's data directly from the cached zip file
    """
    try:
        # Get the path to the cached dataset file
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/datasets--cfilt--PUB")
        
        # Find the snapshot directory
        snapshot_dir = None
        for root, dirs, files in os.walk(cache_dir):
            if f"task_{task_id}.zip" in files:
                zip_path = os.path.join(root, f"task_{task_id}.zip")
                snapshot_dir = root
                break
        
        if not zip_path or not os.path.exists(zip_path):
            print(f"Could not find task_{task_id}.zip in cache")
            return None
            
        print(f"Found zip file for task {task_id}: {zip_path}")
        
        # Load and process the data manually
        data = []
        with zipfile.ZipFile(zip_path, 'r') as z:
            jsonl_filename = f"task_{task_id}.jsonl"
            if jsonl_filename not in z.namelist():
                print(f"Could not find {jsonl_filename} in the zip file")
                return None
                
            with z.open(jsonl_filename) as f:
                for i, line in enumerate(f):
                    try:
                        # Skip the known problematic line in task 14
                        if task_id == "14" and i == 259:  # Row 260 (0-indexed)
                            print(f"Skipping known problematic line 260 in task 14")
                            continue
                            
                        item = json.loads(line.decode('utf-8'))
                        
                        # Convert any non-string "correct answer" to string
                        if "correct answer" in item and not isinstance(item["correct answer"], str):
                            item["correct answer"] = str(item["correct answer"])
                            
                        data.append(item)
                    except Exception as e:
                        print(f"Skipping corrupted line {i+1} in task {task_id}: {str(e)}")
        
        if not data:
            print(f"No valid data found for task {task_id}")
            return None
            
        # Create a Hugging Face dataset
        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        print(f"Successfully loaded task {task_id} with {len(dataset)} examples")
        return dataset
        
    except Exception as e:
        print(f"Error loading task {task_id}: {str(e)}")
        return None

def evaluate_pub():
    results = {}
    
    for task_id in TASKS:
        try:
            print(f"Loading task {task_id}...")
            dataset = load_task_manually(task_id)
            
            if dataset is None or len(dataset) == 0:
                print(f"No data available for task {task_id}")
                results[f"PUB Task {task_id}"] = {"accuracy": 0.0, "recall": 0.0, "f1": 0.0}
                continue
            
            y_true = []
            y_pred = []
            
            for i, example in enumerate(tqdm(dataset, desc=f"Evaluating Task {task_id}")):
                try:
                    # Validate example structure to ensure it's not corrupted
                    if not all(k in example for k in ["pretext", "options", "correct answer"]):
                        print(f"Skipping corrupted example {i}: Missing required fields")
                        continue
                    
                    # Ensure all required fields have valid values
                    question = example["pretext"]
                    options = example["options"]
                    gold_answer = example["correct answer"]
                    
                    # Make sure gold_answer is a string
                    if not isinstance(gold_answer, str):
                        gold_answer = str(gold_answer)
                    
                    if not question or not options or not gold_answer:
                        print(f"Skipping corrupted example {i}: Empty required fields")
                        continue
                        
                    # Verify options is a list
                    if not isinstance(options, list) or len(options) == 0:
                        print(f"Skipping corrupted example {i}: Invalid options format")
                        continue
                    
                    prompt = f"{question}\nOptions:\n" + "\n".join(f"- {opt}" for opt in options) + "\nAnswer:"
                    
                    # Generate response with error handling
                    try:
                        response = generator(prompt, max_new_tokens=10, do_sample=False, return_full_text=False)[0]["generated_text"]
                    except Exception as e:
                        print(f"Error generating response for example {i}: {str(e)}")
                        continue
                        
                    # Match model output to one of the options
                    matched_option = next((opt for opt in options if opt.lower() in response.lower()), None)
                    predicted = matched_option if matched_option else "unknown"
                    
                    y_true.append(gold_answer.strip().lower())
                    y_pred.append(predicted.strip().lower())
                    
                except Exception as e:
                    print(f"Error processing example {i} for task {task_id}: {str(e)}")
                    continue
            
            # Check if we have any valid predictions
            if not y_true or not y_pred:
                print(f"No valid predictions for task {task_id}")
                results[f"PUB Task {task_id}"] = {"accuracy": 0.0, "recall": 0.0, "f1": 0.0}
                continue
                
            # Filter unknowns (optional)
            filtered = [(t, p) for t, p in zip(y_true, y_pred) if p != "unknown"]
            if not filtered:
                results[f"PUB Task {task_id}"] = {"accuracy": 0.0, "recall": 0.0, "f1": 0.0}
                continue
            
            y_true_filtered, y_pred_filtered = zip(*filtered)
            
            try:
                accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
                recall = recall_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)
                f1 = f1_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)
                
                results[f"PUB Task {task_id}"] = {
                    "accuracy": round(accuracy, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                }
            except Exception as e:
                print(f"Error calculating metrics for task {task_id}: {str(e)}")
                results[f"PUB Task {task_id}"] = {"accuracy": 0.0, "recall": 0.0, "f1": 0.0}
        
        except Exception as e:
            print(f"Error processing task {task_id}: {str(e)}")
            results[f"PUB Task {task_id}"] = {"accuracy": 0.0, "recall": 0.0, "f1": 0.0}
    
    return results

def save_results_to_csv(results_dict, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Task", "Accuracy", "Recall", "F1"])
        for task, metrics in results_dict.items():
            writer.writerow([task, metrics["accuracy"], metrics["recall"], metrics["f1"]])
    print(f"✅ Results saved to {filename}")

if __name__ == "__main__":
    print("🔍 Evaluating PUB benchmark with full metrics...\n")
    pub_results = evaluate_pub()
    save_results_to_csv(pub_results, RESULTS_CSV_PATH)