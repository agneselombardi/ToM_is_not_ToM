#from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datasets
import wandb
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
max_length = 2048
batch_size = 16
output_dir = "/extra/agnese.lombardi/ISA_ToM"
model_id = "meta-llama/Meta-Llama-3-8B"
new_model = "output"
mode="ToM"

# def clean_and_rename_dataset(dataset):
#     if "question" in dataset.column_names and "target" in dataset.column_names:
#         dataset = dataset.rename_columns({"question": "prompt", "target": "completion"})
#     dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["prompt", "completion"]])
#     return dataset

def clean_and_rename_dataset(dataset, prompt_col, completion_col):
    dataset = dataset.rename_columns({prompt_col: "prompt", completion_col: "completion"})
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["prompt", "completion"]])
    return dataset

def get_datasets():
    if mode == "ToM":
        dataset = load_dataset("facebook/ExploreToM", split="train")
        train_subset = dataset.select(range(6000))
        val_subset = dataset.select(range(6000, 11000))
        train_data = clean_and_rename_dataset(train_subset, prompt_col="story_structure", completion_col="expected_answer")
        val_dataset = clean_and_rename_dataset(val_subset, prompt_col="infilled_story", completion_col="expected_answer")
        print("Train columns:", train_data.column_names)
        print("Train sample:", train_data[0])
        print("Validation columns:", val_dataset.column_names)
        print("Validation sample:", val_dataset[0])
    else:  # mode == "pragmatics"
        datasets = [
            load_dataset("argilla/synthetic-concise-reasoning-sft", split="train"),
            load_dataset("lighteval/synthetic_reasoning_natural", "hard", split="train"),
        ]
        val_dataset = load_dataset("lighteval/synthetic_reasoning_natural", "hard", split="validation")
        train_data = [clean_and_rename_dataset(d, "question", "target") for d in datasets]
        val_dataset = clean_and_rename_dataset(val_dataset,  "question", "target")
        train_data = concatenate_datasets(train_data)
    print("Columns in dataset:", train_data.column_names)
    print("First sample:", train_data[0])
    return train_data, val_dataset

def formatting_func(example):
    if "prompt" in example and "completion" in example:
        text = f"{example['prompt']}{example['completion']}"
        return {"text": text}
    else:
        return {"text": ""}

model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    cache_dir= output_dir, 
    token=os.environ.get("HF_TOKEN"),
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map = "auto")

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir= output_dir, 
    token=os.environ.get("HF_TOKEN")
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

lora_config = LoraConfig(
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  
    lora_bias="none",  
    task_type=TaskType.CAUSAL_LM)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

train_data, val_data = get_datasets()
train_data = train_data.map(formatting_func)
val_data = val_data.map(formatting_func)

def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

train_data = train_data.map(
    preprocess_function,
    batched=True,
    remove_columns=train_data.column_names  
)
val_data = val_data.map(
    preprocess_function, 
    batched=True,
    remove_columns=val_data.column_names 
)

training_args = SFTConfig(
    output_dir=output_dir,
    report_to="wandb",
    run_name=f"llama3-{mode}-training",
    logging_steps=10,
    eval_steps=200,
    neftune_noise_alpha=5,
    save_steps=600,
    save_total_limit=3,
    eval_strategy="steps",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    remove_unused_columns=False,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,  
    greater_is_better=False,  
)

trainer = SFTTrainer(
    model,
    tokenizer=tokenizer,
    train_dataset= train_data,
    eval_dataset= val_data,
    args=training_args,
    #formatting_func=formatting_prompts_func,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()
trainer.save_model(output_dir + f"/{mode}_fine_tuned")
