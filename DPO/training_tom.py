import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset, concatenate_datasets



base_model_id = "meta-llama/Meta-Llama-3-8B"
token=""
policy_adapter = "/extra/agnese.lombardi/ISA_ToM/ToM_fine_tuned"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir="/extra/agnese.lombardi/ISA_ToM",
local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def load_model(adapter_path):
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        cache_dir="/extra/agnese.lombardi/ISA_ToM",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True
    )
    return PeftModel.from_pretrained(base, adapter_path, is_trainable=True)

policy_model = load_model(policy_adapter)
ref_model = load_model(policy_adapter)
for p in ref_model.parameters():
    p.requires_grad = False
ref_model.eval()

policy_model.train()
policy_model.enable_input_require_grads()
policy_model.print_trainable_parameters()


def normalize_ds1(x):
    return {
        "prompt": x["prompt"],
        "chosen": x["chosen"],
        "rejected": x["rejected"],
    }


def normalize_ds3(x):
    def extract(messages, role):
        for m in messages:
            if m.get("role") == role:
                return m.get("content", "")
        return ""

    return {
        "prompt": extract(x["chosen"], "user"),
        "chosen": extract(x["chosen"], "assistant"),
        "rejected": extract(x["rejected"], "assistant"),
    }



tom1 = load_dataset("Doctor-Shotgun/theory-of-mind-dpo", split="train")
tom2 = load_dataset("onyrotssih/social-i-qa-orpo-dpo-10k", split="train")
tom3 = load_dataset("shayanfirouzian/SocialReasoning_DPO", split="train")
tom1 = tom1.map(normalize_ds1, remove_columns=tom1.column_names)
tom2 = tom2.map(normalize_ds3, remove_columns=tom2.column_names)
tom3 = tom3.map(normalize_ds3, remove_columns=tom3.column_names)



tom_dpo_full = concatenate_datasets([tom1, tom2, tom3])
tom_dpo_full = tom_dpo_full.shuffle(seed=42)
train_size = int(0.9 * len(tom_dpo_full))
tom_dpo_train = tom_dpo_full.select(range(train_size))
tom_dpo_eval = tom_dpo_full.select(range(train_size, len(tom_dpo_full)))

print(f"Training samples: {len(tom_dpo_train)}")
print(f"Validation samples: {len(tom_dpo_eval)}")


dpo_args = DPOConfig(
    output_dir="/extra/agnese.lombardi/ISA_ToM/ToM_dpo",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=5e-6,
    num_train_epochs=3,
    bf16=True,
    fp16=False,
    logging_steps=10,
    save_steps=100,  
    eval_steps=100,
    eval_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    beta=0.3,
    max_prompt_length=512,
    max_length=1280,
    eval_accumulation_steps=4
)

policy_model.gradient_checkpointing_enable()
policy_model.config.use_cache = False


trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=dpo_args,
    train_dataset=tom_dpo_train,
    eval_dataset=tom_dpo_eval,    
    tokenizer=tokenizer,
)


# 1. Policy has grads
assert any(p.requires_grad for p in policy_model.parameters())

# 2. Reference has NO grads
assert not any(p.requires_grad for p in ref_model.parameters())

# 3. Policy and reference differ
with torch.no_grad():
    ids = tokenizer("Hello", return_tensors="pt").to(policy_model.device)
    p_logits = policy_model(**ids).logits
    r_logits = ref_model(**ids).logits
    print(torch.mean(torch.abs(p_logits - r_logits)))


trainer.train()
trainer.save_model()