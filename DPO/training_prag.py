import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset, concatenate_datasets

base_model_id = "meta-llama/Meta-Llama-3-8B"
token=""
policy_adapter = "/extra/agnese.lombardi/ISA_ToM/pragmatics_fine_tuned"


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


def prep_casual(ds):
    return ds.map(
        lambda x: {
            "prompt": x["prompt"],
            "chosen": x["chosen"],
            "rejected": x["rejected"],
        }
    ).remove_columns(
        [c for c in ds.column_names if c not in ["prompt", "chosen", "rejected"]]
    )

def prep_thinker(ds):
    return ds.map(
        lambda x: {
            "prompt": x["user"],
            "chosen": x["chosen"],
            "rejected": x["rejected"],
        }
    ).remove_columns(
        [c for c in ds.column_names if c not in ["prompt", "chosen", "rejected"]]
    )

def flatten_dialogue(dialogue):
    return "\n".join(turn["content"] for turn in dialogue)

def prep_clembench(ds):
    return ds.map(
        lambda x: {
            "prompt": x["prompt"],
            "chosen": flatten_dialogue(x["chosen"]),
            "rejected": flatten_dialogue(x["rejected"]),
        }
    ).remove_columns(
        [c for c in ds.column_names if c not in ["prompt", "chosen", "rejected"]]
    )

casual = prep_casual(
    load_dataset("flammenai/casual-conversation-DPO", split="train")
)

thinker = prep_thinker(
    load_dataset("minchyeom/Thinker-DPO", split="train")
)

clem = prep_clembench(
    load_dataset("clembench-playpen/DPO_dialogue_1neg_old", split="train")
)

prag_dpo = concatenate_datasets([casual, thinker, clem])
prag_dpo = prag_dpo.shuffle(seed=42)

train_size = int(0.9 * len(prag_dpo))
prag_dpo_train = prag_dpo.select(range(train_size))
prag_dpo_eval = prag_dpo.select(range(train_size, len(prag_dpo)))

print(f"Training samples: {len(prag_dpo_train)}")
print(f"Validation samples: {len(prag_dpo_eval)}")

dpo_args = DPOConfig(
    output_dir="/extra/agnese.lombardi/ISA_ToM/pragmatics_dpo",
    per_device_train_batch_size=1,
     per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    learning_rate=5e-6,
    num_train_epochs=3,
    bf16=True,
    fp16=False,
    save_steps=100,
    eval_steps=100,
    eval_strategy="steps",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    beta=0.3,  
    max_prompt_length=1024,  
    max_length=2048, 
    eval_accumulation_steps=4
)

policy_model.gradient_checkpointing_enable()
policy_model.config.use_cache = False

trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=dpo_args,
    train_dataset=prag_dpo_train,
    eval_dataset=prag_dpo_eval,
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
    print(f"Mean logit difference: {torch.mean(torch.abs(p_logits - r_logits))}")


trainer.train()
trainer.save_model()
