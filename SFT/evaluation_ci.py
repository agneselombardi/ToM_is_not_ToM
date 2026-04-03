import pandas as pd
import os
import torch
import random
import torch.nn.functional as F

import minicons
from minicons import scorer
from torch.utils.data import DataLoader
import numpy as np

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device = {DEVICE}")

MODEL_PATH = "/extra/agnese.lombardi/ISA_ToM"
model_id = "ToM_fine_tuned"


def load_mt(model_name, cache_dir="/extra/agnese.lombardi/ISA_ToM"):
    loaded_model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = scorer.IncrementalLMScorer(loaded_model, tokenizer=tokenizer, cache_dir = cache_dir)
    return model

def define_prompt(f, entry, model_name, shuffled_choices):
    base_prompt = "Task: I'll give you"
    answer_prompt = "Answer with the number correspondent to the right option."
    number = "four" if f in ["GCI", "indArt", "Iprinc", "PCI", "spAct"] else "three"

    if f == "bridg":
        c = "sentence"
        d = "Each option is a possible following of the first sentence."
        e = "You must choose the most probable following."
        context = f"Sentence: {entry['Context utterance']}\n"
        question = "\nChoices:\n"
    else:
        c = "context utterance and a target sentence" if f not in ["indArt", "togImpl"] else "target sentence"
        d = "Then, a question about the target sentence."
        e = "You must choose the most probable answer."
        question = f"Question: {entry['Question']}\nChoices:\n"
        context = f"Context utterance: {entry['Context utterance']}\n" if f not in ["indArt", "togImpl"] else ""
        context += f"Target sentence: {entry['Target']}\n" if f != "bridg" else ""
    
    choices_text = "\n".join([f"{num}. {choice}" for num, choice in shuffled_choices])
        
    # if model_name in ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]:
    #     p = [{"role": "system", "content": f"You are a helpful assistant"},
    #          {"role": "user", "content": f"{base_prompt} a {c} and {number} options. {d} {e} {answer_prompt} {context}{question}\n{choices_text}\n The correct answer is:"}]
    # else:
    p = f"{base_prompt} a {c} and {number} options. {d} {e} {answer_prompt} {context}{question}\n{choices_text}\n The correct answer is:"   
    return p

def prompt_prob(f, entry, shuffled_choices):    
    if f == "bridg":
        context = f"{entry['Context utterance']}\n"
        question = ""
    else:
        question = f"{entry['Question']}"
        context = f"{entry['Context utterance']}\n" if f not in ["indArt", "togImpl"] else ""
        context += f"{entry['Target']}\n" if f != "bridg" else ""
    p = f"{context}{question}\n"   
    return p


def main():
    input_dir = "/home/agnese.lombardi/progetto/Pragmatics/Datasets"
    output_dir = "/home/agnese.lombardi/progetto/ToM_Prag_comparison/SFT/results/conversational_implicatures"
    os.makedirs(output_dir, exist_ok=True)

    folder = ["bridg", "coref", "GCI", "indArt", "Iprinc", "PCI", "spAct", "togImpl"]

    # 🔹 Evaluate ONLY ToM_fine_tuned
    model_path = os.path.join(MODEL_PATH, model_id)
    print(f"Loading model: {model_path}")
    model_mini = load_mt(model_name=model_path)

    for f in folder:
        print(f"Evaluating {f}...")

        data = pd.read_json(
            os.path.join(input_dir, f"Dataset_{f}/{f}_data.json")
        )

        data["Prompt"] = ""
        data["choices_prob"] = ""
        data["sentences_prob"] = ""
        data["choice_win"] = ""
        data["Generated Text"] = ""

        for index, entry in data.iterrows():

            choices = entry["Completions"] if f == "bridg" else entry["Choices"]

            numbered_choices = list(enumerate(choices, start=1))
            prompt = define_prompt(f, entry, model_path, numbered_choices)
            prefixes = [prompt] * len(choices)

            # P(choice number | prompt)
            r = [str(i) for i in range(1, len(choices) + 1)]
            choice_logits = torch.tensor(
                model_mini.conditional_score(prefixes, r)
            )
            choice_prob = F.softmax(choice_logits, dim=0).tolist()

            # P(sentence | context)
            sentence_prompt = prompt_prob(f, entry, choices)
            sentence_logits = torch.tensor(
                model_mini.conditional_score(
                    [sentence_prompt] * len(choices), choices
                )
            )
            sentence_prob = F.softmax(sentence_logits, dim=0).tolist()

            choice_prob_pairs = list(zip(choices, choice_prob))
            sentence_prob_pairs = list(zip(choices, sentence_prob))

            data.at[index, "Prompt"] = prompt
            data.at[index, "choices_prob"] = choice_prob_pairs
            data.at[index, "sentences_prob"] = sentence_prob_pairs
            data.at[index, "choice_win"] = max(choice_prob_pairs, key=lambda x: x[1])[0]
            data.at[index, "Generated Text"] = max(sentence_prob_pairs, key=lambda x: x[1])[0]

        output_file = os.path.join(
            output_dir, f"{model_id}_{f}_probabilities.csv"
        )
        data.to_csv(output_file, index=False)
        print(f"Saved {output_file}")

    # 🔹 Clean up
    del model_mini
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
