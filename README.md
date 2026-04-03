# ToM is not ToM

This repository contains the code and results for experiments investigating whether Theory of Mind (ToM) and pragmatic competence (conversational implicatures) are distinct cognitive abilities in large language models.

## Structure

```
.
├── SFT/                  # Supervised Fine-Tuning experiments
│   ├── training.py       # SFT training script
│   ├── evaluation_pub.py # Evaluation on publication benchmark
│   ├── evaluation_ci.py  # Evaluation on conversational implicatures
│   ├── merge_model.ipynb # Model merging notebook
│   └── results/          # Evaluation outputs and model responses
│
├── DPO/                  # Direct Preference Optimization experiments
│   ├── training_prag.py  # DPO training on pragmatics data
│   ├── training_tom.py   # DPO training on ToM data
│   ├── evaluation_pub.py # Evaluation on publication benchmark
│   ├── evaluation_pub_tom.py # Evaluation on ToM benchmark
│   └── results/          # Evaluation outputs and model responses
│
└── analysis.ipynb        # Main analysis and figures
```

## Overview

We fine-tune language models separately on ToM and pragmatics tasks using both SFT and DPO, then evaluate cross-task transfer. The key question is whether training on ToM improves pragmatic inference and vice versa.

**Benchmarks used:**
- FanToM (ToM evaluation)
- Conversational implicatures (pragmatics evaluation)

## Results

See `analysis.ipynb` for the full analysis. Summary figures:
- `fantom_results.png` — model performance on FanToM
- `pub_results.png` — model performance on publication benchmark
