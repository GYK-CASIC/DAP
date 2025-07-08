# DAP
Official resources of "DAP: Enhancing AI-generated Text Detection via Dynamic Adversarial Paraphrasing"

> **DAP** is a co‑evolving framework that alternately improves a *detector* and a *generator* so that the detector becomes robust against dynamically generated adversarial paraphrasings.

---

## 1  Environment
```bash
pip install -r requirements.txt     # install all dependencies (tested with Python 3.11 & CUDA 12.0)
```

---

## 2  Datasets
Our datasets are in the following directories:
```
dataset/tweepfake   # TweepFake
dataset/roc         # ROCStories
```
Each folder should contain:
```
train_warmup.json     # 10% subset to warm‑up the detector
train_rl.json         # 90% subset for adversarial training
test.json             
```

---

## 3  Pipeline (one dataset at a time)

### 3.1  Detector warm‑up
```bash
python 01_train_original.py                       # trains on train_warmup.json
```

### 3.2  Adversarial round *t* (repeat as needed)
1. **Generate paraphrased AI-generated text + Fine-grained Scoring**
   ```bash
   python scripts/00_generation_and_scoring.py
   ```
2. **Fine‑tune detector** on original and selected paraphrases
   ```bash
   python scripts/01_train_original.py            # original AI-generated text
   python scripts/02_train_paraphrase.py          # paraphrased AI-generated text
   ```
3. **DPO training for generator** (LoRA)
   ```bash
   cd scripts/03_LLaMA-Factory
   llamafactory-cli train training_args.yaml
   cd -
   ```

### 3.3  Evaluation
```bash
python scripts/04_inference.py                    # reports AUROC / Accuracy (FPR 1%) on test.json
```

---

## 4  Acknowledgments and Citations
This project borrows or uses code from the following project, for which we are grateful:

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - Implements Direct Preference Optimization (DPO).

---

## 5  Citation
```bibtex
  @article    {dap2025,
  title     = {DAP: Enhancing AI-Generated Text Detection via Dynamic Adversarial Paraphrasing},
  author    = {Guo et al.},
  year      = {2025}
}
```

Licensed under the Apache 2.0 license.
