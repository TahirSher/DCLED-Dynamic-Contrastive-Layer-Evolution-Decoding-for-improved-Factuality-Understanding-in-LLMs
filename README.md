# DCSLED: Dynamic Contrastive Layer-wise Evolution Decoding

**Robust decoding framework for large language models**  
Enhances truthfulness, reasoning, and long-context performance through adaptive layer-wise signal evolution, contrastive guidance, and entropy-aware blending.

Current version: **1.0.0** (January 2026)

## Key Features

- **DCLED** — main decoding method combining SLED-style logit evolution with dynamic layer signals and adaptive contrastive DoLa
- Automatic hyperparameter adaptation based on **model size** (1B → 14B+) and **dataset type**
- Special tuning for **TruthfulQA**, **SEAL-QA** (longseal / seal_0 / seal_hard), **HotpotQA**
- Numerically stable operations (clamped probabilities, safe log-softmax, relative-top filtering)
- Supports ablation studies between:
  - VanillaGreedy
  - DoLa
  - SLED
  - DCLED

## Quick Start

### 1. Clone repository

```bash
git clone https://github.com/TahirSher
DCLED-Dynamic-Contrastive-Layer-Evolution-Decoding-for-improved-Factuality-Understanding-in-LLMs.git
cd DCLED-FRAMEWORK

### 2.Install dependencies
pip install -r requirements.txt
