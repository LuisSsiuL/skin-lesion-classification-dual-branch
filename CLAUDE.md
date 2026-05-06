# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.8+. Training runs on Apple Silicon (MPS) or CUDA GPU.

## Running

Primary work happens in `thesis.ipynb` — run cells sequentially in Jupyter or VSCode.

```bash
jupyter notebook thesis.ipynb
```

`thesis.py` is a standalone ResNet-18 baseline with Google Colab paths hardcoded — not for local use.  
`DualBranch.py` is a reference implementation (RGB + Hermite noise dual ResNet50) — not used in main pipeline.

## Architecture

**Dual Branch ResNet50** — processes paired clinical + dermoscopic images of same lesion.

```
Clinical photo  → ResNet50_A → 2048×7×7 ─┐
                                           ├→ Concat(4096×7×7) → Conv1x1(1024) → BN→ReLU→AvgPool → FC(512) → FC(5)
Dermoscopic photo → ResNet50_B → 2048×7×7 ─┘
```

- Both ResNet50s pretrained ImageNet; layers 1-3 frozen for first 5 epochs, then layer3+layer4 unfrozen
- Trained with 5 independent runs; best val-loss checkpoint saved to `dual_branch_best.pth`

## Dataset

**Derm7pt only** — 1,011 cases, each with paired clinical + dermoscopic image.

| Path | Contents |
|------|----------|
| `dataset/Derm7pt/meta/meta.csv` | Main metadata (diagnosis, sex, location, elevation, image filenames) |
| `dataset/Derm7pt/meta/{train,valid,test}_indexes.csv` | Pre-defined splits (not used — notebook does stratified 70/15/15) |
| `dataset/Derm7pt/images/` | All images (clinic + derm JPGs) |

**5 classes** (after grouping 20 fine-grained diagnoses):

| Label | Diagnoses grouped |
|-------|-------------------|
| MEL | melanoma (all subtypes + metastasis) |
| NV | clark/reed/spitz/dermal/blue/congenital/combined/recurrent nevus |
| BCC | basal cell carcinoma |
| SK | seborrheic keratosis |
| MISC | lentigo, dermatofibroma, vascular lesion, melanosis, miscellaneous |

## Key Design Decisions

- **Class imbalance**: WeightedRandomSampler + weighted CrossEntropyLoss (inverse class frequency); NV dominates
- **Split**: stratified 70/15/15 (707 train / 152 val / 152 test) — Derm7pt pre-defined splits not used
- **5 runs**: model retrained 5× from scratch, best overall val-loss checkpoint used for evaluation
- **Saved artifacts**: `dual_branch_best.pth`, `training_history.pkl`, `evaluation_results.pkl`, `per_class_results.csv`
