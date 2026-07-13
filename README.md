# Dual-Branch Skin Lesion Classification

A deep learning pipeline that classifies skin lesions from paired clinical and dermoscopic photographs of the same lesion.

## Overview

Dermatologists routinely inspect a lesion with both the naked eye (a clinical photo) and a dermatoscope (a magnified, polarised dermoscopic photo), because each modality reveals different diagnostic cues. This project mirrors that workflow with a dual-branch convolutional network: two ResNet50 backbones process the two image types in parallel, and their feature maps are fused before a shared classifier assigns one of five diagnostic classes (MEL, NV, BCC, SK, MISC). The work was developed as an undergraduate thesis and includes a full ablation study isolating the contribution of the multimodal architecture and the training-data scale.

## Approach

- **Dual-branch fusion.** Two ImageNet-pretrained ResNet50 backbones each emit a 2048x7x7 feature map. The fusion head combines them (add / multiply / 1x1-conv concat), then applies BatchNorm, global average pooling, dropout, and two fully connected layers down to five logits.
- **Two paired datasets.** Derm7pt (1,011 cases) and MILK10k (4,361 usable cases after class mapping) are combined into 5,372 paired samples, both remapped onto a shared five-class schema.
- **Class-imbalance handling.** A WeightedRandomSampler plus inverse-frequency weighted cross-entropy counter the dominance of the nevus (NV) class.
- **Staged transfer learning.** Early epochs freeze the lower ResNet layers and train only `layer4` and the fusion head; later epochs unfreeze `layer3` at a halved learning rate.
- **Optuna hyperparameter search.** The final notebook treats every technique as an on/off toggle (fusion mode, loss type, label smoothing, MixUp, auxiliary head, SWA, cross-modal contrastive pretraining) and lets Optuna search per dataset, optimising validation macro-F1 with median pruning.
- **Multiple fusion experiments.** Additional notebooks benchmark add/multiply fusion, bidirectional cross-attention, and SE-ResNet variants against a baseline.

## Tech Stack

- Python, PyTorch, torchvision
- Optuna (hyperparameter search)
- scikit-learn (metrics), pandas, NumPy
- matplotlib, seaborn (visualisation)
- Runs on Apple Silicon (MPS), CUDA, or CPU (device auto-detected)

## Project Structure

```
thesis_final.ipynb       Final model — per-dataset Optuna search, full retrain, evaluation
model.ipynb              Dual-branch ResNet50 + cross-attention + Optuna
thesis.ipynb             Baseline pipeline: EDA -> training -> evaluation
DualBranch.py            Reference dual-branch architecture (RGB + Hermite noise)
thesis.py                Standalone ResNet-18 baseline (Colab paths, reference only)
setup_datasets.py        Downloads Derm7pt + MILK10k from Kaggle into dataset/
experimentation.md       Thesis chapter: datasets, architecture, results, ablation
requirements.txt         Python dependencies
*.csv / *.png            Committed metrics tables and comparison figures
```

## Getting Started

```bash
pip install -r requirements.txt
```

Requires Python 3.8+. Fetch the datasets (needs a Kaggle API token at `~/.kaggle/kaggle.json`):

```bash
python setup_datasets.py
```

Then open a notebook and run the cells sequentially:

```bash
jupyter notebook thesis_final.ipynb
```

In `thesis_final.ipynb`, set the `DATASET_DIR` switch to `Derm7pt` or `Milk10k` and run all cells; a `SMOKE_TEST` flag runs a fast 2-trial dry run to validate the pipeline before a full search.

## Results

Ablation study on the held-out test set (n = 806), reported in `experimentation.md` and `ablation_results.csv`:

| Configuration | Accuracy | Balanced Acc. | F1 (macro) | Cohen's Kappa | MCC |
|---|---|---|---|---|---|
| A - Dual branch, Derm7pt only | 0.7303 | 0.6842 | 0.6408 | 0.5778 | 0.5861 |
| B - Dual branch, combined | 0.7605 | 0.7360 | 0.6955 | 0.6637 | 0.6701 |
| C - Single branch (dermoscopic), combined | 0.6501 | 0.6394 | 0.5650 | 0.5296 | 0.5434 |
| D - Single branch (clinical), combined | 0.4603 | 0.5715 | 0.4472 | 0.3375 | 0.3831 |

The full dual-branch model (B) reaches a macro AUC of 0.9403 on the test set. Two findings stand out: expanding the training corpus with MILK10k (A to B) raises macro-F1 by ~5.5 points, and using both modalities beats the best single-modality model (B vs C) by ~13 points macro-F1, confirming that the clinical and dermoscopic views carry complementary information.
