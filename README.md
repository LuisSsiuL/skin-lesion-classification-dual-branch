# Skin Lesion Classification using Dual Branch ResNet50

Deep learning pipeline for multi-class skin lesion classification using paired clinical + dermoscopic images from Derm7pt.

## Project Structure

```
thesis_regi/
├── model.ipynb           # Primary notebook — Dual-Branch ResNet50 + Cross-Attention + Optuna
├── thesis.ipynb          # Baseline notebook — full pipeline (EDA → training → evaluation)
├── thesis.py             # Standalone ResNet-18 baseline (Colab paths, not for local use)
├── DualBranch.py         # Reference dual-branch architecture (RGB + Hermite noise)
├── requirements.txt
└── dataset/
    └── Derm7pt/
        ├── meta/meta.csv
        ├── meta/{train,valid,test}_indexes.csv
        └── images/       # All clinical + dermoscopic JPGs
```

## Dataset

**Derm7pt** — 1,011 cases, each with a paired clinical (smartphone) photo + dermoscopic photo.

20 fine-grained diagnoses grouped into 5 classes:

| Class | Diagnoses |
|-------|-----------|
| MEL | melanoma (all subtypes + metastasis) |
| NV | clark/reed/spitz/dermal/blue/congenital/combined/recurrent nevus |
| BCC | basal cell carcinoma |
| SK | seborrheic keratosis |
| MISC | lentigo, dermatofibroma, vascular lesion, melanosis, miscellaneous |

Source: http://derm.cs.sfu.ca

## Model Architecture

**Dual Branch ResNet50** — two branches process different image modalities of the same lesion.

```
Clinical photo    → ResNet50_A → 2048×7×7 ─┐
                                             ├→ Concat(4096×7×7) → 1×1 Conv(1024) → BN→ReLU→AvgPool → Dropout → FC(512) → FC(5)
Dermoscopic photo → ResNet50_B → 2048×7×7 ─┘
```

Both ResNet50s pretrained on ImageNet. Transfer learning strategy:
- **Epochs 1-5**: Layers 1-3 frozen, only layer4 trainable
- **Epoch 6+**: Layer3 + layer4 unfrozen with halved LR

## Techniques

- **Class imbalance**: WeightedRandomSampler + weighted CrossEntropyLoss (inverse class frequency)
- **Stratified split**: 70/15/15 (707 train / 152 val / 152 test)
- **5 independent runs**: best val-loss checkpoint saved as `dual_branch_best.pth`
- **Data augmentation**: RandomResizedCrop, flips, rotation, ColorJitter
- **Optimizer**: Adam + StepLR (step=10, gamma=0.5)

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.8+. Runs on **MPS** (Apple Silicon), **CUDA** (NVIDIA GPU), or CPU — device is auto-detected.

- **CUDA**: enables `cudnn.benchmark`, multi-worker DataLoaders (`pin_memory=True`)
- **MPS**: single-process DataLoaders (PyTorch MPS constraint)

Download Derm7pt from http://derm.cs.sfu.ca and extract to `dataset/Derm7pt/`.

## Running

**Primary notebook — `model.ipynb`** (recommended):
- Dual-Branch ResNet50 + Unidirectional Cross-Attention fusion
- Optuna hyperparameter search (20 trials)
- Full evaluation + attention map visualizations

Open and run cells sequentially.

**Baseline — `thesis.ipynb`**:

1. **Cells 1-10**: EDA — load Derm7pt, clean, visualize, group diagnoses into 5 classes
2. **Cells 11-15**: Model setup — dataset class, splits, dataloaders, model definition
3. **Cell 16**: Training loop (5 runs, ~10-25 min on M4 Pro)
4. **Cells 17-22**: Evaluation — metrics, confusion matrix, ROC/PR curves, sample predictions

## Results

Test set (152 samples):

| Metric | Score |
|--------|-------|
| Accuracy | 0.5789 |
| Balanced Accuracy | 0.5704 |
| F1 (macro) | 0.5064 |
| Cohen's Kappa | 0.3946 |
| Macro AUC | 0.8644 |

Per-class results saved to `per_class_results.csv`.
