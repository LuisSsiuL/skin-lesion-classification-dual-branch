# 4. Experimentation

## 4.1 Datasets

This study uses two publicly available paired-image dermatological datasets. Both provide a clinical (naked-eye) photograph and a dermoscopic photograph for each skin lesion case — a structural requirement of the dual-branch architecture.

### 4.1.1 Derm7pt

The Seven-Point Checklist Dermatology Dataset (Derm7pt) [2] consists of **1,011 cases**, each with one clinical and one dermoscopic image acquired under controlled conditions. The 20 original fine-grained diagnoses are consolidated into five diagnostic classes used throughout this study:

| Class | Constituent Diagnoses |
|---|---|
| **MEL** | Melanoma (all subtypes, in situ, metastasis) |
| **NV** | Clark, Reed/Spitz, dermal, blue, congenital, combined, recurrent nevus |
| **BCC** | Basal cell carcinoma |
| **SK** | Seborrheic keratosis |
| **MISC** | Lentigo, dermatofibroma, vascular lesion, melanosis, miscellaneous |

Derm7pt exhibits significant class imbalance — Melanocytic Nevus (NV) accounts for roughly 53% of cases while BCC makes up only ~9%.

### 4.1.2 MILK10k

The Multimodal Image Library for Keratinocytic lesions (MILK10k) is sourced from the ISIC Archive and provides **5,240 cases**, each with one clinical and one dermoscopic image stored as paired ISIC identifiers. The dataset's 11 original ISIC classes are mapped onto the same five-class schema used for Derm7pt, and six classes without a suitable mapping (AKIEC, SCCKA, INF, BEN_OTH, MAL_OTH, and one unlabelled group) are excluded:

| MILK10k Class | Mapped To |
|---|---|
| MEL | MEL |
| NV | NV |
| BCC | BCC |
| BKL (benign keratosis) | SK |
| DF (dermatofibroma) | MISC |
| VASC (vascular lesion) | MISC |

After filtering, **4,361 MILK10k cases** remain. Combined with all 1,011 Derm7pt cases, the full dataset contains **5,372 paired samples**.

### 4.1.3 Data Split and Preprocessing

A stratified 70/15/15 split (random seed 42) is applied at the combined dataset level to ensure proportional class representation across train, validation, and test partitions. The resulting split sizes are approximately 3,760 train / 806 validation / 806 test samples.

All images are resized to **224 × 224 pixels** and normalised to ImageNet mean and standard deviation (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Training images receive the following augmentations:

- `RandomResizedCrop(224, scale=(0.8, 1.0))`
- `RandomHorizontalFlip` and `RandomVerticalFlip`
- `RandomRotation(15°)`
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`

To counter class imbalance, **WeightedRandomSampler** is used at training time (sampling weight proportional to inverse class frequency), and **weighted CrossEntropyLoss** is applied with per-class weights computed from the training partition.

*Figure 1 — Clinical vs. dermoscopic sample pairs from Derm7pt illustrating the complementary visual information in each modality.*

> ![Fig1 Clinical vs Dermoscopic](figures/fig1_clinical_vs_derm.png)

---

## 4.2 Model Architecture

The proposed model is a **Dual-Branch ResNet50** that processes both image modalities simultaneously through two independent but structurally identical ResNet50 backbones, then fuses their feature maps before classification.

```
Clinical photo (224×224×3)  →  ResNet50_A  →  feat_clinic [B, 2048, 7, 7]  ─┐
                                                                               ├→ Concat → [B, 4096, 7, 7]
Dermoscopic photo (224×224×3) →  ResNet50_B  →  feat_derm  [B, 2048, 7, 7]  ─┘
         ↓
  1×1 Conv2d(4096 → 1024) → BN → ReLU
         ↓
  AdaptiveAvgPool2d(1×1)  →  [B, 1024]
         ↓
  Dropout(0.2) → Linear(1024 → 512) → ReLU → BN → Dropout(0.1)
         ↓
  Linear(512 → 5)  →  logits
```

Both ResNet50 backbones are initialised from **ImageNet pretrained weights**. The standard classification head of each backbone is replaced by an identity pass-through so that the spatial feature map at 2048×7×7 is preserved for concatenation.

*Figure 2 — Dual-Branch ResNet50 architecture diagram.*

> ![Fig2 Architecture](figures/fig2_dual_branch_architecture.png)

### 4.2.1 Transfer Learning and Gradual Unfreezing

To stabilise early training on a relatively small dataset, a two-phase unfreezing schedule is applied:

| Phase | Epochs | Frozen Layers | Learning Rate |
|---|---|---|---|
| 1 | 1 – 5 | `conv1`, `bn1`, `layer1`, `layer2`, `layer3` | 1 × 10⁻⁴ |
| 2 | 6 – 30 | `conv1`, `bn1`, `layer1`, `layer2` | 5 × 10⁻⁵ |

Only `layer4` and the fusion head are trained in Phase 1; `layer3` is unfrozen together with `layer4` from epoch 6 onward with the learning rate halved.

---

## 4.3 Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Initial LR | 1 × 10⁻⁴ |
| LR Scheduler | StepLR (step=10, γ=0.5) |
| Batch size | 32 |
| Epochs per run | 30 |
| Independent runs | 5 |
| Checkpoint criterion | Minimum validation loss |
| Dropout | 0.2 (fusion) / 0.1 (pre-classifier) |

Five independent runs are executed with the same hyperparameters but different random initialisations of the fusion head and classifier. The checkpoint with the lowest validation loss across all runs is retained as the final model (`dual_branch_best.pth`). Evaluation is performed once on the held-out test set using this checkpoint.

---

## 4.4 Main Results

The best checkpoint (Config B — dual-branch, combined dataset) achieves the following on the **test set (n = 806)**:

| Metric | Score |
|---|---|
| Accuracy | **0.7605** |
| Balanced Accuracy | **0.7360** |
| F1-score (macro) | **0.6955** |
| Cohen's Kappa | **0.6637** |
| MCC | **0.6701** |
| Macro AUC | **0.9403** |

These results represent a substantial improvement over the preliminary Derm7pt-only baseline (72.37% validation accuracy reported at mid-term) attributable to the expanded training corpus from MILK10k.

*Figure 3 — Training and validation loss/accuracy curves for the best run.*

> ![Fig3 Training Curves](figures/fig3_training_curves.png)

*Figure 4 — Confusion matrix (counts and normalised) on the test set.*

> ![Fig4 Confusion Matrix](figures/fig4_confusion_matrix.png)

*Figure 5 — Macro-average ROC curves and Precision-Recall curves per class.*

> ![Fig5 ROC PR Curves](figures/fig5_roc_pr_curves.png)

### 4.4.1 Per-Class Results

| Class | Precision | Recall | Specificity | F1 | AUC | AP | Support |
|---|---|---|---|---|---|---|---|
| BCC | 0.9438 | 0.7865 | 0.9573 | 0.858 | 0.9663 | 0.9569 | 384 |
| MEL | 0.6263 | 0.5849 | 0.9471 | 0.605 | 0.8940 | 0.6506 | 106 |
| MISC | 0.5676 | 0.7241 | 0.9794 | 0.636 | 0.9735 | 0.7327 | 29 |
| NV | 0.8020 | 0.7980 | 0.9359 | 0.800 | 0.9444 | 0.8513 | 198 |
| SK | 0.4575 | 0.7865 | 0.8842 | 0.579 | 0.9235 | 0.7044 | 89 |
| **Macro Avg** | **0.6794** | **0.7360** | **0.9408** | **0.6955** | **0.9403** | **0.7792** | **806** |

BCC achieves the highest F1 (0.858), benefiting from its visually distinctive morphological features visible in both modalities. Melanoma (MEL) and Seborrheic Keratosis (SK) remain the hardest classes, as both can appear morphologically similar to NV under clinical inspection. The high AUC across all classes (≥ 0.894) indicates strong discriminative capability even where hard classification boundaries exist.

*Figure 6 — Per-class F1 bar chart comparing all ablation configurations.*

> ![Fig6 Per-class F1](figures/fig6_perclass_f1.png)

---

## 4.5 Ablation Study

To isolate the contribution of (a) the dual-branch multimodal architecture and (b) the expanded combined dataset, four configurations are evaluated under controlled conditions (3 independent runs × 20 epochs each, same stratified split):

| Config | Architecture | Training Data | Rationale |
|---|---|---|---|
| **A** | Dual-branch | Derm7pt only (1,011 cases) | Architecture effect without extra data |
| **B** | Dual-branch | Combined (5,372 cases) | Full proposed system |
| **C** | Single-branch (derm only) | Combined (5,372 cases) | Value of clinical modality |
| **D** | Single-branch (clinic only) | Combined (5,372 cases) | Value of dermoscopic modality |

Configs C and D use a single ResNet50 with the same fusion head and classifier but receive only one image modality per sample.

### 4.5.1 Ablation Results

| Config | Accuracy | Balanced Acc | F1 (macro) | Kappa | MCC |
|---|---|---|---|---|---|
| A — Dual / Derm7pt only | 0.7303 | 0.6842 | 0.6408 | 0.5778 | 0.5861 |
| **B — Dual / Combined** | **0.7605** | **0.7360** | **0.6955** | **0.6637** | **0.6701** |
| C — Single-derm / Combined | 0.6501 | 0.6394 | 0.5650 | 0.5296 | 0.5434 |
| D — Single-clinic / Combined | 0.4603 | 0.5715 | 0.4472 | 0.3375 | 0.3831 |

**Dataset scale effect (A → B):** Expanding training data from 1,011 to 5,372 cases raises macro-F1 by +5.5 pp and Kappa by +8.6 pp, confirming the value of integrating MILK10k despite its different acquisition protocol.

**Modality contribution (B vs C/D):** Removing either modality from the combined dataset degrades performance substantially. The dermoscopy-only model (C) outperforms the clinical-only model (D) by a wide margin (+11.8 pp F1), consistent with dermoscopy's richer subsurface pigment information. Critically, the full dual-branch model (B) outperforms even the better single-modality baseline (C) by +13 pp F1, confirming that the two modalities provide complementary information that the fusion head learns to exploit jointly.

*Figure 7 — Bar chart comparing accuracy, balanced accuracy, and macro-F1 across ablation configurations A–D.*

> ![Fig7 Ablation Bar Chart](figures/fig7_ablation_comparison.png)

---

## 4.6 Discussion

The ablation results provide two clear findings. First, paired multimodal input is the strongest driver of performance: the clinical branch, while weaker alone, contributes uniquely to the fusion representation — its absence drops F1 by 13 pp relative to the full system. Second, data scale matters: even with the distribution shift between Derm7pt (hospital dermatoscope) and MILK10k (ISIC crowdsourced), the additional 4,361 cases meaningfully improve generalisation, particularly for minority classes with limited Derm7pt representation.

The per-class analysis reveals that class-level difficulty is correlated with visual inter-class similarity rather than support size alone. MISC (n=29) achieves respectable recall (0.724) thanks to its morphologically distinct member conditions, while SK (n=89) remains poorly discriminated because seborrheic keratosis overlaps visually with NV in both modalities at 224 px resolution.

The high macro AUC (0.9403) across all five classes indicates the model's probability estimates are well-calibrated for ranking purposes, making it a viable first-pass screening tool even where hard classification confidence is insufficient.
