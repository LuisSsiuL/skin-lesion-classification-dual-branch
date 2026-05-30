# Design — Improved 7-Model Singular Notebook

**Date:** 2026-05-30
**File to create:** `thesis_test_singular_improved.ipynb`
**Status:** approved in brainstorming, pending spec review

## Context

The thesis compares 7 dual-branch architectures on a single toggleable dataset
(`thesis_test_singular_derm7pt.ipynb`). The proposed/contribution model is the **Full
Model = Dual ResNet50 + SE channel attention + bidirectional cross-attention**. On
Derm7pt it currently *underperforms* the plain concat baseline (test F1-macro 0.557 vs
0.638) because the shared training loop is basic (Adam, weighted CE, cosine, early-stop on
val-loss) and 707 training images punish heavy attention.

A separate single-model Optuna notebook (`thesis_final.ipynb`) was tried and rejected: its
tuned config overfit at 60 epochs (test F1 0.617), and most of its search space (fusion /
loss / SE / cross-attn) *is* the 7-model comparison axis here, so it does not transfer.

This notebook rebuilds all 7 models on an upgraded training harness, with **one shared
hyperparameter search** (not per-model), so the architecture comparison stays fair while
every model trains better.

**Goal:** improve all 7 honestly; Full Model is the thesis focus but is *not* forced to
win. If it still trails after improvements, that is a legitimate "attention needs more data
than 707 images" finding.

## The 7 models (copied verbatim from the existing notebook)

| # | Model | Class | Role |
|---|-------|-------|------|
| 1 | Single-RGB | `SingleBranchRGBClassifier` | floor (clinic only) |
| 2 | Dual Concat | `DualBranchBaseline` | simple-fusion baseline |
| 3 | SE-ResNet Concat | `DualBranchSEResNet` | channel attention |
| 4 | Add-Fusion | `DualBranchElementwiseFusion('add')` | fusion variant |
| 5 | Mul-Fusion | `DualBranchElementwiseFusion('mul')` | fusion variant |
| 6 | BiCrossAttn | `DualBranchBiCrossAttn` | cross attention |
| 7 | **Full Model** | `DualBranchSECrossCombined` | **SE + cross-attn — main goal** |

All keep their existing `forward(clinic, derm) -> (logits, aux)` interface, so one trainer
drives all 7 unchanged (Single-RGB ignores `derm`).

Fusion type / SE / cross-attn are **model identity**, NOT search parameters — they are the
comparison axis.

## Training improvements (one shared `run_training_improved`, all 7)

Proven cheap wins ported from `thesis_improved.ipynb` (label-smoothing + SWA were the big
helpers; mixup/aux/contrastive-pretrain hurt and are dropped):

| Aspect | Old | New |
|--------|-----|-----|
| Optimizer | Adam | **AdamW** (decoupled weight decay) |
| Schedule | plain cosine | **3-epoch linear warmup → cosine**; rebuild at LR/4 on unfreeze@5 |
| Loss | weighted CE | weighted CE **+ label smoothing** (value from search) |
| Weight averaging | none | **SWA** from epoch 45 (0.75×60), SWALR=LR/8, dual-input BN recompute (`update_bn_dual`) |
| Checkpoint selection | best val-**loss** | best val-**F1-macro** snapshot; at end compare snapshot vs SWA-averaged on val, keep the higher |
| Early stop | val-loss, patience 20 | val-**F1**, patience 20, **paused once SWA phase starts** so averaging completes |

**Kept unchanged:** freeze→unfreeze@5, grad-clip 1.0, class-weighted loss +
`WeightedRandomSampler`, batch 64, 60 final epochs, dataset toggle, 7-metric eval, §6 grand
comparison.

**Dropped (proven losers / too slow):** paired MixUp, derm aux head, contrastive
pretrain-and-replace, per-model Optuna.

### Optional aux-contrastive toggle (OFF by default)

The *smart* contrastive form — joint loss `total = CE_smoothed + λ·NT_Xent(z_clinic,
z_derm)` with small projection heads, λ∈[0.1,0.3] — keeps ImageNet features instead of
overwriting them (the pretrain-replace A7 variant was the worst performer and is excluded).
Added as a toggle, **off by default**, available to flip on for the Full Model as a
side-experiment. Not part of the fair baseline run. When enabled it adds two small
projection heads to the Full Model only (the other 6 classes stay verbatim); when off, the
Full Model is also unchanged from the original.

## Hyperparameter search — ONE shared search (decision: option B)

Tune **once** on the Full Model (the contribution + heaviest/most-overfit model), then
**lock** the best HPs and train all 7 with them. Shared HPs → fair architecture comparison;
single search → bounded time.

- **Tuner:** Optuna, `TPESampler(seed=42, multivariate=True)` + `MedianPruner`.
- **Search space (shared training HPs only):** `lr` log[1e-4, 3e-3], `weight_decay`
  log[1e-5, 3e-3], `dropout` [0.2, 0.5], `label_smoothing` [0.0, 0.15].
- **Objective:** maximize val F1-macro (reported per epoch for pruning).
- **Warm-start trial 0** with the existing already-tuned config (lr 0.00075, dropout 0.32,
  wd 0.000957, label_smoothing 0.1) so search starts from a known-good point.
- **Budget:** ~15 trials, `SEARCH_EPOCHS ≈ 25` with pruning.
- **Resumable:** SQLite storage `optuna_singular_{DATASET_NAME}.db`, `load_if_exists=True`.
- **Lock:** write best HPs to `singular_bestparams_{DATASET_NAME}.json`; all 7 final
  trainings read this dict.

NOT searched (fixed): fusion (= model identity), SWA on, warmup on, AdamW, batch 64,
freeze@5, grad-clip, 60 final epochs, patience 20.

## Infrastructure wins (kept from thesis_final — cost nothing)

- CUDA **AMP** (autocast + GradScaler), no-op on MPS/CPU.
- Pinned memory + persistent multi-worker DataLoaders (workers enabled on Windows/CUDA, 0 on MPS/CPU).
- **7-metric** report: accuracy, balanced accuracy, F1-macro, F1-weighted, Cohen κ, MCC,
  log-loss.
- **Dataset-suffixed artifacts** (`_improved_{dataset}`) so the two machines and the old
  notebook never overwrite each other.

## Literature positioning & realistic targets

Derm7pt diagnosis (5-class — our exact task) is a hard, tiny (1,011-case), imbalanced
benchmark. Published results:

| Method | DIAG acc | Modalities |
|--------|----------|-----------|
| Kawahara 2019 (Inception comb) | 74.2% | clinical + dermoscopic |
| HcCNN | 69.9% | clinical + dermoscopic |
| AMFAM | 75.4% | clinical + dermoscopic + metadata |
| TFormer | 77.5% | clinical + dermoscopic + metadata |
| FusionM4Net | 77.6% | clinical + dermoscopic + metadata |
| SkinM2Former (SOTA) | 77.85% | clinical + dermoscopic + metadata |
| **Our Dual Concat (current)** | **73.0%** | clinical + dermoscopic |
| Our Full Model (current) | 66.5% | clinical + dermoscopic |

Takeaways:
- Our image-only concat baseline (73%) already matches Kawahara's image-only combined model
  (74.2%) and trails SOTA by only ~4–5 pts — and **all SOTA above also use patient
  metadata + multi-task 7-point supervision**, which we deliberately do not.
- Diagnosis tops out ~78%. Training improvements realistically buy **+1 to +4 pts**, not a
  dramatic jump. The thesis should frame results accordingly.
- Cross-attention works in SOTA, but with transformers + metadata + cross-validation + heavy
  regularization. Our cross-attn lagging on 707 images / a single split is the expected
  "attention is data-hungry" finding — defensible, not a bug.
- The "98.66%" / "0.99" figures in some papers are per-criterion binary averages over the
  7-point checklist, NOT 5-class diagnosis. Not comparable; do not cite as our target.

**Decisions (confirmed):** stay **image-only** (clinical + dermoscopic, no metadata) for a
clean "competitive without metadata" story; keep the existing **custom stratified 70/15/15
split** (seed 42) for consistency across our runs.

## Notebook structure (cells)

1. Setup: imports (+ optuna, AMP, SWA), seed, device.
2. Dataset selector toggle (`DATASET_DIR` → derm7pt / milk10k).
3. Data assembly + loaders (paired images, stratified 70/15/15, class weights,
   `drop_last=True`, workers/pin).
4. Backbone + all 7 model classes (copied verbatim).
5. Helpers: `warmup_cosine_lambda`, `val_f1_macro`, `update_bn_dual`, AMP scaler,
   `run_training_improved`, `evaluate` (7-metric), comparison/plot helpers.
6. **§Search:** Optuna study on Full Model → lock best HPs to JSON.
7. **§2–§5:** per-section train cells — each builds its model, calls
   `run_training_improved` with the locked HPs, saves `{stem}_improved_{dataset}_best.pth`.
8. **§6 Grand Comparison:** evaluate all 7 on held-out test; 7-metric table + bar chart;
   **old-vs-improved** delta table.

## Data flow

paired (clinic, derm) → model → (logits, _) → weighted+smoothed loss (optionally + λ·NT-Xent
if toggle on) → per-epoch val F1 tracked → best snapshot or SWA model saved → §6 evaluates
all 7 on test → CSV/JSON + figures.

## Verification

- `SMOKE` flag (subset frac + few epochs + ~2 trials) validates the whole chain in minutes,
  then full run.
- Confirm all 7 instantiate and `run_training_improved` completes 1 epoch each; verify
  `update_bn_dual` works across every forward signature including Single-RGB.
- Confirm the search produces a locked-HP JSON and the 7 trainings consume it.
- Deliverable: old-vs-improved 7-metric table per dataset.

## Time budget

Per dataset on CUDA: one ~15-trial search on the Full Model (~1–3h with pruning) + 7 final
trainings (~1–2h) ≈ **~3–5h**. No mixup / no contrastive-pretrain / no per-model search →
no blow-up. M5 is far slower (run derm7pt there at most; milk10k on the CUDA box).

## Out of scope (documented as thesis future work)

- **Patient-metadata branch** — the #1 lever in every Derm7pt SOTA method (~4 pts). Excluded
  now for a clean image-only comparison; document as the highest-value next step.
- **Official Derm7pt split + nested CV** — needed for direct comparability to the published
  table; we keep the custom 70/15/15 for now.
- **Multi-task 7-point criteria supervision** — auxiliary regularization used by SOTA.
- Per-model hyperparameter tuning (confounds the comparison).
- Contrastive pretrain-and-replace (proven worst, expensive).
- Architectural changes to the 7 models (kept verbatim for a clean before/after).
