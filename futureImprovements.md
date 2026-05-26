# Future Improvements

Potential improvements to explore without modifying core architecture or dataset.

---

## 1. Focal Loss

Replace weighted cross-entropy with Focal Loss. Adds a modulating factor `(1 - p_t)^gamma` that down-weights easy, well-classified examples and forces training focus on hard minority samples (BCC=29, SK=32).

- Keep existing class weights as `alpha`
- Start with `gamma=2.0`
- No architecture change — drop-in loss replacement

**Reference:** [Deep CNN-Transformer with Focal Loss, PMC 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9818899/)

---

## 2. MixUp Augmentation

Blend two training images and interpolate their labels during training:
```
blended_image = λ * img_A + (1 - λ) * img_B
blended_label = λ * label_A + (1 - λ) * label_B
```
Prevents the model from memorizing pixel patterns. Improves minority class generalization and reduces overfitting on small datasets like Derm7pt (~1k samples).

- Apply with probability 0.3–0.5
- `lambda ~ Beta(0.4, 0.4)`
- Built into PyTorch torchvision v0.15+

**Reference:** [LAMA: Lesion-Aware Mixup Augmentation, Springer 2024](https://link.springer.com/article/10.1007/s10278-024-01000-5)

---

## 3. Dermoscopy-Biased Auxiliary Loss

Add a separate auxiliary loss head on the dermoscopy branch alone, alongside the main fused output loss:
```
total_loss = main_loss(fused_output) + λ * aux_loss(derm_branch_output)
```
Forces the dermoscopy branch to become a stronger solo classifier, contributing better features to fusion. Clinical branch stays as supporting context. Tested directly on Derm7pt — reported ~77.85% balanced accuracy vs our current ~67%.

- No backbone change
- ~30 extra lines in training loop
- Start with `λ = 0.3`

**Reference:** [Single-Shared Network with Prior-Inspired Loss, arxiv 2024](https://arxiv.org/pdf/2403.19203)
