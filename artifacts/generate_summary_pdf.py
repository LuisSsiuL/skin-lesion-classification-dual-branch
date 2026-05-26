from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

OUTPUT = "model_summary.pdf"

doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
)
W = A4[0] - 4*cm

DARK_BLUE  = colors.HexColor("#1a3a5c")
MID_BLUE   = colors.HexColor("#2563a8")
GRAY_BG    = colors.HexColor("#f3f4f6")
GRAY_LINE  = colors.HexColor("#d1d5db")
WHITE      = colors.white
BLACK      = colors.black
ACCENT     = colors.HexColor("#e65100")

C1 = colors.HexColor("#1f77b4")  # Baseline
C2 = colors.HexColor("#ff7f0e")  # CrossAttn
C3 = colors.HexColor("#2ca02c")  # ChannelAttn
C4 = colors.HexColor("#d62728")  # Cross+ChannelAttn

title_style    = ParagraphStyle("T", fontSize=17, fontName="Helvetica-Bold",
                    textColor=WHITE, alignment=TA_CENTER, spaceAfter=3)
subtitle_style = ParagraphStyle("S", fontSize=9, fontName="Helvetica",
                    textColor=colors.HexColor("#cbd5e1"), alignment=TA_CENTER)
section_style  = ParagraphStyle("Sec", fontSize=10.5, fontName="Helvetica-Bold",
                    textColor=DARK_BLUE, spaceBefore=12, spaceAfter=5)
body_style     = ParagraphStyle("B", fontSize=9, fontName="Helvetica",
                    textColor=BLACK, leading=14, spaceAfter=4)
mono_style     = ParagraphStyle("M", fontSize=7.8, fontName="Courier",
                    textColor=colors.HexColor("#1e293b"), leading=12,
                    backColor=GRAY_BG, leftIndent=8, rightIndent=8,
                    spaceBefore=2, spaceAfter=4)
caption_style  = ParagraphStyle("Cap", fontSize=7.5, fontName="Helvetica-Oblique",
                    textColor=colors.HexColor("#6b7280"), alignment=TA_CENTER)
note_style     = ParagraphStyle("N", fontSize=8.5, fontName="Helvetica",
                    textColor=colors.HexColor("#374151"), leading=13,
                    leftIndent=8, spaceAfter=3)

def section(title):
    return [
        Spacer(1, 4),
        HRFlowable(width="100%", thickness=1.5, color=MID_BLUE, spaceAfter=3),
        Paragraph(title, section_style),
    ]

def tbl(data, col_widths, header=True, row_colors=None):
    style = [
        ("FONTNAME",       (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8.5),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 8),
        ("GRID",           (0, 0), (-1, -1), 0.4, GRAY_LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [GRAY_BG, WHITE]),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
            ("TEXTCOLOR",  (0, 0), (-1, 0), WHITE),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ]
    if row_colors:
        for row_idx, color in row_colors.items():
            style.append(("BACKGROUND", (0, row_idx), (0, row_idx), color))
            style.append(("TEXTCOLOR",  (0, row_idx), (0, row_idx), WHITE))
            style.append(("FONTNAME",   (0, row_idx), (0, row_idx), "Helvetica-Bold"))
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle(style))
    return t

# ── Title block ───────────────────────────────────────────────────────────────
header_tbl = Table(
    [[Paragraph("Dual-Branch ResNet50 — Ablation Study", title_style)],
     [Paragraph("4-Model Comparison  |  model.ipynb  |  2026", subtitle_style)]],
    colWidths=[W],
)
header_tbl.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, -1), DARK_BLUE),
    ("TOPPADDING",    (0, 0), (-1, -1), 14),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ("LEFTPADDING",   (0, 0), (-1, -1), 12),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
]))

story = [header_tbl, Spacer(1, 10)]

# ── 1. Overview ───────────────────────────────────────────────────────────────
story += section("1  Study Overview")

story.append(tbl(
    [["#", "Model", "Fusion Strategy", "Attn output?"],
     ["1", "Baseline",          "Concat → Conv1×1 → AvgPool",                    "—"],
     ["2", "CrossAttn",         "Unidirectional cross-attention (Q=clinic, K,V=derm)", "✓ [B,49,49]"],
     ["3", "ChannelAttn",       "SE channel attention per branch → Concat",       "—"],
     ["4", "Cross+ChannelAttn", "Channel attention → Cross-attention",            "✓ [B,49,49]"]],
    [W*0.05, W*0.22, W*0.58, W*0.15],
    row_colors={1: C1, 2: C2, 3: C3, 4: C4},
))
story.append(Paragraph(
    "All four models share the same backbone, dataset, splits, and training protocol. "
    "Only the <b>fusion mechanism</b> differs — making this a controlled ablation study.",
    note_style))

# ── 2. Shared Backbone ────────────────────────────────────────────────────────
story += section("2  Shared Backbone")

arch_text = (
    "Clinical photo  →  ResNet50_A  →  [B, 2048, 7, 7]  ─┐\n"
    "                                                       ├→  [FUSION]  →  Classifier  →  5 classes\n"
    "Dermoscopic     →  ResNet50_B  →  [B, 2048, 7, 7]  ─┘"
)
story.append(Paragraph(arch_text.replace("\n", "<br/>"), mono_style))
story.append(tbl(
    [["Component", "Detail"],
     ["Backbone (×2)",     "ResNet50, ImageNet pretrained (frozen layers 1–3 during warm-up)"],
     ["Spatial tokens",    "7×7 feature map → 49 tokens per branch"],
     ["Freeze schedule",   "Epochs 1–5: layers 1–3 frozen, layer4 trainable"],
     ["Unfreeze",          "Epoch 6+: layer3+4 unfrozen, lr halved"],
     ["Classifier head",   "Varies by model (see Section 3)"],
     ["Output",            "5-class logits  (BCC, MEL, MISC, NV, SK)"]],
    [W*0.28, W*0.72],
))

# ── 3. Model Architectures ────────────────────────────────────────────────────
story += section("3  Model Architectures")

# Model 1
story.append(Paragraph("<b>Model 1 — Baseline</b>  (no attention)", body_style))
story.append(Paragraph(
    "feat_c [B,2048,7,7]  +  feat_d [B,2048,7,7]  →  Cat [B,4096,7,7]<br/>"
    "→  Conv2d(4096→1024, 1×1)  →  BN  →  ReLU  →  AvgPool  →  [B,1024]<br/>"
    "→  Linear(1024→512)  →  BN  →  ReLU  →  Dropout  →  Linear(512→5)",
    mono_style))

# Model 2
story.append(Paragraph("<b>Model 2 — CrossAttn</b>  (unidirectional cross-attention)", body_style))
story.append(Paragraph(
    "Q ← Linear(2048→256) on clinic tokens  [B, 49, 256]<br/>"
    "K,V ← Linear(2048→256) on derm tokens  [B, 49, 256]<br/>"
    "MultiheadAttention(num_heads=4)  →  Residual+LN  →  FFN(GELU)  →  Residual+LN<br/>"
    "→  AvgPool  →  [B, 256]  →  Linear(256→128)  →  BN  →  ReLU  →  Dropout  →  Linear(128→5)",
    mono_style))

# Model 3
story.append(Paragraph("<b>Model 3 — ChannelAttn</b>  (SE channel attention per branch)", body_style))
story.append(Paragraph(
    "feat_c  →  ChannelAttention(2048, reduction=16)  →  scaled feat_c<br/>"
    "feat_d  →  ChannelAttention(2048, reduction=16)  →  scaled feat_d<br/>"
    "Cat [B,4096,7,7]  →  Conv1×1(1024)  →  BN  →  ReLU  →  AvgPool<br/>"
    "→  Linear(1024→512)  →  BN  →  ReLU  →  Dropout  →  Linear(512→5)",
    mono_style))

# Model 4
story.append(Paragraph("<b>Model 4 — Cross+ChannelAttn</b>  (channel attention → cross-attention)", body_style))
story.append(Paragraph(
    "feat_c  →  ChannelAttention  →  refined_c  [B, 2048, 7, 7]<br/>"
    "feat_d  →  ChannelAttention  →  refined_d  [B, 2048, 7, 7]<br/>"
    "Q ← refined_c,  K,V ← refined_d  →  CrossAttentionFusion  →  [B, 256]<br/>"
    "→  Linear(256→128)  →  BN  →  ReLU  →  Dropout  →  Linear(128→5)",
    mono_style))

story.append(tbl(
    [["SE Block (ChannelAttention)", "Detail"],
     ["Avg-pool path",  "AdaptiveAvgPool2d(1) → FC(2048→128→2048)"],
     ["Max-pool path",  "AdaptiveMaxPool2d(1) → FC(2048→128→2048)  (shared weights)"],
     ["Scale",          "Sigmoid(avg_out + max_out) → multiply feature map channel-wise"],
     ["Reduction ratio","16  (2048 → 128 → 2048)"]],
    [W*0.30, W*0.70],
))

# ── 4. Fixed Hyperparameters ──────────────────────────────────────────────────
story += section("4  Fixed Hyperparameters  (all models identical)")

story.append(tbl(
    [["Parameter", "Value", "Applies to"],
     ["Learning rate (warm-up)",  "0.000750",  "All models"],
     ["Learning rate (after unfreeze)", "0.000375  (lr / 2)", "All models"],
     ["Dropout",                  "0.32",      "All models"],
     ["embed_dim",                "256",       "CrossAttn, Cross+ChannelAttn"],
     ["num_heads",                "4",         "CrossAttn, Cross+ChannelAttn"],
     ["weight_decay",             "0.000957",  "All models"],
     ["batch_size",               "64",        "All models"],
     ["Max epochs",               "30",        "All models"],
     ["Early stopping patience",  "7  (monitors val_loss)", "All models"],
     ["Unfreeze epoch",           "6  (after epoch 5)",     "All models"],
     ["LR scheduler",             "StepLR(step=10, γ=0.5)", "All models"],
     ["Loss function",            "CrossEntropyLoss with inverse-freq class weights", "All models"],
     ["Optimizer",                "Adam",      "All models"],
     ["Seed",                     "42",        "All models"]],
    [W*0.32, W*0.22, W*0.46],
))

# ── 5. Training Protocol ──────────────────────────────────────────────────────
story += section("5  Training Protocol")

story.append(tbl(
    [["Phase", "Epochs", "Layers trainable", "LR"],
     ["Warm-up",      "1–5",  "layer4 + fusion + classifier  (layers 1–3 frozen)", "0.000750"],
     ["Full training","6–30", "layer3 + layer4 + fusion + classifier",              "0.000375"]],
    [W*0.15, W*0.12, W*0.53, W*0.20],
))
story.append(Paragraph(
    "Early stopping restores best-val-loss weights automatically. "
    "Each model is trained independently from scratch with a fresh optimizer and scheduler.",
    note_style))

# ── 6. Dataset ────────────────────────────────────────────────────────────────
story += section("6  Dataset & Splits")

story.append(tbl(
    [["Source", "Samples", "Type"],
     ["Derm7pt",   "1,011", "Paired clinical + dermoscopic images"],
     ["MILK10k",   "4,361", "Paired clinical + dermoscopic images"],
     ["Combined",  "5,372", "Stratified 70 / 15 / 15 split"]],
    [W*0.20, W*0.15, W*0.65],
))
story.append(Spacer(1, 6))
story.append(tbl(
    [["Split", "Samples"],
     ["Train",      "3,760"],
     ["Validation",   "806"],
     ["Test",          "806"]],
    [W*0.5, W*0.5],
))
story.append(Spacer(1, 6))
story.append(tbl(
    [["Class", "Description", "Loss Weight"],
     ["BCC",  "Basal Cell Carcinoma",          "0.419"],
     ["MEL",  "Melanoma (all subtypes)",        "1.532"],
     ["MISC", "Lentigo, DF, Vascular, etc.",    "5.489"],
     ["NV",   "Nevus (all subtypes)",           "0.813"],
     ["SK",   "Seborrheic Keratosis",           "1.825"]],
    [W*0.10, W*0.65, W*0.25],
))
story.append(Paragraph(
    "Class imbalance handled via <b>WeightedRandomSampler</b> (training batches) "
    "and <b>weighted CrossEntropyLoss</b> (inverse class frequency).",
    note_style))

# ── 7. Saved Artifacts ────────────────────────────────────────────────────────
story += section("7  Saved Artifacts  (per run)")

story.append(tbl(
    [["File", "Contents"],
     ["model_baseline.pth",          "Baseline weights + history + test metrics"],
     ["model_cross_attn.pth",        "CrossAttn weights + history + test metrics"],
     ["model_channel_attn.pth",      "ChannelAttn weights + history + test metrics"],
     ["model_cross_channel_attn.pth","Cross+ChannelAttn weights + history + test metrics"],
     ["all_histories.pkl",           "Train/val loss + accuracy per epoch for all 4 models"],
     ["all_metrics.pkl",             "Test metrics dict for all 4 models"],
     ["all_eval_data.pkl",           "y_true, y_pred, y_probs, attn weights for all 4 models"],
     ["comparison_metrics.csv",      "Side-by-side metrics table (CSV)"]],
    [W*0.38, W*0.62],
))

# ── 8. Visualizations ────────────────────────────────────────────────────────
story += section("8  Comparison Visualizations  (Section 10)")

story.append(tbl(
    [["Plot", "Description"],
     ["10.1  Training curves",      "Loss + accuracy per epoch; solid=val, dashed=train; 4 colors"],
     ["10.2  Metrics bar + F1 heatmap","Grouped bar chart (6 metrics) + per-class F1 heatmap (4×5)"],
     ["10.3  ROC curves",           "One subplot per class (5 total), 4 lines per model"],
     ["10.4  Confusion matrices",   "2×2 grid of normalized confusion matrices (one per model)"],
     ["10.5  Cross-attention maps", "Attention overlay on dermoscopic images (CrossAttn & Cross+Channel only)"],
     ["10.6  Sample predictions",   "Correct vs incorrect examples from CrossAttn model"]],
    [W*0.28, W*0.72],
))

# ── footer ────────────────────────────────────────────────────────────────────
story.append(Spacer(1, 14))
story.append(HRFlowable(width="100%", thickness=0.5, color=GRAY_LINE))
story.append(Paragraph(
    "model.ipynb  |  Dual-Branch ResNet50 Ablation Study  |  2026",
    caption_style))

doc.build(story)
print(f"PDF saved: {OUTPUT}")
