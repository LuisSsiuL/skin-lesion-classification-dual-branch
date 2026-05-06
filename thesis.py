import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np

# ==========================================
# 1. SETUP & DATA PREPROCESSING
# ==========================================
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define data augmentation and normalization for training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define standard transformations for validation
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. LOAD THE DATASET
# ==========================================
# IMPORTANT: Change these to your actual Google Drive paths!
train_dir = '/content/drive/MyDrive/My_Skin_Data/train'
val_dir = '/content/drive/MyDrive/My_Skin_Data/val'

try:
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_disease_classes = len(train_dataset.classes)
    print(f"Loaded {num_disease_classes} disease classes.")
except FileNotFoundError:
    print("Dataset folders not found. Please check your Drive paths or run with dummy data.")
    # Fallback to 7 classes (like Derm7pt) if folders aren't set up yet
    num_disease_classes = 7

# ==========================================
# 3. BUILD THE RESNET-18 BASELINE MODEL
# ==========================================
# Load pre-trained weights to capture foundational visual features
resnet18_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Replace the final fully connected layer for your specific dataset
num_features = resnet18_model.fc.in_features
resnet18_model.fc = nn.Linear(num_features, num_disease_classes)

# Move model to GPU
resnet18_model = resnet18_model.to(device)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18_model.parameters(), lr=0.001)
epochs = 5 # Set to 5 for a quick preliminary test

print("\n--- Starting Training ---")
for epoch in range(epochs):
    resnet18_model.train()
    running_loss = 0.0

    # Skip training loop if data isn't loaded (for demonstration purposes)
    if 'train_loader' not in locals():
        print("Skipping training loop: No data loaders available.")
        break

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = resnet18_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {running_loss/len(train_loader):.4f}")

# ==========================================
# 5. EVALUATION AND METRICS CALCULATION
# ==========================================
print("\n--- Calculating Preliminary Metrics ---")
resnet18_model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    if 'val_loader' in locals():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = resnet18_model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            # Store results for metrics
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_scores = np.array(all_probs)

        # Calculate standard metrics
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Calculate mean Average Precision (mAP)
        y_true_binarized = label_binarize(y_true, classes=range(num_disease_classes))
        mAP = average_precision_score(y_true_binarized, y_scores, average="macro")

        print("\n=========================================")
        print("  MID-TERM REPORT PRELIMINARY RESULTS  ")
        print("=========================================")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall:    {recall * 100:.2f}%")
        print(f"F1-score:  {f1 * 100:.2f}%")
        print(f"mAP:       {mAP * 100:.2f}%")
        print("=========================================")
    else:
        print("Cannot calculate metrics without validation data.")
