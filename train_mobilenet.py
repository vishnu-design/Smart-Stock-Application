"""
SmartStock CV Subsystem — Model 1: MobileNetV2 (Transfer Learning)
==================================================================
Uses pre-trained MobileNetV2 (ImageNet weights) with a custom classifier head.
Fine-tunes on the synthetic damaged goods dataset.

Requirements:
    pip install torch torchvision scikit-learn matplotlib seaborn pillow

Usage:
    python train_mobilenet.py --data_dir ./dataset --epochs 15 --batch_size 32
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ── CLI args ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train MobileNetV2 for damage detection")
parser.add_argument("--data_dir", type=str, default="./dataset")
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--output_dir", type=str, default="./models")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
Path(args.output_dir).mkdir(exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
CLASSES = ["undamaged", "damaged"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class DamageDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.samples = []
        self.transform = transform
        for cls in CLASSES:
            folder = Path(root) / split / cls
            for f in sorted(folder.glob("*.jpg")):
                self.samples.append((str(f), CLASS_TO_IDX[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


train_ds = DamageDataset(args.data_dir, "train", TRAIN_TRANSFORMS)
val_ds   = DamageDataset(args.data_dir, "val",   EVAL_TRANSFORMS)
test_ds  = DamageDataset(args.data_dir, "test",  EVAL_TRANSFORMS)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)
print(f"Dataset — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

# ── Model ──────────────────────────────────────────────────────────────────────
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze all layers except the classifier
for param in model.features.parameters():
    param.requires_grad = False

# Replace the classifier head
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 2),
)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters — total: {total_params:,}  trainable: {trainable:,}")

# ── Training loop ──────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
best_val_acc = 0.0
best_model_path = Path(args.output_dir) / "mobilenet_best.pth"


def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if training:
                optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


print("\n── Training MobileNetV2 ──────────────────────────────────────")
start = time.time()
for epoch in range(1, args.epochs + 1):
    tr_loss, tr_acc = run_epoch(train_loader, training=True)
    va_loss, va_acc = run_epoch(val_loader,   training=False)
    scheduler.step(va_loss)

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)

    flag = " ★ best" if va_acc > best_val_acc else ""
    if va_acc > best_val_acc:
        best_val_acc = va_acc
        torch.save(model.state_dict(), best_model_path)
    print(f"  Epoch {epoch:02d}/{args.epochs}  "
          f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
          f"val_loss={va_loss:.4f}  val_acc={va_acc:.4f}{flag}")

print(f"\nTraining complete in {time.time()-start:.1f}s  |  best val_acc={best_val_acc:.4f}")

# ── Evaluation on test set ─────────────────────────────────────────────────────
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.eval()
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

print("\n── Test Set Results ─────────────────────────────────────────")
acc  = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec  = recall_score(all_labels, all_preds)
f1   = f1_score(all_labels, all_preds)
cm   = confusion_matrix(all_labels, all_preds)

print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print("\n" + classification_report(all_labels, all_preds, target_names=CLASSES))

# ── Save metrics ───────────────────────────────────────────────────────────────
metrics = {
    "model": "MobileNetV2",
    "accuracy": round(acc, 4), "precision": round(prec, 4),
    "recall": round(rec, 4), "f1": round(f1, 4),
    "confusion_matrix": cm.tolist(),
    "history": history,
    "epochs": args.epochs, "best_val_acc": round(best_val_acc, 4),
}
metrics_path = Path(args.output_dir) / "mobilenet_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved → {metrics_path}")

# ── Confusion matrix plot ──────────────────────────────────────────────────────
import seaborn as sns
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=CLASSES, yticklabels=CLASSES)
axes[0].set_title("MobileNetV2 — Confusion Matrix (Test Set)")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

# Training curves
epochs_range = range(1, args.epochs + 1)
axes[1].plot(epochs_range, history["train_acc"], label="Train Acc", marker="o", ms=3)
axes[1].plot(epochs_range, history["val_acc"],   label="Val Acc",   marker="s", ms=3)
axes[1].set_title("MobileNetV2 — Training Curves")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plot_path = Path(args.output_dir) / "mobilenet_results.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plot saved → {plot_path}")
print("\nDone.")
