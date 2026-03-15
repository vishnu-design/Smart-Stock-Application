"""
SmartStock CV Subsystem — Model 2: YOLOv8 Classification
=========================================================
Uses Ultralytics YOLOv8n-cls (nano classification) for damaged goods detection.
YOLOv8-cls is a modern unified model — faster inference, designed for edge deployment.

Requirements:
    pip install ultralytics scikit-learn matplotlib seaborn

Usage:
    python train_yolov8.py --data_dir ./dataset --epochs 30
"""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train YOLOv8-cls for damage detection")
parser.add_argument("--data_dir",   type=str, default="./dataset")
parser.add_argument("--epochs",     type=int, default=30)
parser.add_argument("--imgsz",      type=int, default=224)
parser.add_argument("--batch",      type=int, default=32)
parser.add_argument("--output_dir", type=str, default="./models")
args = parser.parse_args()

Path(args.output_dir).mkdir(exist_ok=True)

# ── YOLOv8 expects a specific folder structure:
#    dataset/
#        train/undamaged/*.jpg  train/damaged/*.jpg
#        val/undamaged/*.jpg    val/damaged/*.jpg
#        test/undamaged/*.jpg   test/damaged/*.jpg
#
#   This matches exactly what generate_dataset.py produces — no restructuring needed.

print("── Training YOLOv8n-cls ──────────────────────────────────────")
print(f"Dataset:  {args.data_dir}")
print(f"Epochs:   {args.epochs}")
print(f"Img size: {args.imgsz}")

from ultralytics import YOLO

# yolov8n-cls = nano classification model (fastest, smallest)
model = YOLO("yolov8n-cls.pt")

results = model.train(
    data=args.data_dir,
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    name="smartstock_yolo",
    project=args.output_dir,
    exist_ok=True,
    verbose=True,
    patience=10,            # early stopping
    augment=True,           # built-in augmentation
    degrees=15,             # rotation augment
    fliplr=0.5,             # horizontal flip
    hsv_v=0.3,              # brightness jitter
    hsv_s=0.2,              # saturation jitter
)

print("\n── Evaluating on test set ──────────────────────────────────")
best_weights = Path(args.output_dir) / "smartstock_yolo" / "weights" / "best.pt"
model_best = YOLO(str(best_weights))

# Predict on test set
test_path = Path(args.data_dir) / "test"
CLASSES = ["undamaged", "damaged"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

all_preds, all_labels = [], []
for cls in CLASSES:
    cls_folder = test_path / cls
    true_label = CLASS_TO_IDX[cls]
    for img_path in sorted(cls_folder.glob("*.jpg")):
        result = model_best.predict(str(img_path), verbose=False)[0]
        pred_idx = result.probs.top1
        all_preds.append(pred_idx)
        all_labels.append(true_label)

print("\n── Test Set Results ──────────────────────────────────────────")
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
    "model": "YOLOv8n-cls",
    "accuracy": round(acc, 4), "precision": round(prec, 4),
    "recall": round(rec, 4), "f1": round(f1, 4),
    "confusion_matrix": cm.tolist(),
}
metrics_path = Path(args.output_dir) / "yolo_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved → {metrics_path}")

# ── Confusion matrix plot ──────────────────────────────────────────────────────
import seaborn as sns
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax,
            xticklabels=CLASSES, yticklabels=CLASSES)
ax.set_title("YOLOv8n-cls — Confusion Matrix (Test Set)")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plot_path = Path(args.output_dir) / "yolo_cm.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plot saved → {plot_path}")
print("\nDone. Best weights → ", best_weights)
