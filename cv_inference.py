"""
SmartStock CV Subsystem — Inference Module
==========================================
Loads the best trained model (MobileNetV2 or YOLOv8) and runs
prediction on a single PIL Image. Used by the Streamlit app.

Usage:
    from cv_inference import DamageDetector
    detector = DamageDetector(model_type="mobilenet", weights_path="models/mobilenet_best.pth")
    result = detector.predict(pil_image)
    # result = {"label": "damaged", "confidence": 0.94, "damaged_prob": 0.94}
"""

from pathlib import Path
from PIL import Image
import numpy as np


CLASSES = ["undamaged", "damaged"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class DamageDetector:
    """
    Unified inference wrapper for MobileNetV2 or YOLOv8-cls.
    Falls back to a demo mode (rule-based simulation) if model weights
    are not yet available — useful during development and demos.
    """

    def __init__(self, model_type: str = "mobilenet",
                 weights_path: str = None,
                 demo_mode: bool = False):
        """
        Args:
            model_type:   "mobilenet" | "yolo"
            weights_path: path to .pth (mobilenet) or .pt (yolo) weights file
            demo_mode:    if True, skip loading weights and use heuristic prediction
        """
        self.model_type = model_type.lower()
        self.demo_mode  = demo_mode or (weights_path is None)
        self.model      = None

        if not self.demo_mode:
            self._load_model(weights_path)

    def _load_model(self, weights_path: str):
        if self.model_type == "mobilenet":
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = models.mobilenet_v2(weights=None)
            model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(model.last_channel, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 2),
            )
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            model.eval()
            self.model = model.to(self.device)

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
            import torch as _torch
            self._torch = _torch

        elif self.model_type == "yolo":
            from ultralytics import YOLO
            self.model = YOLO(weights_path)

    def predict(self, image: Image.Image) -> dict:
        """
        Run damage detection on a PIL Image.

        Returns:
            {
                "label":        "damaged" | "undamaged",
                "confidence":   float (0-1),
                "damaged_prob": float (0-1),
                "model_used":   str,
            }
        """
        image = image.convert("RGB").resize((224, 224))

        if self.demo_mode:
            return self._demo_predict(image)

        if self.model_type == "mobilenet":
            return self._mobilenet_predict(image)
        elif self.model_type == "yolo":
            return self._yolo_predict(image)

    def _mobilenet_predict(self, image: Image.Image) -> dict:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            logits = self.model(tensor)
            probs  = self._torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        return {
            "label":        CLASSES[pred_idx],
            "confidence":   float(probs[pred_idx]),
            "damaged_prob": float(probs[1]),
            "model_used":   "MobileNetV2",
        }

    def _yolo_predict(self, image: Image.Image) -> dict:
        result = self.model.predict(image, verbose=False)[0]
        pred_idx   = result.probs.top1
        confidence = float(result.probs.top1conf)
        probs      = result.probs.data.cpu().numpy()
        return {
            "label":        CLASSES[pred_idx],
            "confidence":   confidence,
            "damaged_prob": float(probs[1]),
            "model_used":   "YOLOv8n-cls",
        }

    def _demo_predict(self, image: Image.Image) -> dict:
        """
        Heuristic simulation for demo / development when no trained
        weights are available. Uses crush signal and local variance
        as the primary damage indicators.
        """
        from PIL import ImageFilter
        arr = np.array(image).astype(np.float32)

        # Signal 1: horizontal row discontinuity → crush deformation
        row_means    = arr.mean(axis=(1, 2))
        row_diff     = np.abs(np.diff(row_means)).max()
        crush_signal = min(row_diff / 25.0, 1.0)

        # Signal 2: local patch variance → cracks / scratches
        patch     = arr[80:144, 80:144]
        local_var = min(np.var(patch) / 1200.0, 1.0)

        # Signal 3: dark pixel ratio → dent discolouration
        dark_pixels = min(np.mean(arr < 80) * 15.0, 1.0)

        # Signal 4: edge density (lower weight — similar across both classes)
        grey          = image.convert("L")
        edges         = grey.filter(ImageFilter.FIND_EDGES)
        edge_density  = np.mean(np.array(edges)) / 255.0

        damage_score = (
            0.45 * crush_signal +
            0.30 * local_var    +
            0.15 * dark_pixels  +
            0.10 * edge_density
        )

        # Deterministic per-image noise using pixel fingerprint
        rng_seed = int(arr[100, 100, 0]) + int(arr[50, 50, 1]) + int(arr[150, 150, 2])
        rng      = np.random.RandomState(rng_seed % (2**31))
        noise    = rng.uniform(-0.06, 0.06)
        damage_score = float(np.clip(damage_score + noise, 0.0, 1.0))

        threshold = 0.42
        if damage_score > threshold:
            label      = "damaged"
            confidence = min(0.60 + damage_score * 0.35, 0.99)
        else:
            label      = "undamaged"
            confidence = min(0.60 + (1 - damage_score) * 0.35, 0.99)

        return {
            "label":        label,
            "confidence":   round(confidence, 4),
            "damaged_prob": round(damage_score, 4),
            "model_used":   "Demo (heuristic — train models for real inference)",
        }
