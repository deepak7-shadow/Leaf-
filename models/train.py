"""MobileNetV3-Small training pipeline for Leaf Authenticity Detection.

3-class architecture:
- Class 0: REAL_LEAF   (genuine living crop leaves: healthy & diseased)
- Class 1: FAKE_LEAF   (printed, screened, artificial leaf images)
- Class 2: NOT_LEAF    (non-leaf objects: bags, clothing, faces, skin, plastic, background)

Features:
- Balanced sampling across all 3 classes
- Field augmentations (lighting, motion blur, color jitter, outdoor shifts)
- Print/screen artifact augmentations on synthetic images
- Backbone transfer learning with warmup and fine-tuning
- Automatic ONNX export for fast edge/webcam inference
"""

from typing import Dict, List, Optional, Tuple
import argparse
import os
import random
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from models.export_onnx import export_mobilenet_v3_small

# ─── Label Map ────────────────────────────────────────────────────────────────
LABEL_REAL     = 0  # genuine living leaf
LABEL_FAKE     = 1  # printed / screened / artificial leaf
LABEL_NOT_LEAF = 2  # non-leaf object (bag, clothing, face, plastic, background…)
NUM_CLASSES    = 3
CLASS_NAMES    = ["REAL", "FAKE", "NOT_LEAF"]


# ─── Augmentations ────────────────────────────────────────────────────────────

class FieldAugmentation:
    """Fast field condition simulation: lighting, blur, noise, compression."""

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        choice = random.randint(0, 5)
        if choice == 0:
            # Blur
            k = random.choice([3, 5])
            return cv2.GaussianBlur(img_bgr, (k, k), 0)
        elif choice == 1:
            # Noise
            noise = np.random.randint(-15, 15, img_bgr.shape, dtype=np.int16)
            return np.clip(img_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif choice == 2:
            # Outdoor lighting / brightness
            factor = random.uniform(0.7, 1.3)
            return np.clip(img_bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        elif choice == 3:
            # Motion blur
            ksize = 5
            kernel = np.zeros((ksize, ksize), dtype=np.float32)
            kernel[ksize // 2, :] = 1.0 / ksize
            return cv2.filter2D(img_bgr, -1, kernel)
        elif choice == 4:
            # JPEG compression
            quality = random.randint(40, 80)
            _, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            return cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return img_bgr


class ArtifactAugmentation:
    """Simulates screen moire, halftone, and glare for FAKE images."""

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        choice = random.randint(0, 3)
        h, w = img_bgr.shape[:2]
        if choice == 0:
            # Moire lines
            freq = random.randint(4, 10)
            lines = np.tile(np.sin(np.linspace(0, freq * np.pi, h))[:, None, None], (1, w, 3))
            return np.clip(img_bgr.astype(np.float32) + lines * 12.0, 0, 255).astype(np.uint8)
        elif choice == 1:
            # Glare
            cx, cy = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
            rad = random.randint(15, 30)
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (cx, cy), rad, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            return np.clip(img_bgr.astype(np.float32) + mask[:, :, None] * 60.0, 0, 255).astype(np.uint8)
        elif choice == 2:
            # Screen tint
            img = img_bgr.astype(np.float32)
            img[:, :, 0] = np.clip(img[:, :, 0] * 1.06, 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * 0.95, 0, 255)
            return img.astype(np.uint8)
        return img_bgr


# ─── Dataset ──────────────────────────────────────────────────────────────────

class FastLeafDataset(Dataset):
    """Memory-efficient 3-class dataset."""

    def __init__(self, samples: List[Tuple[str, int]], is_train: bool = True, transform=None):
        self.samples = samples
        self.is_train = is_train
        self.transform = transform
        self.field_aug = FieldAugmentation()
        self.artifact_aug = ArtifactAugmentation()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        bgr = cv2.imread(path)
        if bgr is None:
            bgr = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.is_train and random.random() < 0.35:
            bgr = self.field_aug(bgr)

        if self.is_train and label == LABEL_FAKE and random.random() < 0.40:
            bgr = self.artifact_aug(bgr)

        # Direct BGR -> RGB resize
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        if self.transform:
            img_tensor = self.transform(resized)
        else:
            img_tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0

        return img_tensor, label


# ─── Training Pipeline ────────────────────────────────────────────────────────

def train_mobilenet_v3(
    data_dir: str = "data/training",
    epochs: int = 6,
    batch_size: int = 32,
    lr: float = 5e-4,
    max_per_class: int = 1200,
    export_onnx: bool = True,
    output_onnx: str = "models/mobilenet_v3_small.onnx",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}", flush=True)
    print(f"[INFO] Architecture: MobileNetV3-Small (Pretrained) | Classes: {CLASS_NAMES}", flush=True)

    exts = ('.png', '.jpg', '.jpeg')
    class_dirs = [(LABEL_REAL, "real"), (LABEL_FAKE, "fake"), (LABEL_NOT_LEAF, "not_leaf")]

    # Collect and balance samples
    train_samples = []
    val_samples = []

    random.seed(42)

    for label, subdir in class_dirs:
        d = os.path.join(data_dir, subdir)
        if not os.path.exists(d):
            continue
        files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(exts)]
        random.shuffle(files)
        selected = files[:max_per_class]

        val_n = max(20, int(len(selected) * 0.15))
        val_samples.extend([(f, label) for f in selected[:val_n]])
        train_samples.extend([(f, label) for f in selected[val_n:]])
        print(f"  Class {CLASS_NAMES[label]}: {len(selected)} loaded ({len(selected)-val_n} train, {val_n} val)", flush=True)

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    print(f"[INFO] Total Train: {len(train_samples)} | Total Val: {len(val_samples)}", flush=True)

    # Fast transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_set = FastLeafDataset(train_samples, is_train=True,  transform=train_transform)
    val_set   = FastLeafDataset(val_samples,   is_train=False, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize pretrained MobileNetV3-Small
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    model.to(device)

    # Balanced class weights
    counts = [sum(1 for _, l in train_samples if l == c) for c in range(NUM_CLASSES)]
    total = sum(counts)
    class_weights = torch.tensor(
        [total / (NUM_CLASSES * max(1, c)) for c in counts],
        dtype=torch.float32
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    best_weights_path = "models/best_weights.pth"
    os.makedirs("models", exist_ok=True)

    print("\n[START] Training loop...", flush=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        correct = 0
        total_n = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_n += labels.size(0)

        scheduler.step()
        train_acc = (correct / total_n) * 100.0 if total_n > 0 else 0.0

        # Validate
        model.eval()
        val_correct = 0
        val_total   = 0
        per_class_correct = [0] * NUM_CLASSES
        per_class_total   = [0] * NUM_CLASSES

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)
                for c in range(NUM_CLASSES):
                    mask = labels == c
                    per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                    per_class_total[c]   += mask.sum().item()

        val_acc = (val_correct / val_total) * 100.0 if val_total > 0 else 0.0
        dur = time.time() - t0

        per_class_str = " | ".join(
            f"{CLASS_NAMES[c]}: {per_class_correct[c]/max(1,per_class_total[c])*100:.0f}%"
            for c in range(NUM_CLASSES)
        )
        print(f"Epoch {epoch:02d}/{epochs:02d} [{dur:.1f}s] - Train: {train_acc:5.1f}% - Val: {val_acc:5.1f}% | {per_class_str}", flush=True)

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)

    print(f"\n[DONE] Best Validation Accuracy: {best_val_acc:.1f}%", flush=True)

    if export_onnx:
        print("[INFO] Exporting to ONNX format...", flush=True)
        export_mobilenet_v3_small(
            output_path=output_onnx,
            weights_path=best_weights_path,
            num_classes=NUM_CLASSES,
        )
        print(f"[SUCCESS] Exported ONNX model to {output_onnx}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3-class MobileNetV3-Small for Leaf Authenticity Detection")
    parser.add_argument("--data-dir",       default="data/training")
    parser.add_argument("--epochs",         type=int,   default=6)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=5e-4)
    parser.add_argument("--max-per-class",  type=int,   default=1200)
    args = parser.parse_args()

    train_mobilenet_v3(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_per_class=args.max_per_class,
    )
