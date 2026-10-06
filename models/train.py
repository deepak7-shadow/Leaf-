"""MobileNetV3-Small training and fine-tuning pipeline for Real vs Fake leaf classification.

Includes:
- Specialized augmentations for detecting printed, cloth, screen, and artificial artifacts
- Transfer learning from pre-trained MobileNetV3-Small
- Validation metrics
- Automatic ONNX export of best model checkpoint
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

# Add parent directory to sys.path if run directly
_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from models.export_onnx import export_mobilenet_v3_small
from utils.synthetic_generator import SyntheticSceneGenerator


class PrintAndArtifactAugmentation:
    """Simulates physical printing, screen moire, and artificial leaf characteristics."""

    def __call__(self, img_np: np.ndarray) -> np.ndarray:
        """Apply random artifact transform to uint8 BGR image."""
        aug = random.choice(["none", "moire", "halftone", "glare", "blur"])
        h, w = img_np.shape[:2]

        if aug == "moire":
            freq = random.randint(4, 12)
            lines = np.tile(np.sin(np.linspace(0, freq * np.pi, h))[:, None, None], (1, w, 3))
            noisy = np.clip(img_np.astype(np.float32) + lines * 15.0, 0, 255).astype(np.uint8)
            return noisy

        elif aug == "halftone":
            levels = random.randint(4, 8)
            step = 256 // levels
            quantized = (img_np // step) * step
            return quantized.astype(np.uint8)

        elif aug == "glare":
            cx, cy = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
            radius = random.randint(15, 35)
            glare_mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(glare_mask, (cx, cy), radius, 1.0, -1)
            glare_mask = cv2.GaussianBlur(glare_mask, (21, 21), 0)
            result = img_np.astype(np.float32) + (glare_mask[:, :, None] * 70.0)
            return np.clip(result, 0, 255).astype(np.uint8)

        elif aug == "blur":
            ksize = random.choice([3, 5])
            return cv2.GaussianBlur(img_np, (ksize, ksize), 0)

        return img_np


class LeafDataset(Dataset):
    """Dataset for Real Living Leaves (0) vs Fake/Printed Leaves (1)."""

    def __init__(self, data_or_samples, is_train: bool = True, transform=None):
        self.transform = transform
        self.is_train = is_train
        self.artifact_aug = PrintAndArtifactAugmentation()

        if isinstance(data_or_samples, list):
            self.samples = data_or_samples
        else:
            self.samples: List[Tuple[str, int]] = []
            real_dir = os.path.join(data_or_samples, "real")
            fake_dir = os.path.join(data_or_samples, "fake")

            for f in os.listdir(real_dir) if os.path.exists(real_dir) else []:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(real_dir, f), 0))

            for f in os.listdir(fake_dir) if os.path.exists(fake_dir) else []:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(fake_dir, f), 1))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        bgr = cv2.imread(path)
        if bgr is None:
            bgr = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.is_train and label == 1 and random.random() < 0.5:
            bgr = self.artifact_aug(bgr)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            img_tensor = self.transform(rgb)
        else:
            resized = cv2.resize(rgb, (224, 224))
            img_tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0

        return img_tensor, label


def train_mobilenet_v3(
    data_dir: str = "data/training",
    epochs: int = 12,
    batch_size: int = 32,
    lr: float = 1e-4,
    export_onnx: bool = True,
    output_onnx: str = "models/mobilenet_v3_small.onnx"
):
    """Train MobileNetV3-Small binary classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    # Webcam-quality simulation augmentations
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.4),
        transforms.ToTensor(),
        transforms.RandomApply([
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.03)
        ], p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "fake")

    samples = []
    for f in os.listdir(real_dir) if os.path.exists(real_dir) else []:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            samples.append((os.path.join(real_dir, f), 0))

    for f in os.listdir(fake_dir) if os.path.exists(fake_dir) else []:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            samples.append((os.path.join(fake_dir, f), 1))

    if not samples:
        print(f"[ERROR] No samples found in {data_dir}!")
        return

    random.seed(42)
    random.shuffle(samples)

    val_size = max(1, int(len(samples) * 0.2))
    train_samples = samples[val_size:]
    val_samples   = samples[:val_size]

    train_set = LeafDataset(train_samples, is_train=True, transform=train_transform)
    val_set   = LeafDataset(val_samples, is_train=False, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    n_real = sum(1 for _, l in train_samples if l == 0)
    n_fake = sum(1 for _, l in train_samples if l == 1)
    print(f"[INFO] Train set: {len(train_set)} ({n_real} Real, {n_fake} Fake). Val set: {len(val_set)}.")

    # Load Pre-trained Model
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    model.to(device)

    # Class-weighted loss for balanced gradient updates
    tot = n_real + n_fake
    class_weights = torch.tensor([tot / (2.0 * max(1, n_real)), tot / (2.0 * max(1, n_fake))], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_weights_path = "models/best_weights.pth"
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

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
            total += labels.size(0)

        train_acc = (correct / total) * 100.0 if total > 0 else 0.0

        # Validation Phase
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = (val_correct / val_total) * 100.0 if val_total > 0 else 0.0
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc:5.1f}% | Val Acc: {val_acc:5.1f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)

    print(f"[SUCCESS] Training complete! Best validation accuracy: {best_val_acc:.1f}%")

    if export_onnx:
        print("[INFO] Exporting fine-tuned model to ONNX...")
        export_mobilenet_v3_small(output_path=output_onnx, weights_path=best_weights_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Small for Leaf Detection")
    parser.add_argument("--data-dir", default="data/training", help="Path to dataset with real/ and fake/ subdirs")
    parser.add_argument("--output-dir", default="models", help="Directory for output models")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    train_mobilenet_v3(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
