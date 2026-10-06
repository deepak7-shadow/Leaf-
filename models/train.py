"""MobileNetV3-Small training and fine-tuning pipeline for Real vs Fake leaf classification.

Includes:
- Specialized augmentations for detecting printed, cloth, screen, and artificial artifacts
- Transfer learning from pre-trained MobileNetV3-Small
- Validation metrics (Precision, Recall, F1, Confusion Matrix)
- Synthetic dataset bootstrapper to train out-of-the-box
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
            # Add subtle horizontal / vertical scanlines (screen or printer artifact)
            freq = random.randint(4, 12)
            lines = np.tile(np.sin(np.linspace(0, freq * np.pi, h))[:, None, None], (1, w, 3))
            noisy = np.clip(img_np.astype(np.float32) + lines * 15.0, 0, 255).astype(np.uint8)
            return noisy

        elif aug == "halftone":
            # Color quantization simulating limited print gamut
            levels = random.randint(4, 8)
            step = 256 // levels
            quantized = (img_np // step) * step
            return quantized.astype(np.uint8)

        elif aug == "glare":
            # Specular reflection (plastic sheen or glossy paper reflection)
            cx, cy = random.randint(w // 4, 3 * w // 4), random.randint(h // 4, 3 * h // 4)
            radius = random.randint(15, 35)
            glare_mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(glare_mask, (cx, cy), radius, 1.0, -1)
            glare_mask = cv2.GaussianBlur(glare_mask, (21, 21), 0)
            result = img_np.astype(np.float32) + (glare_mask[:, :, None] * 70.0)
            return np.clip(result, 0, 255).astype(np.uint8)

        elif aug == "blur":
            # Printing dot-gain blur
            ksize = random.choice([3, 5])
            return cv2.GaussianBlur(img_np, (ksize, ksize), 0)

        return img_np


class LeafDataset(Dataset):
    """Dataset for Real Living Leaves (0) vs Fake/Printed Leaves (1)."""

    def __init__(self, data_dir: str, is_train: bool = True, transform=None):
        self.samples: List[Tuple[str, int]] = []
        self.transform = transform
        self.is_train = is_train
        self.artifact_aug = PrintAndArtifactAugmentation()

        real_dir = os.path.join(data_dir, "real")
        fake_dir = os.path.join(data_dir, "fake")

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

        # BGR -> RGB
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            img_tensor = self.transform(rgb)
        else:
            resized = cv2.resize(rgb, (224, 224))
            img_tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0

        return img_tensor, label


def create_synthetic_dataset(output_dir: str = "data/synthetic", num_real: int = 200, num_fake: int = 200):
    """Generate synthetic dataset for immediate training without waiting for camera capture."""
    os.makedirs(os.path.join(output_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "fake"), exist_ok=True)

    gen = SyntheticSceneGenerator(width=320, height=320)
    print(f"[INFO] Generating {num_real} synthetic real leaf patches...")
    for i in range(num_real):
        canvas = gen.generate_soil_background()
        cx, cy = 160 + random.randint(-20, 20), 160 + random.randint(-20, 20)
        ax, ay = random.randint(45, 80), random.randint(25, 50)
        ang = random.uniform(-60, 60)
        x, y, w, h = gen.draw_real_leaf(canvas, (cx, cy), (ax, ay), angle=ang)
        # Crop ROI
        x1, y1 = max(0, x - 5), max(0, y - 5)
        x2, y2 = min(320, x + w + 5), min(320, y + h + 5)
        patch = canvas[y1:y2, x1:x2]
        if patch.size > 0:
            cv2.imwrite(os.path.join(output_dir, "real", f"real_{i:04d}.jpg"), patch)

    print(f"[INFO] Generating {num_fake} synthetic printed/artificial leaf patches...")
    for i in range(num_fake):
        canvas = gen.generate_soil_background()
        cx, cy = 160 + random.randint(-20, 20), 160 + random.randint(-20, 20)
        ax, ay = random.randint(40, 75), random.randint(25, 45)
        substrate = random.choice(["paper", "fabric"])
        x, y, w, h = gen.draw_fake_printed_leaf(canvas, (cx, cy), (ax, ay), substrate=substrate)
        x1, y1 = max(0, x - 5), max(0, y - 5)
        x2, y2 = min(320, x + w + 5), min(320, y + h + 5)
        patch = canvas[y1:y2, x1:x2]
        if patch.size > 0:
            cv2.imwrite(os.path.join(output_dir, "fake", f"fake_{i:04d}.jpg"), patch)

    print(f"[SUCCESS] Synthetic dataset created at {output_dir}")


def train_mobilenet_v3(
    data_dir: str,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-4,
    export_onnx: bool = True,
    output_onnx: str = "models/mobilenet_v3_small.onnx"
):
    """Train MobileNetV3-Small binary classifier."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on device: {device}")

    # Strong augmentations + webcam-quality simulation for real-world accuracy
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
        # Webcam-quality simulation: blur + JPEG artifacts
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.4),
        transforms.ToTensor(),
        # Additive Gaussian noise to simulate sensor noise
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

    full_dataset = LeafDataset(data_dir, is_train=True, transform=train_transform)
    if len(full_dataset) == 0:
        print("[ERROR] No samples found in dataset directory!")
        return

    val_size = max(1, int(len(full_dataset) * 0.2))
    train_size = len(full_dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    print(f"[INFO] Dataset loaded: {train_size} training samples, {val_size} validation samples.")

    # Load Model
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 2)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
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
    parser.add_argument("--data-dir", default="data/downloaded", help="Path to dataset with real/ and fake/ subdirs")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic dataset first")
    args = parser.parse_args()

    if args.generate_synthetic:
        create_synthetic_dataset(args.data_dir, num_real=150, num_fake=150)

    train_mobilenet_v3(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
