"""AI Nozzle - 10-Class Disease & Authenticity Deep Learning Training Pipeline.

Trains MobileNetV3-Small on 10,000 images (1,000 images per class):
0: Anthracnose
1: Bacterial_Leaf_Spot
2: Black_rot
3: Downy_Mildew
4: Healthy
5: Leaf_Spot_Blight
6: Mosaic_Disease
7: Not_Leaf
8: Powdery_Mildew
9: Rust

Outputs:
- models/crop_model_10class_best.pth (PyTorch checkpoint)
- models/crop_model_10class.onnx (High-performance ONNX edge model)
"""

import json
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

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = r"c:\Users\Admin\Leaf"
CLASS_JSON = os.path.join(BASE_DIR, "models", "class_names.json")
BEST_WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "crop_model_10class_best.pth")
ONNX_OUTPUT_PATH = os.path.join(BASE_DIR, "models", "crop_model_10class.onnx")

BATCH_SIZE = 32
NUM_EPOCHS = 5
LR = 0.001
IMG_SIZE = 224

with open(CLASS_JSON, "r") as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

print(f"[INFO] Initializing training for {NUM_CLASSES} classes: {CLASS_NAMES}")


def build_image_list():
    """Compiles exactly 1,000 image filepaths per class."""
    class_images = {cls: [] for cls in CLASS_NAMES}

    # 1. Disease folders from data/diseases/
    disease_folders = {
        "Anthracnose": os.path.join(BASE_DIR, "data", "diseases", "anthracnose"),
        "Black_rot": os.path.join(BASE_DIR, "data", "diseases", "black_rot"),
        "Downy_Mildew": os.path.join(BASE_DIR, "data", "diseases", "downy_mildew"),
        "Leaf_Spot_Blight": os.path.join(BASE_DIR, "data", "diseases", "leaf_spot_early_blight"),
        "Mosaic_Disease": os.path.join(BASE_DIR, "data", "diseases", "mosaic_virus"),
        "Powdery_Mildew": os.path.join(BASE_DIR, "data", "diseases", "powdery_mildew"),
        "Rust": os.path.join(BASE_DIR, "data", "diseases", "rust_fungus"),
    }
    for cls, path in disease_folders.items():
        if os.path.exists(path):
            files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            random.shuffle(files)
            class_images[cls] = files[:1000]

    # 2. Bacterial Leaf Spot from data/training/real
    real_train_dir = os.path.join(BASE_DIR, "data", "training", "real")
    bact_files = [os.path.join(real_train_dir, f) for f in os.listdir(real_train_dir)
                  if any(k in f.lower() for k in ['bact', 'b.spot', 'pv_peach', 'pv_pepper', 'pv_tomato'])]
    random.shuffle(bact_files)
    class_images["Bacterial_Leaf_Spot"] = bact_files[:1000]

    # 3. Healthy foliage (base + augmentations to 1,000)
    healthy_base = [os.path.join(real_train_dir, f) for f in os.listdir(real_train_dir) if 'healthy' in f.lower()]
    coll_real_dir = os.path.join(BASE_DIR, "data", "collected", "real")
    if os.path.exists(coll_real_dir):
        healthy_base += [os.path.join(coll_real_dir, f) for f in os.listdir(coll_real_dir) if 'healthy' in f.lower()]
    random.shuffle(healthy_base)
    healthy_imgs = list(healthy_base)
    while len(healthy_imgs) < 1000:
        healthy_imgs.append(random.choice(healthy_base))
    class_images["Healthy"] = healthy_imgs[:1000]

    # 4. Not_Leaf (negative controls & distractors)
    not_leaf_dir = os.path.join(BASE_DIR, "data", "training", "not_leaf")
    fake_dir = os.path.join(BASE_DIR, "data", "training", "fake")
    nl_files = [os.path.join(not_leaf_dir, f) for f in os.listdir(not_leaf_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(not_leaf_dir) else []
    if os.path.exists(fake_dir):
        nl_files += [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(nl_files)
    class_images["Not_Leaf"] = nl_files[:1000]

    dataset_samples = []
    for cls_name, paths in class_images.items():
        print(f"  {cls_name:22s}: {len(paths)} images")
        cls_idx = CLASS_TO_IDX[cls_name]
        for p in paths:
            dataset_samples.append((p, cls_idx))

    return dataset_samples


class CropLeafDataset(Dataset):
    def __init__(self, samples, is_train=True):
        self.samples = samples
        self.is_train = is_train

        # ImageNet normalization standard (RGB)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.80, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        except Exception:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        tensor = self.transform(img)
        return tensor, label


def train_model():
    samples = build_image_list()
    random.shuffle(samples)

    split_idx = int(len(samples) * 0.80)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"\n[INFO] Dataset split: {len(train_samples)} training samples, {len(val_samples)} validation samples")

    train_ds = CropLeafDataset(train_samples, is_train=True)
    val_ds = CropLeafDataset(val_samples, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    # Build Model
    print("\n[INFO] Loading pre-trained MobileNetV3-Small backbone...")
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)

    device = torch.device("cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    best_val_acc = 0.0
    start_time = time.time()

    print("\n" + "=" * 65)
    print("  STARTING MODEL TRAINING (5 EPOCHS)")
    print("=" * 65)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print(f"  Epoch [{epoch}/{NUM_EPOCHS}] Step [{step+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                val_correct += preds.eq(targets).sum().item()
                val_total += targets.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - t0

        print(f"--> Epoch {epoch:2d}/{NUM_EPOCHS} completed in {epoch_time:.1f}s | "
              f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% (Loss: {val_loss:.4f})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_WEIGHTS_PATH)
            print(f"    [*] Saved new best model checkpoint to {BEST_WEIGHTS_PATH} (Val Acc: {best_val_acc*100:.2f}%)")

    total_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"  TRAINING COMPLETE! (Total time: {total_time/60:.2f} min)")
    print(f"  Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print("=" * 65)

    # Export to ONNX
    print("\n[INFO] Exporting trained model to ONNX...")
    model.load_state_dict(torch.load(BEST_WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device="cpu")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_OUTPUT_PATH,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
    )

    size_mb = os.path.getsize(ONNX_OUTPUT_PATH) / 1024 / 1024
    print(f"[SUCCESS] Exported 10-class model to {ONNX_OUTPUT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    train_model()
