"""Organizes raw downloaded Drive data into real/fake training structure,
then kicks off model retraining automatically.

Run this AFTER your terminal download completes:
    python organize_and_train.py

It will:
  1. Scan data/raw_download for all images
  2. Sort them into real/ and fake/ based on folder name keywords
  3. Ask you about unknown folders
  4. Merge with existing collected data
  5. Retrain the model
"""

import os, sys, shutil, subprocess
from pathlib import Path

RAW_DIR      = Path("data/raw_download")
TRAINING_DIR = Path("data/training")
COLLECTED    = Path("data/collected")
MODEL_OUT    = "models/mobilenet_v3_small.onnx"
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Keyword rules ─────────────────────────────────────────────────────────────
REAL_KEYWORDS = {
    "real", "genuine", "live", "living", "natural", "original",
    "fresh", "true", "plant", "leaf", "leaves", "garden", "farm",
    "vegetation", "green", "organic", "pexels"          # pexels = real photos
}
FAKE_KEYWORDS = {
    "fake", "artificial", "printed", "plastic", "cloth", "fabric",
    "paper", "synthetic", "false", "imitation", "manufactured",
    "empty", "background", "difficult", "machinery", "tractor",
    "parts", "agricultural"
}


def classify_folder(name: str) -> str:
    lower = name.lower()
    if any(k in lower for k in REAL_KEYWORDS):
        return "real"
    if any(k in lower for k in FAKE_KEYWORDS):
        return "fake"
    return "unknown"


def scan_raw():
    """Scan raw_download and classify each folder."""
    folder_map = {}  # folder_path -> (category, [image_paths])
    
    for root, dirs, files in os.walk(RAW_DIR):
        imgs = [Path(root) / f for f in files
                if Path(f).suffix.lower() in IMAGE_EXTS]
        if not imgs:
            continue
        folder_name = os.path.basename(root)
        category = classify_folder(folder_name)
        rel = os.path.relpath(root, RAW_DIR)
        folder_map[rel] = (category, imgs)
    
    return folder_map


def organize(folder_map, interactive=True):
    """Copy images into training/real and training/fake."""
    (TRAINING_DIR / "real").mkdir(parents=True, exist_ok=True)
    (TRAINING_DIR / "fake").mkdir(parents=True, exist_ok=True)

    real_n = fake_n = skip_n = 0

    for folder_rel, (cat, imgs) in sorted(folder_map.items()):
        if cat == "unknown" and interactive:
            print(f"\n  [?] Unknown folder: '{folder_rel}' ({len(imgs)} images)")
            print(f"      Sample: {imgs[0].name}")
            ans = input("      Is this [r]eal, [f]ake, or [s]kip? ").strip().lower()
            cat = "real" if ans.startswith("r") else ("fake" if ans.startswith("f") else "skip")

        if cat == "skip" or cat == "unknown":
            print(f"  [SKIP]  '{folder_rel}': {len(imgs)} images")
            skip_n += len(imgs)
            continue

        dest = TRAINING_DIR / cat
        label = "REAL" if cat == "real" else "FAKE"
        for img in imgs:
            dst_name = f"drive_{folder_rel.replace(os.sep,'_')}_{img.name}"
            shutil.copy2(img, dest / dst_name)

        print(f"  [{label:4s}]  '{folder_rel}': {len(imgs)} images -> training/{cat}/")
        if cat == "real":
            real_n += len(imgs)
        else:
            fake_n += len(imgs)

    return real_n, fake_n


def add_collected():
    """Ensure collected/ data is also in training/."""
    added_real = added_fake = 0
    for src, dst_cat in [(COLLECTED / "real", "real"), (COLLECTED / "fake", "fake")]:
        if not src.exists():
            continue
        dst = TRAINING_DIR / dst_cat
        for f in src.iterdir():
            if f.suffix.lower() in IMAGE_EXTS:
                dest = dst / f"collected_{f.name}"
                if not dest.exists():
                    shutil.copy2(f, dest)
                    if dst_cat == "real": added_real += 1
                    else: added_fake += 1
    return added_real, added_fake


def count_training():
    real = len([f for f in (TRAINING_DIR/"real").iterdir()
                if f.suffix.lower() in IMAGE_EXTS])
    fake = len([f for f in (TRAINING_DIR/"fake").iterdir()
                if f.suffix.lower() in IMAGE_EXTS])
    return real, fake


def main():
    print("=" * 65)
    print("  ORGANIZE + RETRAIN PIPELINE")
    print("=" * 65)

    # Check raw_download exists
    if not RAW_DIR.exists():
        print(f"[ERROR] {RAW_DIR} not found. Run the download first.")
        sys.exit(1)

    # Scan
    folder_map = scan_raw()
    total_raw = sum(len(imgs) for _, imgs in folder_map.values())
    print(f"\n[STEP 1] Found {total_raw} images in {len(folder_map)} folders:\n")
    for folder_rel, (cat, imgs) in sorted(folder_map.items()):
        tag = f"[{cat.upper():7s}]" if cat != "unknown" else "[  ???  ]"
        print(f"  {tag}  {folder_rel}: {len(imgs)} images")

    if total_raw == 0:
        print("\n[WARNING] No images found in data/raw_download!")
        print("  Make sure your terminal download completed successfully.")
        sys.exit(1)

    # Organize
    print(f"\n[STEP 2] Organizing into training/real and training/fake...")
    real_n, fake_n = organize(folder_map, interactive=sys.stdin.isatty())

    # Merge with collected
    print(f"\n[STEP 3] Merging with existing collected data...")
    ar, af = add_collected()
    print(f"  Added {ar} real + {af} fake from data/collected/")

    # Final count
    real_total, fake_total = count_training()
    total = real_total + fake_total
    print(f"\n[STEP 4] Final training dataset:")
    print(f"  Real:  {real_total}")
    print(f"  Fake:  {fake_total}")
    print(f"  Total: {total}")

    if total < 10:
        print("[ERROR] Too few images to train. Check your dataset.")
        sys.exit(1)

    # Retrain
    print(f"\n[STEP 5] Starting training (15 epochs)...\n")
    cmd = [
        sys.executable, "models/train.py",
        "--data-dir", str(TRAINING_DIR),
        "--epochs", "20",
        "--batch-size", "16",
        "--lr", "5e-5",
    ]
    subprocess.run(cmd, check=True)

    print(f"\n[DONE] Model saved to {MODEL_OUT}")
    print(f"       Launch camera: python camera_test.py")


if __name__ == "__main__":
    main()
