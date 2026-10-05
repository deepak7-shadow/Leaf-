"""Download leaf dataset from Google Drive folder and organize into real/ and fake/ structure.

Usage:
    python download_dataset.py --url "https://drive.google.com/drive/folders/YOUR_FOLDER_ID"
    python download_dataset.py  # Uses the hardcoded URL below
"""

import argparse
import os
import shutil
import sys

DRIVE_URL = "https://drive.google.com/drive/folders/17PGeAZ21cPAuN79O3XFnmIWdQAB0IFnH?usp=drive_link"
OUTPUT_DIR = "data/downloaded"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Keyword heuristics to auto-detect real vs fake subfolders ─────────────────
REAL_KEYWORDS = {"real", "genuine", "live", "living", "natural", "original", "fresh", "true"}
FAKE_KEYWORDS = {"fake", "artificial", "printed", "plastic", "cloth", "fabric",
                 "paper", "synthetic", "false", "imitation", "manufactured"}


def classify_folder_name(name: str) -> str:
    """Return 'real', 'fake', or 'unknown' based on folder name keywords."""
    lower = name.lower()
    if any(k in lower for k in REAL_KEYWORDS):
        return "real"
    if any(k in lower for k in FAKE_KEYWORDS):
        return "fake"
    return "unknown"


def count_images(folder: str) -> int:
    return sum(
        1 for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    )


def organize_downloaded(raw_dir: str, out_dir: str):
    """Walk raw_dir, classify subfolders, and copy images into out_dir/real/ and out_dir/fake/."""
    real_out = os.path.join(out_dir, "real")
    fake_out = os.path.join(out_dir, "fake")
    os.makedirs(real_out, exist_ok=True)
    os.makedirs(fake_out, exist_ok=True)

    real_count = fake_count = unknown_count = 0

    for root, dirs, files in os.walk(raw_dir):
        folder_name = os.path.basename(root)
        category = classify_folder_name(folder_name)

        imgs = [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        if not imgs:
            continue

        if category == "unknown":
            print(f"  [?] Unknown category for folder '{folder_name}' ({len(imgs)} images) -- skipping auto-assign.")
            print(f"      Please manually move images to data/downloaded/real/ or data/downloaded/fake/")
            unknown_count += len(imgs)
            continue

        dest = real_out if category == "real" else fake_out
        for fname in imgs:
            src = os.path.join(root, fname)
            dst_name = f"{folder_name}_{fname}"
            shutil.copy2(src, os.path.join(dest, dst_name))

        if category == "real":
            real_count += len(imgs)
            print(f"  [REAL]  '{folder_name}': {len(imgs)} images")
        else:
            fake_count += len(imgs)
            print(f"  [FAKE]  '{folder_name}': {len(imgs)} images")

    print(f"\n  Total organized -> Real: {real_count}, Fake: {fake_count}, Unknown/Skipped: {unknown_count}")
    return real_count, fake_count


def download_folder(url: str, raw_dir: str):
    """Download Google Drive folder using gdown."""
    try:
        import gdown
    except ImportError:
        print("[ERROR] gdown not installed. Run: .venv/Scripts/pip install gdown")
        sys.exit(1)

    os.makedirs(raw_dir, exist_ok=True)
    print(f"\n[INFO] Downloading from Google Drive...")
    print(f"       URL: {url}")
    print(f"       -> {raw_dir}\n")

    try:
        gdown.download_folder(url=url, output=raw_dir, quiet=False, use_cookies=False)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("\n[TIP] Make sure the Google Drive folder is set to 'Anyone with the link' can view.")
        print("      Right-click folder -> Share -> General Access -> 'Anyone with the link'")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download & organize leaf dataset from Google Drive")
    parser.add_argument("--url", default=DRIVE_URL, help="Google Drive folder URL")
    parser.add_argument("--out", default=OUTPUT_DIR, help="Output directory for organized dataset")
    parser.add_argument("--raw-dir", default="data/raw_download", help="Temp dir for raw downloaded files")
    parser.add_argument("--skip-download", action="store_true", help="Skip download, just reorganize existing raw_dir")
    args = parser.parse_args()

    if not args.skip_download:
        download_folder(args.url, args.raw_dir)
    else:
        print(f"[INFO] Skipping download. Using existing: {args.raw_dir}")

    print(f"\n[INFO] Organizing dataset into real/ and fake/ structure...")
    real_n, fake_n = organize_downloaded(args.raw_dir, args.out)

    if real_n == 0 and fake_n == 0:
        print("\n[WARNING] No images were organized! Check folder names in your dataset.")
        print("          Folder names should contain keywords like 'real', 'fake', 'artificial', etc.")
        print(f"          Raw files are at: {args.raw_dir}")
        print(f"          Manually copy them to:")
        print(f"            {args.out}/real/")
        print(f"            {args.out}/fake/")
    else:
        print(f"\n[SUCCESS] Dataset ready at: {args.out}")
        print(f"          Run training with:")
        print(f"          .venv\\Scripts\\python.exe models/train.py --data-dir {args.out} --epochs 15")


if __name__ == "__main__":
    main()
