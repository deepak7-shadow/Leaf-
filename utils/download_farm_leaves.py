"""Comprehensive downloader for ALL types of authentic real farm leaf images from PlantVillage.

Downloads real photographs across all 38 plant & disease categories in the PlantVillage
agricultural repository (14 crop species, healthy + disease conditions).
Validates each image with OpenCV and records full provenance metadata in manifest.json.
"""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
import cv2
import numpy as np


# Full catalog of ALL 38 real leaf categories with Git tree SHAs
ALL_CATEGORIES: Dict[str, Dict[str, str]] = {
    "Apple___Apple_scab": {
        "sha": "920c7a3e9a65ad0612349d39da0ff1a8a597e03a",
        "crop": "apple",
        "condition": "apple_scab",
        "crop_display": "Apple",
        "condition_display": "Apple Scab (Venturia inaequalis)",
        "type_category": "fungal_disease"
    },
    "Apple___Black_rot": {
        "sha": "d5cf0b76bc72ca6fffa3b83367ded4409b1b24ae",
        "crop": "apple",
        "condition": "black_rot",
        "crop_display": "Apple",
        "condition_display": "Black Rot (Diplodia seriata)",
        "type_category": "fungal_disease"
    },
    "Apple___Cedar_apple_rust": {
        "sha": "22a5a625ee980bd7c3d10c57ac90e429021236cc",
        "crop": "apple",
        "condition": "cedar_rust",
        "crop_display": "Apple",
        "condition_display": "Cedar Apple Rust (Gymnosporangium)",
        "type_category": "fungal_disease"
    },
    "Apple___healthy": {
        "sha": "5526391f930051a87fc0345152164aec7a48056c",
        "crop": "apple",
        "condition": "healthy",
        "crop_display": "Apple",
        "condition_display": "Healthy Living Foliage",
        "type_category": "healthy"
    },
    "Blueberry___healthy": {
        "sha": "794cd233204b52278a220a2f991332fe122e280c",
        "crop": "blueberry",
        "condition": "healthy",
        "crop_display": "Blueberry",
        "condition_display": "Healthy Living Foliage",
        "type_category": "healthy"
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "sha": "09d9eb5a43eaae40683025f12caca6e9d23db287",
        "crop": "cherry",
        "condition": "powdery_mildew",
        "crop_display": "Cherry",
        "condition_display": "Powdery Mildew (Podosphaera)",
        "type_category": "fungal_disease"
    },
    "Cherry_(including_sour)___healthy": {
        "sha": "6a8780145cd6aac767928e3fb2cae2bcebfc9aef",
        "crop": "cherry",
        "condition": "healthy",
        "crop_display": "Cherry",
        "condition_display": "Healthy Living Foliage",
        "type_category": "healthy"
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "sha": "8cc6b2d2ac32539c02a9bd0e39e21062a9536fcc",
        "crop": "corn",
        "condition": "cercospora_leaf_spot",
        "crop_display": "Corn (Maize)",
        "condition_display": "Cercospora Gray Leaf Spot",
        "type_category": "fungal_disease"
    },
    "Corn_(maize)___Common_rust_": {
        "sha": "4b67a1bb45178cf251153251f911037505bf57a0",
        "crop": "corn",
        "condition": "common_rust",
        "crop_display": "Corn (Maize)",
        "condition_display": "Common Rust (Puccinia sorghi)",
        "type_category": "fungal_disease"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "sha": "d9b2c8be02be845df7989102b54124c8034eea3c",
        "crop": "corn",
        "condition": "northern_leaf_blight",
        "crop_display": "Corn (Maize)",
        "condition_display": "Northern Leaf Blight (Exserohilum)",
        "type_category": "fungal_disease"
    },
    "Corn_(maize)___healthy": {
        "sha": "08567ffe34691947d9604c6587d6d777dffa7c4f",
        "crop": "corn",
        "condition": "healthy",
        "crop_display": "Corn (Maize)",
        "condition_display": "Healthy Living Foliage",
        "type_category": "healthy"
    },
    "Grape___Black_rot": {
        "sha": "e14754576fcc879db960a30326a87df11c0007b7",
        "crop": "grape",
        "condition": "black_rot",
        "crop_display": "Grape",
        "condition_display": "Black Rot (Guignardia bidwellii)",
        "type_category": "fungal_disease"
    },
    "Grape___Esca_(Black_Measles)": {
        "sha": "c02c7fecf9bdeb8f100334a278a68d79606786b6",
        "crop": "grape",
        "condition": "esca_black_measles",
        "crop_display": "Grape",
        "condition_display": "Esca / Black Measles",
        "type_category": "fungal_disease"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "sha": "32fd28ead49852cddc05f9322f7eb2ad775303dd",
        "crop": "grape",
        "condition": "leaf_blight",
        "crop_display": "Grape",
        "condition_display": "Isariopsis Leaf Blight",
        "type_category": "fungal_disease"
    },
    "Grape___healthy": {
        "sha": "673d3f9faf6b29229f728fdd82d4a0e011c8db6b",
        "crop": "grape",
        "condition": "healthy",
        "crop_display": "Grape",
        "condition_display": "Healthy Living Foliage",
        "type_category": "healthy"
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "sha": "c65ba0b5e33eb61b16e4b1082c318ecda3de18ac",
        "crop": "orange",
        "condition": "citrus_greening",
        "crop_display": "Orange (Citrus)",
        "condition_display": "Huanglongbing (Citrus Greening)",
        "type_category": "bacterial_disease"
    },
    "Peach___Bacterial_spot": {
        "sha": "108b64af49cee885331bbc19a1e5f8fd6ba1fbc8",
        "crop": "peach",
        "condition": "bacterial_spot",
        "crop_display": "Peach",
        "condition_display": "Bacterial Spot (Xanthomonas)",
        "type_category": "bacterial_disease"
    },
    "Peach___healthy": {
        "sha": "84a66107fd0ca7e3367a84e5005a4cff61bc164c",
        "crop": "peach",
        "condition": "healthy",
        "crop_display": "Peach",
        "condition_display": "Healthy Living Foliage",
        "type_category": "healthy"
    },
    "Pepper,_bell___Bacterial_spot": {
        "sha": "aefbf8c7abfa589de5404378f7c470cb650cc091",
        "crop": "bellpepper",
        "condition": "bacterial_spot",
        "crop_display": "Bell Pepper",
        "condition_display": "Bacterial Spot (Xanthomonas euvesicatoria)",
        "type_category": "bacterial_disease"
    },
    "Pepper,_bell___healthy": {
        "sha": "e17baeac0d0dc73d48eda5d6fe53cdb228bd532d",
        "crop": "bellpepper",
        "condition": "healthy",
        "crop_display": "Bell Pepper",
        "condition_display": "Healthy Living Foliage",
        "type_category": "healthy"
    },
    "Potato___Early_blight": {
        "sha": "6a05d8fce92f7436da5bca72704ba5596a544e96",
        "crop": "potato",
        "condition": "early_blight",
        "crop_display": "Potato",
        "condition_display": "Early Blight (Alternaria solani)",
        "type_category": "fungal_disease"
    },
    "Potato___Late_blight": {
        "sha": "32d92d88e9c267a3a3d92e2167a96148d5197e73",
        "crop": "potato",
        "condition": "late_blight",
        "crop_display": "Potato",
        "condition_display": "Late Blight (Phytophthora infestans)",
        "type_category": "fungal_disease"
    },
    "Potato___healthy": {
        "sha": "9e73447b5d9bea5bcfd5748ae32fb9b0e9babbf0",
        "crop": "potato",
        "condition": "healthy",
        "crop_display": "Potato",
        "condition_display": "Healthy Living Foliage",
        "type_category": "healthy"
    },
    "Raspberry___healthy": {
        "sha": "2bfb25cb9cddb1a3eb2c3fe74c9af39ef1ce606f",
        "crop": "raspberry",
        "condition": "healthy",
        "crop_display": "Raspberry",
        "condition_display": "Healthy Living Foliage",
        "type_category": "healthy"
    },
    "Soybean___healthy": {
        "sha": "8025cb9ca3d2a54189ff0f80685a7a4ba7f9ddc8",
        "crop": "soybean",
        "condition": "healthy",
        "crop_display": "Soybean",
        "condition_display": "Healthy Living Foliage",
        "type_category": "healthy"
    },
    "Squash___Powdery_mildew": {
        "sha": "19b789e0dab8edab2de8978d4842e66c244d82ac",
        "crop": "squash",
        "condition": "powdery_mildew",
        "crop_display": "Squash",
        "condition_display": "Powdery Mildew (Podosphaera)",
        "type_category": "fungal_disease"
    },
    "Strawberry___Leaf_scorch": {
        "sha": "3a87fdd4520efd65785dd40e38989d09118a8d3b",
        "crop": "strawberry",
        "condition": "leaf_scorch",
        "crop_display": "Strawberry",
        "condition_display": "Leaf Scorch (Diplocarpon earlianum)",
        "type_category": "fungal_disease"
    },
    "Strawberry___healthy": {
        "sha": "c0f16b04dd9e1aab4a35883ed7c75ad2ab821084",
        "crop": "strawberry",
        "condition": "healthy",
        "crop_display": "Strawberry",
        "condition_display": "Healthy Living Foliage",
        "type_category": "healthy"
    },
    "Tomato___Bacterial_spot": {
        "sha": "9e1348595d4ea249555a97826dcec026b6a4349f",
        "crop": "tomato",
        "condition": "bacterial_spot",
        "crop_display": "Tomato",
        "condition_display": "Bacterial Spot (Xanthomonas)",
        "type_category": "bacterial_disease"
    },
    "Tomato___Early_blight": {
        "sha": "02df6aeaae1e244b176282be814c3e025e4c82d2",
        "crop": "tomato",
        "condition": "early_blight",
        "crop_display": "Tomato",
        "condition_display": "Early Blight (Alternaria solani)",
        "type_category": "fungal_disease"
    },
    "Tomato___Late_blight": {
        "sha": "8e1c2eabde01d6ddeb1943799d3c552d6e2a47a5",
        "crop": "tomato",
        "condition": "late_blight",
        "crop_display": "Tomato",
        "condition_display": "Late Blight (Phytophthora infestans)",
        "type_category": "fungal_disease"
    },
    "Tomato___Leaf_Mold": {
        "sha": "28b43a2f05128e5714a43d488073842bb062ada1",
        "crop": "tomato",
        "condition": "leaf_mold",
        "crop_display": "Tomato",
        "condition_display": "Leaf Mold (Passalora fulva)",
        "type_category": "fungal_disease"
    },
    "Tomato___Septoria_leaf_spot": {
        "sha": "61f13f9116d32c2e33b7ba4b75affd183c6da5a3",
        "crop": "tomato",
        "condition": "septoria_leaf_spot",
        "crop_display": "Tomato",
        "condition_display": "Septoria Leaf Spot",
        "type_category": "fungal_disease"
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "sha": "a65e469669fe7773edc9d8e87f9c737595612cbe",
        "crop": "tomato",
        "condition": "spider_mites",
        "crop_display": "Tomato",
        "condition_display": "Two-Spotted Spider Mites (Tetranychus urticae)",
        "type_category": "pest_damage"
    },
    "Tomato___Target_Spot": {
        "sha": "876ca4c3521119b7cd65d94d5158022662bc9d3a",
        "crop": "tomato",
        "condition": "target_spot",
        "crop_display": "Tomato",
        "condition_display": "Target Spot (Corynespora cassiicola)",
        "type_category": "fungal_disease"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "sha": "2748706acf606a1c0efc91f196f71428ac858db4",
        "crop": "tomato",
        "condition": "curl_virus",
        "crop_display": "Tomato",
        "condition_display": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "type_category": "viral_disease"
    },
    "Tomato___Tomato_mosaic_virus": {
        "sha": "d1f856e176fdfa7bfcf85d5286c78c521814123a",
        "crop": "tomato",
        "condition": "mosaic_virus",
        "crop_display": "Tomato",
        "condition_display": "Tomato Mosaic Virus (ToMV)",
        "type_category": "viral_disease"
    },
    "Tomato___healthy": {
        "sha": "570719429980b269e772b9782ce33d225ced66c8",
        "crop": "tomato",
        "condition": "healthy",
        "crop_display": "Tomato",
        "condition_display": "Healthy Living Foliage",
        "type_category": "healthy"
    }
}

CACHE_FILE = os.path.join(os.path.dirname(__file__), "category_trees.json")
RAW_BASE_URL = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color"


def fetch_category_file_lists() -> Dict[str, List[str]]:
    """Fetch filenames for all 38 categories using Git Tree API and cache locally."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if len(cached) == len(ALL_CATEGORIES):
                print(f"[CACHE] Loaded file lists for {len(cached)} categories from {CACHE_FILE}")
                return cached
        except Exception:
            pass

    print("[INFO] Fetching category file trees from GitHub API...")
    category_files: Dict[str, List[str]] = {}

    def fetch_single_category(cat_name: str, meta: Dict[str, str]) -> Tuple[str, List[str]]:
        sha = meta["sha"]
        url = f"https://api.github.com/repos/spMohanty/PlantVillage-Dataset/git/trees/{sha}"
        req = urllib.request.Request(url, headers={"User-Agent": "AI-Nozzle-Dataset-Fetcher/2.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            files = [
                x["path"] for x in data.get("tree", [])
                if x.get("path", "").lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            return cat_name, files

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(fetch_single_category, cat, meta)
            for cat, meta in ALL_CATEGORIES.items()
        ]
        for fut in as_completed(futures):
            cat_name, files = fut.result()
            category_files[cat_name] = files
            print(f"  -> Category '{cat_name}': {len(files)} files found")

    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(category_files, f, indent=2)
        print(f"[CACHE] Saved {len(category_files)} category file trees to {CACHE_FILE}")
    except Exception as e:
        print(f"[WARNING] Could not save cache: {e}")

    return category_files


def download_single_image(
    category_name: str,
    original_filename: str,
    target_filename: str,
    output_dir: str,
    meta: Dict[str, str]
) -> Dict:
    """Download single image from GitHub raw, decode, validate with OpenCV, and save."""
    target_path = os.path.join(output_dir, target_filename)

    # If already exists and valid, reuse
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1024:
        img = cv2.imread(target_path)
        if img is not None and img.shape[2] == 3:
            h, w, c = img.shape
            encoded_fn = urllib.parse.quote(original_filename)
            raw_url = f"{RAW_BASE_URL}/{urllib.parse.quote(category_name)}/{encoded_fn}"
            return {
                "filename": target_filename,
                "crop": meta["crop"],
                "condition": meta["condition"],
                "crop_display": meta["crop_display"],
                "condition_display": meta["condition_display"],
                "type_category": meta["type_category"],
                "original_name": original_filename,
                "download_url": raw_url,
                "width": w,
                "height": h,
                "channels": c,
                "source": "PlantVillage (Penn State / EPFL Open Agricultural Dataset)",
                "type": "real_living_farm_leaf"
            }

    encoded_fn = urllib.parse.quote(original_filename)
    raw_url = f"{RAW_BASE_URL}/{urllib.parse.quote(category_name)}/{encoded_fn}"
    req = urllib.request.Request(raw_url, headers={"User-Agent": "AI-Nozzle-Dataset-Fetcher/2.0"})

    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read()

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError(f"Failed to decode image from {raw_url}")

    h, w, c = img.shape
    if c != 3 or h < 64 or w < 64:
        raise ValueError(f"Invalid image dimensions: shape={img.shape} from {raw_url}")

    cv2.imwrite(target_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return {
        "filename": target_filename,
        "crop": meta["crop"],
        "condition": meta["condition"],
        "crop_display": meta["crop_display"],
        "condition_display": meta["condition_display"],
        "type_category": meta["type_category"],
        "original_name": original_filename,
        "download_url": raw_url,
        "width": w,
        "height": h,
        "channels": c,
        "source": "PlantVillage (Penn State / EPFL Open Agricultural Dataset)",
        "type": "real_living_farm_leaf"
    }


def download_all_farm_leaf_types(
    output_dir: str = "data/collected/real",
    samples_per_category: int = 10,
    max_workers: int = 12
) -> List[Dict]:
    """Download authentic photographs covering ALL 38 real leaf categories."""
    os.makedirs(output_dir, exist_ok=True)
    category_files = fetch_category_file_lists()

    tasks = []
    for cat_name, meta in ALL_CATEGORIES.items():
        files = category_files.get(cat_name, [])
        if not files:
            print(f"[WARNING] No files found for category {cat_name}")
            continue

        selected = files[:samples_per_category]
        for i, original_fn in enumerate(selected, start=1):
            target_fn = f"farm_{meta['crop']}_{meta['condition']}_{i:03d}.jpg"
            tasks.append((cat_name, original_fn, target_fn, meta))

    total_tasks = len(tasks)
    print(f"\n[INFO] Starting download and validation of {total_tasks} real farm leaf images (All 38 types)...")
    manifest_records: List[Dict] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                download_single_image, cat, orig_fn, target_fn, output_dir, meta
            ): target_fn
            for cat, orig_fn, target_fn, meta in tasks
        }

        for future in as_completed(future_map):
            target_fn = future_map[future]
            try:
                record = future.result()
                manifest_records.append(record)
                completed += 1
                if completed % 25 == 0 or completed == total_tasks:
                    print(f"  -> Downloaded & validated {completed}/{total_tasks} real leaf images...")
            except Exception as exc:
                print(f"[WARNING] Failed {target_fn}: {exc}")

    # Also include any previously downloaded legacy samples in manifest if present
    for fname in os.listdir(output_dir):
        if fname.startswith("farm_") and fname.endswith(".jpg"):
            if not any(r["filename"] == fname for r in manifest_records):
                fpath = os.path.join(output_dir, fname)
                img = cv2.imread(fpath)
                if img is not None:
                    h, w, c = img.shape
                    # Parse legacy name: farm_<crop>_<idx>.jpg
                    parts = fname.replace(".jpg", "").split("_")
                    crop_name = parts[1] if len(parts) > 1 else "unknown"
                    manifest_records.append({
                        "filename": fname,
                        "crop": crop_name,
                        "condition": "healthy",
                        "crop_display": crop_name.capitalize(),
                        "condition_display": "Healthy Living Foliage",
                        "type_category": "healthy",
                        "original_name": fname,
                        "download_url": "",
                        "width": w,
                        "height": h,
                        "channels": c,
                        "source": "PlantVillage (Penn State / EPFL Open Agricultural Dataset)",
                        "type": "real_living_farm_leaf"
                    })

    # Generate comprehensive manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    crops_summary: Dict[str, int] = {}
    types_summary: Dict[str, int] = {}
    categories_summary: Dict[str, int] = {}

    for r in manifest_records:
        crops_summary[r["crop"]] = crops_summary.get(r["crop"], 0) + 1
        types_summary[r["type_category"]] = types_summary.get(r["type_category"], 0) + 1
        cat_key = f"{r['crop_display']} - {r['condition_display']}"
        categories_summary[cat_key] = categories_summary.get(cat_key, 0) + 1

    manifest_data = {
        "dataset_name": "AI Nozzle Comprehensive Real Farm Leaves Dataset",
        "description": "Authentic, non-AI photographs of living crop leaves covering all 38 categories from real agricultural farms.",
        "source": "Penn State / EPFL PlantVillage Open Agricultural Research Dataset",
        "download_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_images": len(manifest_records),
        "total_unique_crops": len(crops_summary),
        "total_categories_covered": len(categories_summary),
        "condition_types_summary": types_summary,
        "crops_breakdown": crops_summary,
        "categories_breakdown": categories_summary,
        "images": sorted(manifest_records, key=lambda x: x["filename"])
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print("\n" + "=" * 75)
    print(f"[SUCCESS] Total real farm leaf images in dataset: {len(manifest_records)}")
    print(f"[SUMMARY] Crops covered ({len(crops_summary)}): {', '.join(sorted(crops_summary.keys()))}")
    print(f"[SUMMARY] Unique categories covered: {len(categories_summary)}/38")
    print(" Condition types:")
    for ctype, cnt in types_summary.items():
        print(f"   - {ctype.replace('_', ' ').title():20s}: {cnt} images")
    print(f"[MANIFEST] Updated manifest saved to: {manifest_path}")
    print("=" * 75)

    return manifest_records


if __name__ == "__main__":
    out = "data/collected/real"
    samples = 10
    if len(sys.argv) > 1:
        out = sys.argv[1]
    if len(sys.argv) > 2:
        samples = int(sys.argv[2])
    download_all_farm_leaf_types(output_dir=out, samples_per_category=samples)
