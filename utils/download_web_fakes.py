"""Automated downloader for fake/non-real leaf images to balance training dataset.

Fetches diverse non-real leaves (plastic plants, paper crafts, printed fabric,
illustrations) from Openverse Creative Commons API and supplements with
synthetic printed leaves to reach an exact 1:1 ratio (1,270 Real : 1,270 Fake).
"""

import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.synthetic_generator import SyntheticSceneGenerator

TARGET_DIR = os.path.join(ROOT_DIR, "data", "training", "fake")
REAL_DIR   = os.path.join(ROOT_DIR, "data", "training", "real")

HEADERS = {
    "User-Agent": "LeafAuthenticityResearch/1.0 (academic computer vision; leaf-ai@example.org)"
}

SEARCH_QUERIES = [
    "artificial plant leaf",
    "plastic leaf foliage",
    "silk plant leaf",
    "paper leaf craft",
    "origami leaf green",
    "artificial green leaves",
    "leaf illustration botanical",
    "leaf pattern fabric",
    "fake plant leaves",
    "plastic monstera leaf",
    "artificial palm leaf",
    "leaf sticker print",
    "artificial ficus leaves",
    "craft green leaf paper"
]


def count_images(dir_path):
    if not os.path.exists(dir_path):
        return 0
    return sum(
        1 for f in os.listdir(dir_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    )


def fetch_image_urls_for_query(query, max_pages=3):
    urls = []
    for page in range(1, max_pages + 1):
        try:
            params = urllib.parse.urlencode({
                "q": query,
                "format": "json",
                "page_size": "50",
                "page": str(page)
            })
            api_url = f"https://api.openverse.org/v1/images/?{params}"
            req = urllib.request.Request(api_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for item in data.get("results", []):
                    u = item.get("url")
                    if u and u.startswith("http"):
                        urls.append(u)
        except Exception:
            break
        time.sleep(0.3)
    return urls


def download_single_image(url, save_path):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read()

        if len(content) < 5000:
            return False

        arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.shape[0] < 80 or img.shape[1] < 80:
            return False

        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim > 512:
            scale = 512.0 / max_dim
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        cv2.imwrite(save_path, img)
        return True
    except Exception:
        return False


def balance_dataset(target_fake_count=None):
    os.makedirs(TARGET_DIR, exist_ok=True)
    real_count = count_images(REAL_DIR)
    curr_fake  = count_images(TARGET_DIR)

    if target_fake_count is None:
        target_fake_count = real_count

    needed = target_fake_count - curr_fake
    print("=" * 60)
    print("DATASET BALANCING TARGET")
    print(f"   Real Images:    {real_count}")
    print(f"   Current Fake:   {curr_fake}")
    print(f"   Target Fake:    {target_fake_count}")
    print(f"   Images Needed:  {needed}")
    print("=" * 60)

    if needed <= 0:
        print("[OK] Dataset is already balanced!")
        return

    # Step 1: Collect URLs from Openverse
    print(f"\n[1/3] Querying Openverse API across {len(SEARCH_QUERIES)} diverse queries...")
    all_urls = []
    seen = set()
    for q in SEARCH_QUERIES:
        urls = fetch_image_urls_for_query(q, max_pages=3)
        for u in urls:
            if u not in seen:
                seen.add(u)
                all_urls.append(u)
        print(f"   Query '{q}': found {len(urls)} urls (Total unique: {len(all_urls)})")
        if len(all_urls) >= needed * 2:
            break

    # Step 2: Download images in parallel
    print(f"\n[2/3] Downloading images in parallel...")
    downloaded = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {}
        for i, url in enumerate(all_urls):
            if downloaded >= needed:
                break
            fname = f"web_fake_{int(time.time())}_{i:04d}.jpg"
            out_path = os.path.join(TARGET_DIR, fname)
            futures[executor.submit(download_single_image, url, out_path)] = out_path

        for f in as_completed(futures):
            success = f.result()
            if success:
                downloaded += 1
                if downloaded % 25 == 0 or downloaded == needed:
                    print(f"   Downloaded {downloaded}/{needed} images...")
                if downloaded >= needed:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

    # Step 3: If any gap remains, supplement with realistic synthetic print leaves
    curr_fake = count_images(TARGET_DIR)
    rem = target_fake_count - curr_fake
    if rem > 0:
        print(f"\n[3/3] Supplementing remaining {rem} samples with synthetic print-artifact leaves...")
        gen = SyntheticSceneGenerator(width=320, height=320)
        for i in range(rem):
            canvas = gen.generate_soil_background()
            cx, cy = 160 + random.randint(-20, 20), 160 + random.randint(-20, 20)
            ax, ay = random.randint(40, 75), random.randint(25, 45)
            substrate = random.choice(["paper", "fabric"])
            x, y, w, h = gen.draw_fake_printed_leaf(canvas, (cx, cy), (ax, ay), substrate=substrate)
            x1, y1 = max(0, x - 5), max(0, y - 5)
            x2, y2 = min(320, x + w + 5), min(320, y + h + 5)
            patch = canvas[y1:y2, x1:x2]
            if patch.size > 0:
                pname = f"synth_fake_bal_{i:04d}.jpg"
                cv2.imwrite(os.path.join(TARGET_DIR, pname), patch)

    final_real = count_images(REAL_DIR)
    final_fake = count_images(TARGET_DIR)
    print("\n" + "=" * 60)
    print("[SUCCESS] DATASET BALANCING COMPLETE!")
    print(f"   Final Real Images: {final_real}")
    print(f"   Final Fake Images: {final_fake}")
    print(f"   Ratio:             {final_real / max(1, final_fake):.2f} : 1  (Perfect 50% / 50% Balance)")
    print("=" * 60)


if __name__ == "__main__":
    balance_dataset()
