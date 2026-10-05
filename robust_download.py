"""Direct Google Drive downloader — lists files via Drive API and downloads all with retries.

Usage:
    python robust_download.py
"""

import os, sys, time, random, json, re, requests
from pathlib import Path

FOLDER_ID  = "17PGeAZ21cPAuN79O3XFnmIWdQAB0IFnH"
OUTPUT_DIR = Path("data/raw_download")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_RETRIES = 4
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})


# ── Folder listing via gdown ──────────────────────────────────────────────────

def list_drive_folder(folder_id):
    """List all files in a Google Drive folder using gdown's list endpoint."""
    import gdown
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    
    # Use gdown's file listing (works for public folders)
    try:
        files = gdown.download_folder(
            url=url,
            output=str(OUTPUT_DIR),
            quiet=False,
            use_cookies=False,
        )
        return files
    except Exception as e:
        return None


def get_folder_contents_api(folder_id):
    """Get folder contents using Google Drive's public folder listing."""
    # Use the Drive folder API (no auth needed for public folders)
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    
    # Fetch the folder page HTML and extract file info
    try:
        resp = SESSION.get(url, timeout=15)
        html = resp.text
        
        # Extract JSON data embedded in the page
        # Google Drive embeds file data as JSON in the page
        pattern = r'\["([^"]+)","([^"]+\.(?:jpg|jpeg|png|bmp|webp))"'
        matches = re.findall(pattern, html, re.IGNORECASE)
        
        files = []
        for file_id, name in matches:
            files.append({"id": file_id, "name": name})
        
        return files
    except Exception as e:
        print(f"[ERROR] Could not fetch folder contents: {e}")
        return []


def download_one(file_id, dest: Path, name: str) -> bool:
    """Download a single Drive file with retry + confirmation token handling."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if dest.exists() and dest.stat().st_size > 5000:
        return True  # Already downloaded
    
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = SESSION.get(url, stream=True, timeout=30, allow_redirects=True)
            
            # Handle large file confirmation token
            if "virus scan warning" in resp.text.lower() or "confirm" in resp.url.lower():
                # Extract confirmation token
                token_match = re.search(r'confirm=([0-9A-Za-z_\-]+)', resp.text)
                if token_match:
                    token = token_match.group(1)
                    url2 = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
                    resp = SESSION.get(url2, stream=True, timeout=30)
            
            if resp.status_code == 200:
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" in content_type and resp.status_code == 200:
                    # Likely a rate limit page, not the actual file
                    pass
                else:
                    with open(dest, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=32768):
                            f.write(chunk)
                    if dest.stat().st_size > 5000:
                        return True
            
        except Exception as e:
            pass
        
        if attempt < MAX_RETRIES:
            delay = 2 * (2 ** attempt) + random.uniform(0, 2)
            time.sleep(delay)
    
    return False


def run_gdown_download():
    """Run gdown download_folder which handles everything automatically."""
    import gdown
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
    
    print(f"[INFO] Downloading ALL files from Google Drive...")
    print(f"[INFO] Folder: {url}")
    print(f"[INFO] Output: {OUTPUT_DIR}")
    print(f"[INFO] This may take a while for large datasets...\n")
    
    try:
        # Try the newest gdown API
        result = gdown.download_folder(
            url=url,
            output=str(OUTPUT_DIR),
            quiet=False,
            use_cookies=False,
        )
        return result
    except TypeError:
        # Older gdown without some params
        try:
            result = gdown.download_folder(
                url=url,
                output=str(OUTPUT_DIR),
                quiet=False,
            )
            return result
        except Exception as e:
            print(f"[ERROR] gdown.download_folder failed: {e}")
            return None


def count_images():
    total = 0
    by_folder = {}
    for root, dirs, files in os.walk(OUTPUT_DIR):
        imgs = [f for f in files if Path(f).suffix.lower() in IMAGE_EXTS]
        if imgs:
            rel = os.path.relpath(root, OUTPUT_DIR)
            by_folder[rel] = len(imgs)
            total += len(imgs)
    return total, by_folder


def main():
    print("=" * 65)
    print("  GOOGLE DRIVE — FULL DATASET DOWNLOADER")
    print("=" * 65)
    
    # Run gdown
    result = run_gdown_download()
    
    # Count what we got
    total, by_folder = count_images()
    
    print(f"\n{'='*65}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"{'='*65}")
    print(f"  Total images downloaded: {total}")
    for folder, count in sorted(by_folder.items(), key=lambda x: -x[1]):
        print(f"    [{count:4d}]  {folder}")
    
    if total == 0:
        print("\n[ERROR] No images downloaded!")
        print("  The Drive folder may still be restricted.")
        print("  Please ensure sharing is set to 'Anyone with the link'.")
        sys.exit(1)
    
    print(f"\n[SUCCESS] {total} images ready in {OUTPUT_DIR}")
    print(f"\n[NEXT STEP] Organize into real/fake and retrain:")
    print(f"  python download_dataset.py --skip-download --out data/drive_dataset --raw-dir data/raw_download")


if __name__ == "__main__":
    main()
