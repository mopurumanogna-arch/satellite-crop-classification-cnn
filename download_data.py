"""
download_data.py

Downloads the REAL EuroSAT dataset: actual Sentinel-2 satellite
images covering 10 land-cover classes, including Annual Crop and
Permanent Crop. This is a well known dataset used in real remote
sensing research (Helber et al., 2017).

This is roughly a 90 MB download. Run this file first - it needs
an internet connection. It creates a folder called eurosat_data/2750
with one subfolder per class, each full of real satellite images.
"""

import requests
import zipfile
import os

URL = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
ZIP_PATH = "EuroSAT.zip"
EXTRACT_DIR = "eurosat_data"

print("Downloading EuroSAT satellite image dataset (~90 MB)...")
print("This can take a few minutes depending on your internet speed.\n")

try:
    with requests.get(URL, stream=True, timeout=180) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = downloaded / total * 100
                    print(
                        f"\rDownloaded {downloaded // (1024*1024)} MB / "
                        f"{total // (1024*1024)} MB ({percent:.0f}%)",
                        end="",
                    )
except requests.exceptions.RequestException as e:
    print(f"\nDownload failed: {e}")
    raise SystemExit

print("\nDownload complete. Extracting...")

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print(f"Extracted to {EXTRACT_DIR}/")

# sanity check: list classes and image counts
data_dir = os.path.join(EXTRACT_DIR, "2750")
if os.path.isdir(data_dir):
    classes = sorted(
        c for c in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, c))
    )
    print(f"\nFound {len(classes)} classes:")
    for c in classes:
        n_images = len(os.listdir(os.path.join(data_dir, c)))
        print(f"  {c}: {n_images} images")
else:
    print(
        "\nCould not find the expected '2750' folder after extracting. "
        "Check the contents of the eurosat_data folder manually."
    )

print("\nNext step: run explore_data.py")
