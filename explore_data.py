"""
explore_data.py

A quick look at the EuroSAT satellite images before training
anything: how many images exist per class, and what the images
actually look like.

Run download_data.py first, then run this file.

Saves 2 images:
  eda_class_distribution.png
  eda_sample_images.png
"""

import os
import matplotlib.pyplot as plt
from PIL import Image

DATA_DIR = os.path.join("eurosat_data", "2750")

if not os.path.isdir(DATA_DIR):
    print(f"Could not find {DATA_DIR}. Run download_data.py first.")
    raise SystemExit

classes = sorted(
    c for c in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, c))
)

counts = {}
for c in classes:
    class_path = os.path.join(DATA_DIR, c)
    counts[c] = len(os.listdir(class_path))

print("Images per class:")
for c, n in counts.items():
    print(f"  {c}: {n}")

# ---- class distribution chart ----
plt.figure(figsize=(9, 5))
plt.bar(counts.keys(), counts.values(), color="#4c72b0")
plt.title("EuroSAT: Number of Images per Class")
plt.ylabel("Number of images")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("eda_class_distribution.png")
plt.close()
print("\nSaved chart: eda_class_distribution.png")

# ---- sample image grid, one real satellite image per class ----
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for ax, c in zip(axes.flatten(), classes):
    class_path = os.path.join(DATA_DIR, c)
    sample_file = sorted(os.listdir(class_path))[0]
    img = Image.open(os.path.join(class_path, sample_file))
    ax.imshow(img)
    ax.set_title(c, fontsize=9)
    ax.axis("off")
plt.suptitle("A Real Sentinel-2 Satellite Image From Each Class")
plt.tight_layout()
plt.savefig("eda_sample_images.png")
plt.close()
print("Saved chart: eda_sample_images.png")

print("\nNext step: run train_cnn.py")
