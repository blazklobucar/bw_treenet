import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from osgeo import gdal

gdal.UseExceptions()

img_dir = os.path.expanduser("~/bw_treenet/data/processed/malmo/tiles/images")
lbl_dir = os.path.expanduser("~/bw_treenet/data/processed/malmo/tiles/labels")

files = sorted([f for f in os.listdir(img_dir) if f.endswith(".tif")])
sample = files[len(files)//2]
print(f"Checking tile: {sample}")

img_ds = gdal.Open(os.path.join(img_dir, sample))
img = img_ds.GetRasterBand(1).ReadAsArray()
img_ds = None

lbl_ds = gdal.Open(os.path.join(lbl_dir, sample))
lbl = lbl_ds.GetRasterBand(1).ReadAsArray()
lbl_ds = None

print(f"Image shape: {img.shape}  min={img.min()}  max={img.max()}")
print(f"Label shape: {lbl.shape}  unique values: {np.unique(lbl)}")
print(f"Tree fraction: {lbl.sum() / (1000*1000):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Grayscale image")
axes[1].imshow(img, cmap="gray")
axes[1].imshow(lbl, cmap="Greens", alpha=0.5)
axes[1].set_title("Image + canopy overlay")
plt.tight_layout()
out = os.path.expanduser("~/bw_treenet/results/tile_check.png")
plt.savefig(out, dpi=100)
print(f"Saved to {out}")