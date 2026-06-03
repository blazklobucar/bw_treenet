import os
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

LABELS_DIR = os.path.expanduser(
    "~/bw_treenet/data/processed/malmo/tiles/labels")

fractions = []
files = sorted([f for f in os.listdir(LABELS_DIR) if f.endswith(".tif")])

for f in files:
    ds = gdal.Open(os.path.join(LABELS_DIR, f))
    arr = ds.GetRasterBand(1).ReadAsArray()
    fractions.append(arr.sum() / (1000 * 1000))
    ds = None

fractions = np.array(fractions)
print(f"Tiles            : {len(fractions)}")
print(f"Mean tree frac   : {fractions.mean():.4f}")
print(f"Median tree frac : {np.median(fractions):.4f}")
print(f"Min tree frac    : {fractions.min():.4f}")
print(f"Max tree frac    : {fractions.max():.4f}")
print(f"Tiles >10% trees : {(fractions > 0.10).sum()}")
print(f"Tiles >25% trees : {(fractions > 0.25).sum()}")
print(f"Tiles >50% trees : {(fractions > 0.50).sum()}")