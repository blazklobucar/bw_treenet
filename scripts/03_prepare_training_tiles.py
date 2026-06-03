import os
import numpy as np
from osgeo import gdal, osr
from pathlib import Path

gdal.UseExceptions()

# ── paths ──────────────────────────────────────────────────────────────────
RGBI_DIR    = os.path.expanduser("~/bw_treenet/data/raw/malmo/rgbi_latest")
CANOPY_RASTER = os.path.expanduser(
    "~/bw_treenet/data/processed/malmo/canopy_binary_05m.tif")
OUT_IMAGES  = os.path.expanduser(
    "~/bw_treenet/data/processed/malmo/tiles/images")
OUT_LABELS  = os.path.expanduser(
    "~/bw_treenet/data/processed/malmo/tiles/labels")

# ── parameters ─────────────────────────────────────────────────────────────
TILE_SIZE   = 1000          # pixels
TARGET_RES  = 0.5           # metres - match historical imagery
MIN_TREE_FRACTION = 0.01    # skip tiles with less than 1% tree cover
EPSG        = 3006

# ── setup ──────────────────────────────────────────────────────────────────
Path(OUT_IMAGES).mkdir(parents=True, exist_ok=True)
Path(OUT_LABELS).mkdir(parents=True, exist_ok=True)

# ── load canopy raster once ────────────────────────────────────────────────
print("Loading canopy raster...")
canopy_ds = gdal.Open(CANOPY_RASTER)
canopy_gt = canopy_ds.GetGeoTransform()
canopy_xmin = canopy_gt[0]
canopy_ymax = canopy_gt[3]
canopy_res  = canopy_gt[1]

def world_to_pixel(x, y, gt):
    col = int((x - gt[0]) / gt[1])
    row = int((y - gt[3]) / gt[5])
    return col, row

# ── process each RGBI tile ─────────────────────────────────────────────────
rgbi_tiles = sorted([f for f in os.listdir(RGBI_DIR) if f.endswith(".tif")])
print(f"Found {len(rgbi_tiles)} RGBI tiles")

total_saved = 0
total_skipped = 0

for tile_name in rgbi_tiles:
    tile_path = os.path.join(RGBI_DIR, tile_name)
    ds = gdal.Open(tile_path)
    gt = ds.GetGeoTransform()

    tile_xmin = gt[0]
    tile_ymax = gt[3]
    tile_xmax = tile_xmin + ds.RasterXSize * gt[1]
    tile_ymin = tile_ymax + ds.RasterYSize * gt[5]
    src_res   = abs(gt[1])

    print(f"\nProcessing {tile_name}")
    print(f"  Source res: {src_res}m  extent: ({tile_xmin:.0f},{tile_ymin:.0f})"
          f" - ({tile_xmax:.0f},{tile_ymax:.0f})")

    # resample RGBI tile to 0.5m grayscale in memory
    scale_factor = src_res / TARGET_RES
    out_width  = int(ds.RasterXSize * scale_factor)
    out_height = int(ds.RasterYSize * scale_factor)

    # read RGB bands and convert to grayscale
    r = ds.GetRasterBand(1).ReadAsArray(
        buf_xsize=out_width, buf_ysize=out_height).astype(np.float32)
    g = ds.GetRasterBand(2).ReadAsArray(
        buf_xsize=out_width, buf_ysize=out_height).astype(np.float32)
    b = ds.GetRasterBand(3).ReadAsArray(
        buf_xsize=out_width, buf_ysize=out_height).astype(np.float32)

    gray = ((r + g + b) / 3.0).astype(np.uint8)
    print(f"  Grayscale array: {gray.shape}  min={gray.min()}  max={gray.max()}")

    # extract matching canopy patch
    canopy_col, canopy_row = world_to_pixel(tile_xmin, tile_ymax, canopy_gt)

    # cut into 1000x1000 tiles
    n_tiles_x = out_width  // TILE_SIZE
    n_tiles_y = out_height // TILE_SIZE

    tile_saved = 0
    tile_skipped = 0

    for row in range(n_tiles_y):
        for col in range(n_tiles_x):
            # image patch
            img_patch = gray[
                row*TILE_SIZE:(row+1)*TILE_SIZE,
                col*TILE_SIZE:(col+1)*TILE_SIZE]

            # corresponding canopy patch
            c_row = canopy_row + row * TILE_SIZE
            c_col = canopy_col + col * TILE_SIZE
            lbl_patch = canopy_ds.GetRasterBand(1).ReadAsArray(
                c_col, c_row, TILE_SIZE, TILE_SIZE)

            if lbl_patch is None:
                tile_skipped += 1
                continue

            # skip tiles with almost no tree cover
            tree_frac = lbl_patch.sum() / (TILE_SIZE * TILE_SIZE)
            if tree_frac < MIN_TREE_FRACTION:
                tile_skipped += 1
                continue

            # compute geotransform for this patch
            patch_xmin = tile_xmin + col * TILE_SIZE * TARGET_RES
            patch_ymax = tile_ymax - row * TILE_SIZE * TARGET_RES

            patch_gt = [patch_xmin, TARGET_RES, 0,
                        patch_ymax, 0, -TARGET_RES]

            srs = osr.SpatialReference()
            srs.ImportFromEPSG(EPSG)
            wkt = srs.ExportToWkt()

            # save image tile
            base = f"malmo_{int(patch_xmin)}_{int(patch_ymax)}"
            img_path = os.path.join(OUT_IMAGES, base + ".tif")
            lbl_path = os.path.join(OUT_LABELS, base + ".tif")

            driver = gdal.GetDriverByName("GTiff")

            img_ds = driver.Create(img_path, TILE_SIZE, TILE_SIZE, 1,
                                   gdal.GDT_Byte)
            img_ds.SetGeoTransform(patch_gt)
            img_ds.SetProjection(wkt)
            img_ds.GetRasterBand(1).WriteArray(img_patch)
            img_ds.FlushCache()
            img_ds = None

            lbl_ds = driver.Create(lbl_path, TILE_SIZE, TILE_SIZE, 1,
                                   gdal.GDT_Byte)
            lbl_ds.SetGeoTransform(patch_gt)
            lbl_ds.SetProjection(wkt)
            lbl_ds.GetRasterBand(1).WriteArray(lbl_patch)
            lbl_ds.FlushCache()
            lbl_ds = None

            tile_saved += 1

    print(f"  Saved: {tile_saved}  Skipped: {tile_skipped}")
    total_saved += tile_saved
    total_skipped += tile_skipped

print(f"\nTotal tiles saved  : {total_saved}")
print(f"Total tiles skipped: {total_skipped}")
print(f"Images dir: {OUT_IMAGES}")
print(f"Labels dir: {OUT_LABELS}")
print("\nDone.")