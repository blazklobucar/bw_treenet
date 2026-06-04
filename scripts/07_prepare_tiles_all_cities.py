import os
import numpy as np
from osgeo import gdal, osr
from pathlib import Path

gdal.UseExceptions()

# ── config ─────────────────────────────────────────────────────────────────
TILE_SIZE         = 1000
TARGET_RES        = 0.5
MIN_TREE_FRACTION = 0.01
EPSG              = 3006

# ── city definitions ────────────────────────────────────────────────────────
# Stockholm excluded - RGBI capture is early leaf flush (spring 2023)
# Will use Malmo+Gothenburg trained model for inference
# Revisit if leaf-on imagery becomes available
CITIES = {
    "malmo": {
        "rgbi_dir":      os.path.expanduser(
            "~/bw_treenet/data/raw/malmo/rgbi_latest"),
        "canopy_raster": os.path.expanduser(
            "~/bw_treenet/data/processed/malmo/canopy_binary_05m.tif"),
        "out_images":    os.path.expanduser(
            "~/bw_treenet/data/processed/malmo/tiles/images/"),
        "out_labels":    os.path.expanduser(
            "~/bw_treenet/data/processed/malmo/tiles/labels/"),
    },
    "gtb": {
        "rgbi_dir":      os.path.expanduser(
            "~/bw_treenet/data/raw/gtb/rgbi_latest"),
        "canopy_raster": os.path.expanduser(
            "~/bw_treenet/data/processed/gtb/canopy_binary_05m.tif"),
        "out_images":    os.path.expanduser(
            "~/bw_treenet/data/processed/gtb/tiles/images/"),
        "out_labels":    os.path.expanduser(
            "~/bw_treenet/data/processed/gtb/tiles/labels/"),
    },
}

# ── helpers ─────────────────────────────────────────────────────────────────
def world_to_pixel(x, y, gt):
    col = int((x - gt[0]) / gt[1])
    row = int((y - gt[3]) / gt[5])
    return col, row


def process_city(city_name, cfg):
    rgbi_dir    = cfg["rgbi_dir"]
    canopy_path = cfg["canopy_raster"]
    out_images  = cfg["out_images"]
    out_labels  = cfg["out_labels"]

    if not os.path.exists(rgbi_dir):
        print(f"\n[{city_name}] RGBI directory not found: {rgbi_dir} -- skipping")
        return 0, 0

    if not os.path.exists(canopy_path):
        print(f"\n[{city_name}] Canopy raster not found: {canopy_path} -- skipping")
        return 0, 0

    Path(out_images).mkdir(parents=True, exist_ok=True)
    Path(out_labels).mkdir(parents=True, exist_ok=True)

    canopy_ds = gdal.Open(canopy_path)
    canopy_gt = canopy_ds.GetGeoTransform()

    rgbi_tiles = sorted([
        f for f in os.listdir(rgbi_dir) if f.endswith(".tif")])
    print(f"\n[{city_name}] Found {len(rgbi_tiles)} RGBI tiles")

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    wkt = srs.ExportToWkt()
    driver = gdal.GetDriverByName("GTiff")

    total_saved   = 0
    total_skipped = 0

    for tile_name in rgbi_tiles:
        tile_path = os.path.join(rgbi_dir, tile_name)
        ds = gdal.Open(tile_path)
        gt = ds.GetGeoTransform()

        tile_xmin = gt[0]
        tile_ymax = gt[3]
        tile_xmax = tile_xmin + ds.RasterXSize * gt[1]
        tile_ymin = tile_ymax + ds.RasterYSize * gt[5]
        src_res   = abs(gt[1])

        scale = src_res / TARGET_RES
        out_w = int(ds.RasterXSize * scale)
        out_h = int(ds.RasterYSize * scale)

        n_bands = ds.RasterCount
        if n_bands >= 3:
            r = ds.GetRasterBand(1).ReadAsArray(
                buf_xsize=out_w, buf_ysize=out_h).astype(np.float32)
            g = ds.GetRasterBand(2).ReadAsArray(
                buf_xsize=out_w, buf_ysize=out_h).astype(np.float32)
            b = ds.GetRasterBand(3).ReadAsArray(
                buf_xsize=out_w, buf_ysize=out_h).astype(np.float32)
            gray = ((r + g + b) / 3.0).astype(np.uint8)
        else:
            gray = ds.GetRasterBand(1).ReadAsArray(
                buf_xsize=out_w, buf_ysize=out_h)

        canopy_col, canopy_row = world_to_pixel(
            tile_xmin, tile_ymax, canopy_gt)

        n_tiles_x = out_w // TILE_SIZE
        n_tiles_y = out_h // TILE_SIZE

        tile_saved   = 0
        tile_skipped = 0

        for row in range(n_tiles_y):
            for col in range(n_tiles_x):
                img_patch = gray[
                    row*TILE_SIZE:(row+1)*TILE_SIZE,
                    col*TILE_SIZE:(col+1)*TILE_SIZE]

                c_row = canopy_row + row * TILE_SIZE
                c_col = canopy_col + col * TILE_SIZE

                # skip if outside canopy raster bounds
                if (c_col < 0 or c_row < 0 or
                    c_col + TILE_SIZE > canopy_ds.RasterXSize or
                    c_row + TILE_SIZE > canopy_ds.RasterYSize):
                    tile_skipped += 1
                    continue

                lbl_patch = canopy_ds.GetRasterBand(1).ReadAsArray(
                    c_col, c_row, TILE_SIZE, TILE_SIZE)

                if lbl_patch is None:
                    tile_skipped += 1
                    continue

                tree_frac = lbl_patch.sum() / (TILE_SIZE * TILE_SIZE)
                if tree_frac < MIN_TREE_FRACTION:
                    tile_skipped += 1
                    continue

                patch_xmin = tile_xmin + col * TILE_SIZE * TARGET_RES
                patch_ymax = tile_ymax - row * TILE_SIZE * TARGET_RES
                patch_gt   = [patch_xmin, TARGET_RES, 0,
                               patch_ymax, 0, -TARGET_RES]

                base     = f"{city_name}_{int(patch_xmin)}_{int(patch_ymax)}"
                img_path = os.path.join(out_images, base + ".tif")
                lbl_path = os.path.join(out_labels, base + ".tif")

                img_ds = driver.Create(
                    img_path, TILE_SIZE, TILE_SIZE, 1, gdal.GDT_Byte)
                img_ds.SetGeoTransform(patch_gt)
                img_ds.SetProjection(wkt)
                img_ds.GetRasterBand(1).WriteArray(img_patch)
                img_ds.FlushCache()
                img_ds = None

                lbl_ds = driver.Create(
                    lbl_path, TILE_SIZE, TILE_SIZE, 1, gdal.GDT_Byte)
                lbl_ds.SetGeoTransform(patch_gt)
                lbl_ds.SetProjection(wkt)
                lbl_ds.GetRasterBand(1).WriteArray(lbl_patch)
                lbl_ds.FlushCache()
                lbl_ds = None

                tile_saved += 1

        print(f"  {tile_name}: saved={tile_saved} skipped={tile_skipped}")
        total_saved   += tile_saved
        total_skipped += tile_skipped
        ds = None

    canopy_ds = None
    print(f"[{city_name}] Total saved={total_saved} skipped={total_skipped}")
    return total_saved, total_skipped


# ── run ─────────────────────────────────────────────────────────────────────
print("Starting multi-city tile preparation...")
grand_total = 0

for city, cfg in CITIES.items():
    saved, skipped = process_city(city, cfg)
    grand_total += saved

print(f"\nGrand total tiles saved: {grand_total}")
print("Done.")