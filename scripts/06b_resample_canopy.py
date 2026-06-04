import os
from osgeo import gdal

gdal.UseExceptions()

# ── cities to resample ─────────────────────────────────────────────────────
CITIES = {
    "gtb": {
        "input":  os.path.expanduser(
            "~/bw_treenet/data/raw/gtb/canopy/tradtackning_binar3m.tif"),
        "output": os.path.expanduser(
            "~/bw_treenet/data/processed/gtb/canopy_binary_05m.tif"),
    },
    "sth": {
        "input":  os.path.expanduser(
            "~/bw_treenet/data/raw/sth/canopy/tradtackning_binar3m.tif"),
        "output": os.path.expanduser(
            "~/bw_treenet/data/processed/sth/canopy_binary_05m.tif"),
    },
}

TARGET_RES = 0.5

for city, paths in CITIES.items():
    os.makedirs(os.path.dirname(paths["output"]), exist_ok=True)
    print(f"\nProcessing {city}...")

    # open source
    src = gdal.Open(paths["input"])
    gt  = src.GetGeoTransform()
    src_res = abs(gt[1])

    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + src.RasterXSize * src_res
    ymin = ymax - src.RasterYSize * src_res

    out_width  = int((xmax - xmin) / TARGET_RES)
    out_height = int((ymax - ymin) / TARGET_RES)

    print(f"  Source    : {src.RasterXSize} x {src.RasterYSize} @ {src_res}m")
    print(f"  Target    : {out_width} x {out_height} @ {TARGET_RES}m")
    print(f"  Extent    : ({xmin:.0f}, {ymin:.0f}) - ({xmax:.0f}, {ymax:.0f})")

    # resample using nearest neighbour to preserve binary values
    gdal.Warp(
        paths["output"],
        src,
        width=out_width,
        height=out_height,
        resampleAlg=gdal.GRA_NearestNeighbour,
        creationOptions=["COMPRESS=LZW", "TILED=YES",
                         "BLOCKXSIZE=256", "BLOCKYSIZE=256"]
    )
    src = None

    # verify
    ds  = gdal.Open(paths["output"])
    bnd = ds.GetRasterBand(1)
    stats = bnd.ComputeStatistics(False)
    print(f"  Output min={stats[0]} max={stats[1]} mean={stats[2]:.4f}")
    print(f"  File size : {os.path.getsize(paths['output'])/1e6:.1f} MB")
    ds = None
    print(f"  Done.")

print("\nAll cities resampled.")