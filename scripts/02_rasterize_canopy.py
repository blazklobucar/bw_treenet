import os
import subprocess
from osgeo import gdal, ogr

gdal.UseExceptions()

# ── paths ──────────────────────────────────────────────────────────────────
CANOPY = os.path.expanduser(
    "~/bw_treenet/data/processed/malmo/canopy_3006.geojson")
OUTPUT = os.path.expanduser(
    "~/bw_treenet/data/processed/malmo/canopy_binary_05m.tif")

# ── extent from RGBI coverage (from inspect script) ───────────────────────
# RGBI full extent: (365000, 6152500) - (385000, 6172500)
# We use this as our training area since it defines where we have
# both modern training images and historical target images
XMIN = 365000
YMIN = 6152500
XMAX = 385000
YMAX = 6172500
PIXEL_SIZE = 0.5
EPSG = 3006

# ── rasterize ──────────────────────────────────────────────────────────────
def rasterize_canopy():
    print("Rasterizing canopy polygons to binary raster...")
    print(f"  Extent      : ({XMIN}, {YMIN}) - ({XMAX}, {YMAX})")
    print(f"  Pixel size  : {PIXEL_SIZE} m")
    print(f"  Output size : {int((XMAX-XMIN)/PIXEL_SIZE)} x {int((YMAX-YMIN)/PIXEL_SIZE)} px")
    print(f"  Output      : {OUTPUT}")

    # calculate output dimensions
    width  = int((XMAX - XMIN) / PIXEL_SIZE)
    height = int((YMAX - YMIN) / PIXEL_SIZE)

    # create output raster
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(OUTPUT, width, height, 1, gdal.GDT_Byte,
                           options=["COMPRESS=LZW", "TILED=YES",
                                    "BLOCKXSIZE=256", "BLOCKYSIZE=256"])

    # set geotransform and projection
    out_ds.SetGeoTransform([XMIN, PIXEL_SIZE, 0, YMAX, 0, -PIXEL_SIZE])

    from osgeo import osr
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    out_ds.SetProjection(srs.ExportToWkt())

    # fill with 0 (non-tree)
    band = out_ds.GetRasterBand(1)
    band.Fill(0)
    band.SetNoDataValue(255)

    # burn canopy polygons as 1 (tree)
    canopy_ds = ogr.Open(CANOPY)
    layer = canopy_ds.GetLayer()

    print("  Burning polygons...")
    gdal.RasterizeLayer(out_ds, [1], layer, burn_values=[1])

    out_ds.FlushCache()
    out_ds = None
    canopy_ds = None

    print("  Done.")

    # verify output
    print("\nVerifying output...")
    ds = gdal.Open(OUTPUT)
    band = ds.GetRasterBand(1)
    stats = band.ComputeStatistics(False)
    print(f"  Min: {stats[0]}  Max: {stats[1]}")
    print(f"  Mean (tree fraction): {stats[2]:.4f}")

    width = ds.RasterXSize
    height = ds.RasterYSize
    print(f"  Output size: {width} x {height} px")
    print(f"  File size  : {os.path.getsize(OUTPUT) / 1e6:.1f} MB")
    ds = None

    print("\nRasterization complete.")


rasterize_canopy()