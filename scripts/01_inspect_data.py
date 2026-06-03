import os
from osgeo import gdal, ogr, osr

gdal.UseExceptions()

BASE = os.path.expanduser("~/bw_treenet/data/raw/malmo")

DATASETS = {
    "1960s": os.path.join(BASE, "1960_OF_gray_mmo"),
    "1970s": os.path.join(BASE, "1970_OF_gray_mmo"),
    "1990s": os.path.join(BASE, "1990_OF_gray_mmo"),
    "RGBI":  os.path.join(BASE, "rgbi_latest"),
}

CANOPY = os.path.expanduser("~/bw_treenet/data/processed/malmo/canopy_3006.geojson")


def inspect_raster_folder(name, folder):
    tifs = sorted([f for f in os.listdir(folder) if f.endswith(".tif")])
    if not tifs:
        print(f"{name}: no .tif files found")
        return

    print("\n" + "="*60)
    print(f"  {name}  -  {len(tifs)} tiles")
    print("="*60)

    sample = os.path.join(folder, tifs[0])
    ds = gdal.Open(sample)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    bands = ds.RasterCount
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    pixel_size = abs(gt[1])

    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + xsize * gt[1]
    ymin = ymax + ysize * gt[5]

    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    epsg = srs.GetAttrValue("AUTHORITY", 1)

    print(f"  Sample tile : {tifs[0]}")
    print(f"  Tile size   : {xsize} x {ysize} px")
    print(f"  Pixel size  : {pixel_size} m")
    print(f"  Bands       : {bands}")
    print(f"  EPSG        : {epsg}")
    print(f"  Tile extent : ({xmin:.0f}, {ymin:.0f}) - ({xmax:.0f}, {ymax:.0f})")

    all_xmin = []
    all_ymin = []
    all_xmax = []
    all_ymax = []

    for t in tifs:
        d = gdal.Open(os.path.join(folder, t))
        g = d.GetGeoTransform()
        all_xmin.append(g[0])
        all_ymax.append(g[3])
        all_xmax.append(g[0] + d.RasterXSize * g[1])
        all_ymin.append(g[3] + d.RasterYSize * g[5])

    ext = "({:.0f}, {:.0f}) - ({:.0f}, {:.0f})".format(
        min(all_xmin), min(all_ymin), max(all_xmax), max(all_ymax))
    area = "{:.1f} x {:.1f} km".format(
        (max(all_xmax) - min(all_xmin)) / 1000,
        (max(all_ymax) - min(all_ymin)) / 1000)

    print(f"  Full extent : {ext}")
    print(f"  Area covered: {area}")


def inspect_vector(path):
    print("\n" + "="*60)
    print("  CANOPY VECTOR")
    print("="*60)
    ds = ogr.Open(path)
    layer = ds.GetLayer()
    extent = layer.GetExtent()
    count = layer.GetFeatureCount()
    srs = layer.GetSpatialRef()
    epsg = srs.GetAttrValue("AUTHORITY", 1)
    print(f"  File        : {os.path.basename(path)}")
    print(f"  Features    : {count}")
    print(f"  EPSG        : {epsg}")
    print(f"  Extent      : ({extent[0]:.0f}, {extent[2]:.0f}) - ({extent[1]:.0f}, {extent[3]:.0f})")


for name, folder in DATASETS.items():
    inspect_raster_folder(name, folder)

inspect_vector(CANOPY)

print("\nDone.")