"""
26b_export_windows.py -- export chosen windows from 26_era_visual_check.py as GeoTIFFs for QGIS.

For each window: <name>_img.tif (input), <name>_swiss46.tif and <name>_v19.tif
(1 = canopy, 0 = nodata/transparent), all georeferenced in the frame's CRS (EPSG:3006).

Run on a GPU node after 26_era_visual_check.py:
    source ~/bw_setup.sh
    python scripts/26b_export_windows.py gtb_1960_02 gtb_1970_00 gtb_1970_05 gtb_1990_01 gtb_1990_04 \
                                         malmo_1960_02 malmo_1960_04 malmo_1960_05
Output: results/era_visual_check/qgis/
"""
import os, sys, csv, importlib.util
import numpy as np
import rasterio
from rasterio.windows import Window

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "results", "era_visual_check")
OUT = os.path.join(SRC, "qgis")

spec = importlib.util.spec_from_file_location("c26", os.path.join(HERE, "scripts", "26_era_visual_check.py"))
c26 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c26)
b = c26.b


def main(wanted):
    os.makedirs(OUT, exist_ok=True)
    rows = list(csv.DictReader(open(os.path.join(SRC, "windows.csv"))))
    counters, index = {}, {}
    for r in rows:                                   # rebuild the names used for the PNGs
        i = counters.get(r["city_era"], 0); counters[r["city_era"]] = i + 1
        index[f"{r['city_era']}_{i:02d}"] = r
    missing = [w for w in wanted if w not in index]
    if missing:
        sys.exit(f"unknown window names: {missing}")

    m_sw, _ = b.build("swiss46_eq_05m")
    m_v19, _ = b.build("v19")
    m_f1, _ = b.build("ft46_s1"); m_f2, _ = b.build("ft46_s2")
    m_s1, _ = b.build("sc_s1"); m_s2, _ = b.build("sc_s2")
    for name in wanted:
        r = index[name]
        path = os.path.join(HERE, c26.IMG[r["city_era"]], r["frame"])
        w = Window(int(r["x"]), int(r["y"]), c26.WIN, c26.WIN)
        with rasterio.open(path) as src:
            img = src.read(1, window=w)
            prof = src.profile.copy()
            prof.update(width=c26.WIN, height=c26.WIN, count=1, transform=src.window_transform(w),
                        compress="deflate", tiled=False)
            prof.pop("blockxsize", None); prof.pop("blockysize", None)
        with rasterio.open(os.path.join(OUT, f"{name}_img.tif"), "w", **{**prof, "dtype": "uint8", "nodata": 0}) as d:
            d.write(img.astype(np.uint8), 1)
        for tag, model, t in [("swiss46", m_sw, c26.T_SW), ("v19", m_v19, c26.T_V19)]:
            mask = (b.predict(model, img.astype(np.float32), 1) >= t).astype(np.uint8)
            with rasterio.open(os.path.join(OUT, f"{name}_{tag}.tif"), "w",
                               **{**prof, "dtype": "uint8", "nodata": 0}) as d:
                d.write(mask, 1)
        x32 = img.astype(np.float32)
        pf = (b.predict(m_f1, x32, 1) + b.predict(m_f2, x32, 1)) / 2
        with rasterio.open(os.path.join(OUT, f"{name}_ft46ens.tif"), "w",
                           **{**prof, "dtype": "uint8", "nodata": 0}) as d:
            d.write((pf >= 0.5).astype(np.uint8), 1)
        ps = (b.predict(m_s1, x32, 1) + b.predict(m_s2, x32, 1)) / 2
        with rasterio.open(os.path.join(OUT, f"{name}_scens.tif"), "w",
                           **{**prof, "dtype": "uint8", "nodata": 0}) as d:
            d.write((ps >= 0.5).astype(np.uint8), 1)
        print(f"  wrote {name} ({r['frame']})")
    print(f"\nGeoTIFFs in {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
