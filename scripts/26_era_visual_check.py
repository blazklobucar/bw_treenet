"""
26_era_visual_check.py -- zero-shot Swiss 1946 vs v19 on random urban windows from every city-era.

No reference labels exist for 1960s/1990s, so this is a sanity check, not an accuracy score:
  * PNG per window: input | Swiss 1946 (t=0.5) | v19 (t=0.6), canopy cover inside tatort in the titles
  * CSV per window: canopy cover per model inside tatort, and IoU between the two models' masks
  * printed summary per city-era: mean cover per model and mean model agreement
Look for an era where one model clearly fails (blank, speckled, or tracking brightness instead of trees)
and for eras where the two models' cover estimates diverge -- that is where the production choice
would change Paper 2's trends.

Windows: 1000 x 1000 px from the raw frames (0.5 m eras = 500 m; 1990s at native 1 m = 1 km),
<=5% nodata, >=50% inside the 2023 tatort. Both models run at native resolution, as in production.
Seeded, so the same windows come out every run.

Run on a GPU node:  source ~/bw_setup.sh && python scripts/26_era_visual_check.py
Output: results/era_visual_check/
"""
import os, csv, glob, importlib.util
import numpy as np
import rasterio
from rasterio.windows import Window, bounds as win_bounds
from rasterio.features import geometry_mask
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "era_visual_check")
N_PER = 6            # windows per city-era
WIN = 1000
T_SW, T_V19 = 0.5, 0.6
IMG = {
    "malmo_1960": "data/raw/malmo/1960_OF_gray_mmo",
    "malmo_1970": "data/raw/malmo/1970_OF_gray_mmo",
    "malmo_1990": "data/raw/malmo/1990_OF_gray_mmo",
    "gtb_1960":   "data/raw/gtb/1960_OF_gray_gtb",
    "gtb_1970":   "data/raw/gtb/1970_OF_gray_gtb",
    "gtb_1990":   "data/raw/gtb/1990_OF_rgb_gtb",     # single-band despite the name
    "sth_1960":   "data/raw/sth/1960_OF_gray_sth",
    "sth_1970":   "data/raw/sth/1970_OF_gray_sth",
    "sth_1990":   "data/raw/sth/1990_OF_gray_sth",
}

spec = importlib.util.spec_from_file_location("b24", os.path.join(HERE, "scripts", "24_baseline_swiss.py"))
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

TAT_ALL = gpd.read_file(os.path.join(HERE, "data", "Tatorter_2023.gpkg"))
_tat_cache = {}
def tatort_in(crs):
    key = str(crs)
    if key not in _tat_cache:
        _tat_cache[key] = TAT_ALL.to_crs(crs)
    return _tat_cache[key]


def pick_windows(ce, rng):
    frames = sorted(glob.glob(os.path.join(HERE, IMG[ce], "*.tif")))
    out, tries = [], 0
    while len(out) < N_PER and tries < 800 and frames:
        tries += 1
        f = frames[int(rng.integers(len(frames)))]
        try:
            with rasterio.open(f) as r:
                if r.width < WIN or r.height < WIN:
                    continue
                x = int(rng.integers(0, r.width - WIN + 1)); y = int(rng.integers(0, r.height - WIN + 1))
                w = Window(x, y, WIN, WIN)
                a = r.read(1, window=w)
                tr, bnd, crs, res = r.window_transform(w), win_bounds(w, r.transform), r.crs, r.res[0]
        except Exception as e:                      # e.g. the known LZW-corrupt frames
            print(f"  [skip] {os.path.basename(f)}: {type(e).__name__}")
            frames.remove(f)
            continue
        if (a == 0).mean() > 0.05:
            continue
        tat = tatort_in(crs)
        geoms = tat.iloc[list(tat.sindex.intersection(bnd))].geometry
        if len(geoms) == 0:
            continue
        inside = ~geometry_mask(geoms, out_shape=(WIN, WIN), transform=tr)
        if inside.mean() < 0.5:
            continue
        out.append(dict(frame=os.path.basename(f), x=x, y=y, res=res,
                        img=a.astype(np.float32), inside=inside))
    return out


def overlay(ax, img, mask, title):
    lo, hi = np.percentile(img[img > 0], [2, 98]) if (img > 0).any() else (0, 255)
    ax.imshow(np.clip((img - lo) / max(hi - lo, 1), 0, 1), cmap="gray", interpolation="nearest")
    if mask is not None:
        rgba = np.zeros(mask.shape + (4,), np.float32)
        rgba[mask] = (0.1, 0.85, 0.2, 0.45)
        ax.imshow(rgba, interpolation="nearest")
    ax.set_title(title, fontsize=10); ax.axis("off")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(42)
    print("loading models")
    m_sw, _ = b.build("swiss46_eq_05m")
    m_v19, _ = b.build("v19")
    rows = []
    for ce in IMG:
        print(f"== {ce}")
        wins = pick_windows(ce, rng)
        if len(wins) < N_PER:
            print(f"  only {len(wins)} windows found")
        for i, w in enumerate(wins):
            p_sw = b.predict(m_sw, w["img"], 1) >= T_SW
            p_v = b.predict(m_v19, w["img"], 1) >= T_V19
            ins = w["inside"]
            c_sw, c_v = p_sw[ins].mean(), p_v[ins].mean()
            inter = (p_sw & p_v & ins).sum(); union = ((p_sw | p_v) & ins).sum()
            agree = inter / union if union else float("nan")
            rows.append(dict(city_era=ce, frame=w["frame"], x=w["x"], y=w["y"], res_m=w["res"],
                             cover_swiss46=round(float(c_sw), 4), cover_v19=round(float(c_v), 4),
                             agreement_iou=round(float(agree), 4)))
            fig, ax = plt.subplots(1, 3, figsize=(15, 5.3))
            side = WIN * w["res"]
            overlay(ax[0], w["img"], None, f"{ce}  {w['frame']}  ({side:.0f} m window)")
            overlay(ax[1], w["img"], p_sw, f"Swiss 1946 zero-shot @{T_SW}: cover {100*c_sw:.1f}%")
            overlay(ax[2], w["img"], p_v, f"v19 @{T_V19}: cover {100*c_v:.1f}%   agreement IoU {agree:.2f}")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"{ce}_{i:02d}.png"), dpi=90)
            plt.close(fig)
            print(f"  {i:02d} {w['frame']:40s} swiss46 {100*c_sw:5.1f}%  v19 {100*c_v:5.1f}%  agree {agree:.2f}")

    with open(os.path.join(OUT, "windows.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

    print(f"\n{'city_era':12s} {'n':>2s} {'swiss46':>8s} {'v19':>8s} {'diff':>7s} {'agree':>6s}")
    for ce in IMG:
        r = [x for x in rows if x["city_era"] == ce]
        if not r:
            continue
        cs = np.mean([x["cover_swiss46"] for x in r]); cv = np.mean([x["cover_v19"] for x in r])
        ag = np.nanmean([x["agreement_iou"] for x in r])
        print(f"{ce:12s} {len(r):2d} {100*cs:7.1f}% {100*cv:7.1f}% {100*(cs-cv):+6.1f} {ag:6.2f}")
    print(f"\nPNGs and windows.csv in {OUT}")


if __name__ == "__main__":
    main()
