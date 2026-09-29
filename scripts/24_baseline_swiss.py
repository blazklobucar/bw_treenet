"""
24_baseline_swiss.py -- zero-shot baseline: released Swiss BWTreeNet weights vs v19.

Benchmarks: Malmo 1970 tiles (held out) and the STH 1970 clip, tatort-masked, scored with
the same compute_metrics and taper-blended sliding window as 10_score_models.py /
11_score_sth.py (imported, not copied).

Swiss models are run with Gui et al.'s ORIGINAL model class
(third_party/gui_upstream/BWTreeNet.py: BatchNorm, stored statistics, eval mode),
loaded with strict=True, with their shipped preprocessing (SwissTest.py patch_process:
per-window histogram equalisation). Test-time augmentation is OFF for all models.

Configs
  v19             our production model, native 0.5 m
  swiss80_eq_05m  Swiss 1980s weights, equalisation, native 0.5 m
  swiss46_eq_05m  Swiss 1946 weights, equalisation, native 0.5 m
  swiss80_eq_1m   Swiss 1980s, image 2x2-mean downsampled to 1 m, probability upsampled (nearest)
  swiss46_eq_1m   Swiss 1946, same at 1 m
  swiss80_le_05m  Swiss 1980s, LE input instead of equalisation (secondary check)

Run on a GPU node:
    source ~/bw_setup.sh
    python scripts/24_baseline_swiss.py                      # everything
    python scripts/24_baseline_swiss.py --configs v19 --bench malmo   # quick sanity check

SANITY CHECK: v19 must reproduce Malmo 0.580 @0.5 / 0.569 @0.6 and STH 0.655 / 0.659.
If it does not, the data loading here differs from 10/11 and no other number is valid.

Outputs (results/baseline_swiss/):
  summary.csv    pooled metrics per bench x config x threshold (0.30-0.80)
  per_tile.csv   Malmo per-tile IoU at 0.5 and 0.6
  probs_<bench>_<config>.npz   tree probability maps (float16), for re-thresholding and figures
"""
import os, sys, csv, glob, time, argparse, importlib.util
import numpy as np
import torch
import torch.nn as nn
import rasterio

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UP = os.path.join(HERE, "third_party", "gui_upstream")
OUT = os.path.join(HERE, "results", "baseline_swiss")
SWISS = {
    "swiss80": os.path.join(HERE, "weights", "swiss", "BWTreeNet_SwissHistorical_1980s.pth"),
    "swiss46": os.path.join(HERE, "weights", "swiss", "BWTreeNet_SwissHistorical_1946.pth"),
}
LE_MODEL = os.path.join(HERE, "BWTreeNet", "LuminanceEnhancer", "model.py")
LE_WEIGHTS = os.path.join(HERE, "BWTreeNet", "LuminanceEnhancer", "weights", "Epoch99.pth")
THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.801, 0.05)]
ALL_CONFIGS = ["v19", "swiss80_eq_05m", "swiss46_eq_05m",
               "swiss80_eq_1m", "swiss46_eq_1m", "swiss80_le_05m"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# our scoring code, reused as-is (requires main() behind `if __name__ == "__main__"`)
S10 = load_module("score10", os.path.join(HERE, "scripts", "10_score_models.py"))
S11 = load_module("score11", os.path.join(HERE, "scripts", "11_score_sth.py"))
DEVICE = S10.DEVICE

# Gui et al.'s unmodified model class
UPM = load_module("gui_upstream_bwtreenet", os.path.join(UP, "BWTreeNet.py"))

# ---- Swiss preprocessing: histogram equalisation as in SwissTest.patch_process ----------
try:
    _util = load_module("gui_upstream_util", os.path.join(UP, "util", "util.py"))
    EqualizeHist = _util.EqualizeHist
    EQ_SRC = "upstream util/util.py"
except Exception as e:  # e.g. util.py imports something missing from this env
    EQ_SRC = f"FALLBACK reimplementation ({type(e).__name__}: {e})"

    class EqualizeHist:
        def __init__(self, image, bins=65536, normalize_max=255, normalize_type="uint8"):
            self.image = image
            self.bins = int(image.max()) + 1          # upstream ignores the bins argument
            self.normalize_max = normalize_max
            self.normalize_type = normalize_type

        def operation(self):
            flat = self.image.flatten().astype(np.int64)
            hist = np.bincount(flat, minlength=self.bins).astype(np.float64)
            cs = np.cumsum(hist)
            nj = (cs - cs.min()) * self.normalize_max
            N = cs.max() - cs.min()
            cs = (nj / N).astype(self.normalize_type)
            return cs[flat].reshape(self.image.shape)


def swiss_equalise(p):
    p = p.copy()
    p[p > 255] = 255
    p = np.nan_to_num(p, nan=0)
    p = EqualizeHist(np.rint(p).astype(np.int64), bins=255).operation()
    return np.asarray(p, dtype=np.float64)


class SwissRunner(nn.Module):
    """Wraps the upstream network so predict_full() can call it like our models."""
    def __init__(self, net, prep, le=None):
        super().__init__()
        self.net, self.prep, self.le = net, prep, le

    @torch.no_grad()
    def forward(self, t):
        if self.prep == "eq":
            x = swiss_equalise(t[0, 0].cpu().numpy())
            t = torch.from_numpy(x).float()[None, None].to(t.device)
        elif self.prep == "le":
            xe, _ = self.le(t / 255.0)
            t = xe * 255.0
        return self.net(t)


def load_swiss(path):
    net = UPM.BWTreeNet(n_class=2).to(DEVICE)
    sd = torch.load(path, map_location=DEVICE, weights_only=False)
    net.load_state_dict(sd, strict=True)          # any key mismatch raises here
    n_bn = sum(isinstance(m, nn.BatchNorm2d) for m in net.modules())
    print(f"  loaded {os.path.basename(path)} strict=True, BatchNorm2d layers: {n_bn} (expect 55)")
    net.eval()
    return net


def load_le():
    m = load_module("le_model", LE_MODEL)
    le = m.enhance_net_nopool(scale_factor=1).to(DEVICE)
    le.load_state_dict(torch.load(LE_WEIGHTS, map_location=DEVICE))
    le.eval()
    return le


def build(cfg):
    if cfg == "v19":
        return S10.load_model(S10.MODELS["v19"], False), 1
    weights, prep, res = cfg.split("_")
    le = load_le() if prep == "le" else None
    return SwissRunner(load_swiss(SWISS[weights]), prep, le).eval(), (2 if res == "1m" else 1)


def predict(model, image, scale):
    if scale == 1:
        return S10.predict_full(model, image)
    H, W = image.shape
    h2, w2 = H // 2, W // 2
    small = image[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).mean(axis=(1, 3))
    small = np.rint(small)                        # Swiss input is integer grey levels
    pr = S10.predict_full(model, small)           # reflect-pads to 1000 px if smaller
    up = np.repeat(np.repeat(pr, 2, 0), 2, 1)
    out = np.zeros((H, W), np.float32)
    out[:up.shape[0], :up.shape[1]] = up
    if H % 2: out[-1, :] = out[-2, :]
    if W % 2: out[:, -1] = out[:, -2]
    return out


# ---- data ------------------------------------------------------------------------------
def _match(d, name):
    p = os.path.join(d, name)
    if os.path.exists(p):
        return p
    key = "_".join(os.path.splitext(name)[0].split("_")[-2:])
    c = [f for f in glob.glob(os.path.join(d, "*.tif")) if key in os.path.basename(f)]
    if len(c) == 1:
        return c[0]
    raise FileNotFoundError(f"no unique match for {name} in {d}: {c}")


def malmo_data():
    for ip in sorted(glob.glob(os.path.join(S10.IMG_DIR, "*.tif"))):
        name = os.path.basename(ip)
        with rasterio.open(ip) as r: img = r.read(1).astype(np.float32)
        with rasterio.open(_match(S10.LBL_DIR, name)) as r: gt = r.read(1) > 0
        with rasterio.open(_match(S10.TAT_DIR, name)) as r: tat = r.read(1) > 0
        yield name, img, gt, tat


def sth_data():
    with rasterio.open(S11.VAL_CLIP) as r:
        img = r.read(1).astype(np.float32)
    gt = np.asarray(S11.rasterize_labels(S11.VAL_SHP, S11.VAL_CLIP)) > 0
    tat = S11.tatort_mask(S11.VAL_CLIP, S11.TATORT)
    tat = np.ones_like(gt) if tat is None else np.asarray(tat).astype(bool)
    yield "sth_1970_clip", img, gt, tat


BENCH = {"malmo": malmo_data, "sth": sth_data}


# ---- scoring ---------------------------------------------------------------------------
def pooled(c):
    TP, FP, FN, TN = c
    eps = 1e-10
    P = TP / (TP + FP + eps); R = TP / (TP + FN + eps)
    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                IoU=TP / (TP + FP + FN + eps), Precision=P, Recall=R,
                F1=2 * P * R / (P + R + eps), AreaRatio=(TP + FP) / (TP + FN + eps))


def run(configs, benches):
    os.makedirs(OUT, exist_ok=True)
    print(f"equalisation source: {EQ_SRC}")
    summary, per_tile = [], []
    for bench in benches:
        data = list(BENCH[bench]())
        print(f"\n=== {bench}: {len(data)} image(s), "
              f"{sum(int(t.sum()) for _, _, _, t in data):,} tatort pixels ===")
        for cfg in configs:
            t0 = time.time()
            print(f"[{bench}] {cfg}")
            model, scale = build(cfg)
            counts = {t: np.zeros(4, np.int64) for t in THRESHOLDS}
            probs = {}
            for name, img, gt, tat in data:
                pr = predict(model, img, scale)
                probs[name] = pr.astype(np.float16)
                g = gt[tat]; p_in = pr[tat]
                for t in THRESHOLDS:
                    m = S10.compute_metrics(p_in >= t, g)
                    counts[t] += [m["TP"], m["FP"], m["FN"], m["TN"]]
                    if bench == "malmo" and t in (0.5, 0.6):
                        per_tile.append(dict(config=cfg, tile=name, threshold=t,
                                             IoU=m["IoU"], tatort_px=int(tat.sum())))
            for t in THRESHOLDS:
                summary.append(dict(bench=bench, config=cfg, threshold=t, **pooled(counts[t])))
            np.savez_compressed(os.path.join(OUT, f"probs_{bench}_{cfg}.npz"), **probs)
            del model; torch.cuda.empty_cache()
            print(f"    done in {time.time() - t0:.0f} s")

    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    if per_tile:
        with open(os.path.join(OUT, "per_tile.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_tile[0])); w.writeheader(); w.writerows(per_tile)

    for bench in benches:
        print(f"\n--- {bench} (tatort-masked, pooled) ---")
        print(f"{'config':16s} {'IoU@0.5':>8s} {'IoU@0.6':>8s} {'P@0.6':>6s} {'R@0.6':>6s} "
              f"{'Area@0.6':>8s} {'best t':>6s} {'IoU@best':>8s}")
        for cfg in configs:
            rows = {r["threshold"]: r for r in summary if r["bench"] == bench and r["config"] == cfg}
            best = max(rows.values(), key=lambda r: r["IoU"])
            r5, r6 = rows[0.5], rows[0.6]
            print(f"{cfg:16s} {r5['IoU']:8.3f} {r6['IoU']:8.3f} {r6['Precision']:6.3f} "
                  f"{r6['Recall']:6.3f} {r6['AreaRatio']:8.3f} {best['threshold']:6.2f} {best['IoU']:8.3f}")
    print("\nBest-threshold values are tuned on the test data: optimistic for every model.")
    print(f"Outputs in {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=ALL_CONFIGS, choices=ALL_CONFIGS)
    ap.add_argument("--bench", nargs="+", default=["malmo", "sth"], choices=list(BENCH))
    a = ap.parse_args()
    run(a.configs, a.bench)
