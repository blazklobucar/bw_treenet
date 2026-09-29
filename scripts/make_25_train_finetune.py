"""
make_25_train_finetune.py -- builds scripts/25_train_finetune.py from scripts/07_train_v18.py.

Why a patch instead of a new script: the training loop, loss call, validation IoU, LR schedule
and early stopping stay byte-identical to the v18/v19 runs. Only these things change:
  1. model class   -> Gui et al.'s original (third_party/gui_upstream/BWTreeNet.py): BatchNorm, no LE
  2. init          -> --init swiss46 | swiss80 (strict=True load) | scratch
  3. preprocessing -> per-tile histogram equalisation (SwissTest.patch_process), applied after degradation
  4. validation    -> gtb/tiles_1970 + gtb/tiles_1970new (removed from training); Malmo 1970 and the
                      STH clip become pure test sets
  5. training data -> malmo/tiles_1970new_excl (the one tile overlapping the Malmo benchmark removed)
  6. --lr (default 1e-4 for swiss*, 1e-3 for scratch), --seed
Every anchor must match exactly once, otherwise the script stops and changes nothing.

Run once from the repo root:  python scripts/make_25_train_finetune.py
"""
import os, re, sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "scripts", "07_train_v18.py")
DST = os.path.join(HERE, "scripts", "25_train_finetune.py")
s = open(SRC).read()


def sub(old, new, label):
    global s
    n = s.count(old)
    if n != 1:
        sys.exit(f"[STOP] anchor '{label}' found {n} times (need exactly 1); nothing written")
    s = s.replace(old, new)
    print(f"[ok] {label}")


def sub_re(pattern, new, label):
    global s
    m = re.findall(pattern, s, flags=re.S)
    if len(m) != 1:
        sys.exit(f"[STOP] anchor '{label}' matched {len(m)} times (need exactly 1); nothing written")
    s = re.sub(pattern, lambda _: new, s, flags=re.S)
    print(f"[ok] {label}")


# 1. model class: Gui's original
sub("from model.BWTreeNet import BWTreeNet\n",
    '''import importlib.util as _ilu
_UP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "gui_upstream")
_spec = _ilu.spec_from_file_location("gui_upstream_bwtreenet", os.path.join(_UP, "BWTreeNet.py"))
_upm = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_upm)
BWTreeNet = _upm.BWTreeNet          # Gui et al. original: BatchNorm, no built-in LE
''', "model import")

# 4/5. training and validation sets
sub('    "malmo/tiles_1970new", "gtb/tiles_1970new", "sth/tiles_1970new",\n',
    '    "malmo/tiles_1970new_excl", "sth/tiles_1970new",          # gtb 1970new -> validation\n',
    "ALL_SETS 1970new line")
sub('    "gtb/tiles_1960", "gtb/tiles_1970", "gtb/tiles_1990",\n',
    '    "gtb/tiles_1960", "gtb/tiles_1990",                       # gtb 1970 held out entirely\n',
    "ALL_SETS gtb line")
sub('HOLDOUT_SET = "malmo/tiles_1970"',
    'VAL_SETS = ["gtb/tiles_1970new"]   # validation: same era, all tiles with canopy; gtb/tiles_1970 held out entirely\n'
    'HOLDOUT_SET = "malmo/tiles_1970"', "HOLDOUT_SET")
sub("    val_inner = make_set(HOLDOUT_SET)",
    "    val_inner = ConcatDataset([d for d in (make_set(r) for r in VAL_SETS) if d is not None])",
    "val_inner")
sub("val(holdout Malmo1970)", "val(GTB 1970)", "val print")

# 3. equalisation wrapper, applied last in make_set (after degradation)
sub('        print(f"[degrade] modern set {rel} wrapped with historical degradation")\n    return ds',
    '        print(f"[degrade] modern set {rel} wrapped with historical degradation")\n'
    '    ds = Equalise(ds)\n    return ds', "make_set equalise")

sub("def build_augs():",
    '''def equalise_fast(arr):
    """Vectorised copy of Gui et al.'s EqualizeHist (util/util.py) as used in SwissTest.patch_process.
    Verified against the upstream implementation by --smoke."""
    a = np.nan_to_num(np.asarray(arr, dtype=np.float64), nan=0.0)
    a = np.clip(np.rint(a), 0, 255).astype(np.uint8)
    flat = a.flatten().astype(np.int64)
    hist = np.bincount(flat, minlength=int(a.max()) + 1).astype(np.float64)
    cs = np.cumsum(hist)
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()
    cs = (nj / N).astype(np.uint8)
    return cs[flat].reshape(a.shape).astype(np.float32)


def _check_equalise():
    spec = _ilu.spec_from_file_location("gui_upstream_util", os.path.join(_UP, "util", "util.py"))
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod)
    rng = np.random.default_rng(0)
    tests = [rng.integers(0, 256, (1000, 1000)),
             np.clip(rng.normal(90, 25, (1000, 1000)), 0, 255),
             np.clip(rng.gamma(2.0, 30.0, (1000, 1000)), 0, 255)]
    for i, t in enumerate(tests):
        t8 = np.rint(t).astype(np.uint8)
        up = np.asarray(mod.EqualizeHist(t8.astype(np.int64), bins=255).operation(), dtype=np.float64)
        fa = equalise_fast(t8).astype(np.float64)
        d = np.abs(up - fa).max()
        print(f"[smoke] equalise check {i}: max abs diff vs upstream = {d}")
        if d > 0:
            sys.exit("[STOP] fast equalisation differs from upstream EqualizeHist")


class Equalise(Dataset):
    """Swiss preprocessing: per-tile histogram equalisation of Forest_dataset output."""
    def __init__(self, inner):
        self.inner = inner
    def __len__(self):
        return len(self.inner)
    def __getitem__(self, i):
        out = self.inner[i]
        img, lbl, rest = out[0], out[1], out[2:]
        was_tensor = torch.is_tensor(img)
        arr = img.detach().cpu().numpy() if was_tensor else np.asarray(img)
        eq = equalise_fast(arr)
        img = torch.from_numpy(np.ascontiguousarray(eq)).to(out[0].dtype) if was_tensor else eq
        return (img, lbl, *rest)


def build_augs():''', "Equalise class")

# 2/6. arguments
sub('    ap.add_argument("--tag", default="v18a")\n',
    '''    ap.add_argument("--tag", default="v18a")
    ap.add_argument("--init", choices=["swiss46", "swiss80", "scratch"], required=True)
    ap.add_argument("--lr", type=float, default=None,
                    help="default 1e-4 for swiss*, LR (1e-3) for scratch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze-bn", action="store_true", help="keep BatchNorm running stats fixed")
''', "argparse")

sub("    args = ap.parse_args()\n",
    '''    args = ap.parse_args()
    if args.drop_le:
        sys.exit("[STOP] --drop-le does not apply: the upstream model has no LE")
    if args.lr is None:
        args.lr = 1e-4 if args.init.startswith("swiss") else LR
    import random
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    print(f"[cfg] init={args.init} lr={args.lr} seed={args.seed} preprocessing=equalise")
    if args.smoke:
        _check_equalise()
''', "parse_args")

sub_re(r'    model = BWTreeNet\(n_class=N_CLASSES\)\.to\(DEVICE\)\n.*?print\("\[model\] training from scratch"\)\n',
       '''    model = BWTreeNet(n_class=N_CLASSES).to(DEVICE)
    if args.init.startswith("swiss"):
        _p = os.path.join(BASE, "weights", "swiss",
                          {"swiss46": "BWTreeNet_SwissHistorical_1946.pth",
                           "swiss80": "BWTreeNet_SwissHistorical_1980s.pth"}[args.init])
        model.load_state_dict(torch.load(_p, map_location=DEVICE, weights_only=False), strict=True)
        print(f"[model] initialised from {os.path.basename(_p)} (strict=True)")
    else:
        print("[model] upstream architecture, training from scratch")
    if args.freeze_bn:
        _orig_train = model.train
        def _train_keep_bn(mode=True):
            _orig_train(mode)
            for _m in model.modules():
                if isinstance(_m, torch.nn.BatchNorm2d):
                    _m.eval()
            return model
        model.train = _train_keep_bn
        print("[model] BatchNorm running statistics frozen")
''', "model init block")

sub("lr=LR, momentum=0.9", "lr=args.lr, momentum=0.9", "optimizer lr")
sub("held-out Malmo1970 IoU", "val (GTB 1970new) IoU", "final print")

s = ('# GENERATED by scripts/make_25_train_finetune.py from 07_train_v18.py -- edit the generator, not this file.\n'
     + s)
open(DST, "w").write(s)
print(f"\nwrote {DST}")
