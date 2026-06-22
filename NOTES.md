# BWTreeNet Swedish Urban Tree Canopy — Pipeline Notes

**Project:** Historical urban tree canopy cover change mapping (1960s–present)  
**Cities:** Malmö, Gothenburg, Stockholm  
**Funded:** SLU, until end of 2027  
**Last updated:** 2026-06-12

---

## Current Status

**Best working model: v12** (val IoU 0.6740, combined CE+IoU loss + GroupNorm + LE + CLAHE). Inference confirmed working on Malmö 1960s imagery — trees detected at plausible cover percentages (3–37%). Accuracy assessment completed against 901 Malmö 1959 manual labels: **IoU 0.30, F1 0.47 at threshold 0.6**.

Currently digitising 7 new label clips (Malmö/GTB/STH × 1960s/1970s/1990s) for v13 training. Pipeline running on NAISS Arrhenius (GH200).

### Key findings from paper (Gui et al. 2025 IEEE TGRS)
- Original training: batch size 3, fixed LR 0.01, SGD, 100 epochs, 1m resolution
- Original inference: moving window stride 600, central area 800×800, **4-direction flip ensemble** (5 predictions majority vote)
- Their tree cover IoU on Swiss 1980s historical test set: **83.46%** (mIoU 87.09%)
- Data augmentation had ~10% impact on OA — critical for domain transfer
- LE + LAM ablation: adds ~3% OA improvement

### Accuracy assessment results (v12 vs Malmö 1959 labels)
| Threshold | IoU | F1 | Precision | Recall |
|---|---|---|---|---|
| 0.50 | 0.303 | 0.465 | 0.365 | 0.639 |
| 0.60 | **0.304** | **0.467** | 0.397 | 0.565 |
| 0.70 | 0.295 | 0.456 | 0.436 | 0.478 |
| 0.90 | 0.186 | 0.314 | 0.575 | 0.216 |

**Interpretation:** Best threshold 0.6. IoU 0.30 reflects domain gap between modern training data and 1960s historical imagery. Label mismatch also contributes — validation uses individual crown polygons while model predicts continuous canopy cover. Visual inspection confirms model is spatially coherent; sea pixels and low vegetation are main false positive sources.

**Known model weaknesses:**
- Low vegetation (shrubs, hedges) classified as tree canopy
- Small solitary street trees missed or under-detected
- Sea/water pixels produce false positives (needs masking)

---

## Critical Bugs Fixed (in order of discovery)

### 1. Double softmax
**Problem:** `BWTreeNet.py` `OutConv` applies `nn.Softmax(dim=1)` internally. The original `lossfunction.py` and our `08_inference.py` were both applying `F.softmax()` again on the output, producing collapsed near-uniform probabilities.  
**Fix:** Removed `F.softmax()` from `lossfunction.py` and `08_inference.py`.  
**File:** `BWTreeNet/GuiTest/loss/lossfunction.py`

### 2. Softmax missing dim argument
**Problem:** Original `nn.Softmax()` in `OutConv` had no `dim` argument, causing channel collapse.  
**Fix:** Changed to `nn.Softmax(dim=1)`.  
**File:** `BWTreeNet/GuiTest/model/BWTreeNet.py` line 162

### 3. IoU loss computed over both classes
**Problem:** Original loss averaged IoU over background and tree class. Model could achieve ~0.55 trivially by predicting all background.  
**Fix:** Rewrote loss to compute IoU on tree class (index 1) only.  
**File:** `BWTreeNet/GuiTest/loss/lossfunction.py`

### 4. BatchNorm grid artefact at inference
**Problem:** BatchNorm statistics vary per patch during sliding window inference, causing visible grid lines.  
**Proper fix (v10+):** Replaced all 13 `BatchNorm2d` layers with `GroupNorm(8, C)`.  
**File:** `BWTreeNet/GuiTest/model/BWTreeNet.py`

### 5. Pure IoU loss collapses to all-background with GroupNorm
**Problem:** Pure tree-class IoU loss causes all-background collapse with GroupNorm. Val IoU reported high (~0.69) but model predicted zero trees.  
**Fix:** Combined CE + IoU loss (0.5 × NLLLoss + 0.5 × tree-class IoU).  
**File:** `BWTreeNet/GuiTest/loss/lossfunction.py`

### 6. Luminance Enhancer missing from implementation
**Problem:** LE module not integrated in BWTreeNet forward pass despite being in the paper.  
**Fix:** Integrated via `importlib` to avoid circular imports. LE weights loaded frozen from `LuminanceEnhancer/weights/Epoch99.pth`.  
**File:** `BWTreeNet/GuiTest/model/BWTreeNet.py`

### 7. Inference normalisation mismatch
**Problem:** Inference script normalised images to [0,1] by dividing by 255. Model expects raw [0,255] values (LE module and 255-x inversion require uint8 range).  
**Fix:** Removed `/255.0` from normalise function in `08_inference.py`.  
**File:** `scripts/08_inference.py`

---

## Training History

| Version | Notes | Best val IoU | Inference quality |
|---|---|---|---|
| v1–v2 | Early Malmö-only runs | ~0.593 | Not tested |
| v3 | All 3 cities, broken loss | 0.604 | All zeros (broken) |
| v4 | Lower LR, broken loss | 0.590 | All zeros (broken) |
| v5 | Fixed loss, broken softmax | 0.655 | All zeros (broken) |
| v6 | Swiss pretrained, broken softmax | 0.577 | All zeros (broken) |
| v7 | All fixes, from scratch, BatchNorm | 0.6553 | 8–12% tree cover, grid artefact |
| v8 | + GTB 1960s labels (36 tiles) | 0.6786 | 15–39% over-prediction |
| v9 | + STH 1960s labels (36 tiles) | 0.6791 | 4–5% under-prediction |
| v10 | GroupNorm, pure IoU loss | 0.6727 | Zero predictions (IoU collapse) |
| v10b | Continue from v10 | 0.6921 | Zero predictions (IoU collapse) |
| v11 | GroupNorm + LE + CE+IoU loss | 0.6801 | Not tested (likely zero — normalisation bug) |
| v11b | Continue from v11 | 0.6731 | Not tested |
| v12 | + CLAHE augmentation | **0.6740** | ✅ Working — 3–37% tree cover |
| v12b | Continue from v12 | 0.6739 | ✅ Working |
| v13 | + 7 new label clips (1960s/70s/90s) | planned | TBD |

---

## Dataset

| City | Modern tiles | Historical label tiles | Notes |
|---|---|---|---|
| Malmö | 597 | 0 (901 polygons reserved for validation) | GeoJSON 2022 labels |
| Gothenburg | 2707 | 36 (1960s clip) | Boverket raster + manual 1960s labels |
| Stockholm | 1157 | 36 (1960s clip) | Boverket raster + manual 1960s labels |
| **Total** | **4461** | **72** | |

**Clips prepared for labelling (v13):**
- `malmo_1960_clip.tif` — 2000×2000px, 0.5m, same area as validation
- `malmo_1970_clip.tif` — 2000×2000px, 0.5m
- `malmo_1990_clip.tif` — 2000×2000px, 0.5m (different area)
- `gtb_1970_clip.tif` — 2000×2000px, 0.5m
- `gtb_1990_clip.tif` — 1000×1000px, 1m resolution
- `sth_1970_clip.tif` — 2000×2000px, 0.5m
- `sth_1990_clip.tif` — 1000×1000px, 1m resolution

Saved in: `~/bw_treenet/data/raw/clips_for_labelling/`  
Labelling convention: separate shapefile per clip, merged canopy areas, trees only (not shrubs), draw to clip edge for continuous cover.

---

## Inference

**Script:** `scripts/08_inference.py`  
**Strategy:** Sliding window, 1000×1000px patches, 200px overlap, `model.train()` mode  
**Important:** Must use `model.train()` not `model.eval()` — GroupNorm with current weights collapses to zero predictions in eval mode  
**Image normalisation:** Raw [0,255] values — do NOT divide by 255  
**Current best weights:** `models/bwtreenet_v12_best.pt`

**TODO — flip ensemble (from paper):** 4-direction flip + original = 5 predictions, majority vote. Expected +1–3 IoU points. Not yet implemented.

**Results (Malmö 1960s, v12):**
- Tree cover range: 3–37% across 12 tiles
- Modern Malmö reference: ~14% canopy cover
- Sea pixels produce high false positives — needs coastline mask
- Low vegetation (shrubs) included in predictions

**Post-processing applied:**
- Building footprint masking (Malmö only — `byggnad_sverige.gpkg`)
- Sea/water masking: not yet implemented

**Accuracy assessment (v12 vs 901 Malmö 1959 labels):**
- Best IoU: 0.304 at threshold 0.6
- Best F1: 0.467
- Script: `scripts/09_accuracy_assessment.py`
- Results: `results/accuracy_assessment_v12.csv`
- Difference raster: `results/accuracy_assessment_v12_diff.tif`

---

## Planned Model Versions (staged improvement)

### v13 — Additional era-matched labels
- 7 new label clips: Malmö/GTB/STH × 1960s/1970s/1990s (~252 new tiles with 50% overlap)
- Include street tree examples in labelling
- Labels: trees only, not shrubs, separate file per clip
- Expected gain: +1–3 IoU on historical inference

### v14 — Minimum crown size filter (post-processing)
- Morphological opening to remove patches smaller than minimum crown area (~4m² = 16 pixels)
- Removes shrub/low vegetation noise without retraining
- No architecture change needed — apply to v13 inference outputs
- Expected gain: improved precision, reduced false positives

### v15 — Fourier/texture channel
- Add local isotropy map as second input channel (isotropic vs directional frequency energy)
- Directly encodes broccoli texture of tree canopy
- Requires architecture change: BWTreeNet input from 1 to 2 channels
- Full retrain needed
- Comparison experiment for paper methods section

### v16 — Height model masking
- Apply Lantmäteriet national elevation model as post-processing mask
- Pixels below 3m height threshold = not tree
- Removes low vegetation and shrubs systematically
- Check data availability from Lantmäteriet

---

## Known Issues / To-Do

1. **Flip ensemble** — implement in `08_inference.py`. Free improvement, no retraining.
2. **Sea/coastline mask** — source from Lantmäteriet or OSM for Malmö. Apply as post-processing.
3. **GTB/STH building footprints** — source from Lantmäteriet or OSM.
4. **Street tree detection** — model misses small solitary trees. Addressed partially by including street tree examples in v13 labels.
5. **Low vegetation confusion** — model classifies shrubs/hedges as trees. Addressed by v14 crown size filter and v16 height model.
6. **Full city inference** — run all 3 cities × 4 time periods on Arrhenius once v13 confirmed working.
7. **Change detection script** — compute pixel-level canopy gain/loss between epochs.
8. **ArcGIS Online upload** — convert inference outputs to COG, upload for stakeholder web map.
9. **Delphi study** — ethical review needed before distributing Round 1 questionnaire (September 2026).
10. **Malmö 1959 training labels** — convert subset of 901 validation polygons to training (700 train / 200 val). Highest expected single improvement.

---

## Key Paths

```
~/bw_treenet/
├── BWTreeNet/GuiTest/          — patched BWTreeNet source (GroupNorm, LE, fixed loss)
├── data/raw/malmo|gtb|sth/     — raw imagery per city per epoch
├── data/raw/clips_for_labelling/ — 7 clips prepared for v13 labelling
├── data/processed/             — canopy rasters, training tiles, historical label tiles
├── models/                     — saved checkpoints (not in git)
├── scripts/
│   ├── 06_train.py             — main training script (v12 config on Arrhenius)
│   ├── 08_inference.py         — sliding window inference
│   ├── 09_accuracy_assessment.py — accuracy vs manual labels
│   └── run_training.sh         — SLURM job script for Arrhenius
└── results/                    — training logs, inference outputs, accuracy assessment
```

---

## Compute

**Local:** Endeavour OS, RTX 4070 Ti Super 16GB, 32GB RAM  
**NAISS Arrhenius:** GH200 96GB HBM3, ~5x faster than local GPU, 200 GPU-h/month  
**Allocation:** NAISS 2026/4-1108, active until 2027-07-01  
**GitHub:** github.com/blazklobucar/bw_treenet
