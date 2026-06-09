# BWTreeNet Swedish Urban Tree Canopy — Pipeline Notes

**Project:** Historical urban tree canopy cover change mapping (1960s–present)  
**Cities:** Malmö, Gothenburg, Stockholm  
**Funded:** SLU, until end of 2027  
**Last updated:** 2026-06-08

---

## Current Status

End-to-end pipeline working. Best model is **v10** (GroupNorm, val IoU 0.6727), with v10b continuation run in progress. Inference on Malmö 1960s produces plausible results. Formal accuracy assessment against 901 Malmö manual labels not yet completed.

---

## Critical Bugs Fixed (in order of discovery)

### 1. Double softmax
**Problem:** `BWTreeNet.py` `OutConv` applies `nn.Softmax(dim=1)` internally. The original `lossfunction.py` and our `08_inference.py` were both applying `F.softmax()` again on the output, producing collapsed near-uniform probabilities. Model trained but predicted all background.  
**Fix:** Removed `F.softmax()` from `lossfunction.py` and `08_inference.py`. The model output is already a probability distribution.  
**File:** `BWTreeNet/GuiTest/loss/lossfunction.py`

### 2. Softmax missing dim argument
**Problem:** Original `nn.Softmax()` in `OutConv` had no `dim` argument, causing it to softmax across the wrong dimension and collapse channel 0 to 1.0 and channel 1 to 0.0.  
**Fix:** Changed to `nn.Softmax(dim=1)`.  
**File:** `BWTreeNet/GuiTest/model/BWTreeNet.py` line 162

### 3. IoU loss computed over both classes
**Problem:** Original `IoULoss.loss_function()` averaged IoU over background (class 0) and tree (class 1). With ~70% background pixels the model could achieve ~0.55 loss trivially by predicting all background. The reported val IoU was actually background class IoU, not tree class IoU.  
**Fix:** Rewrote loss to compute IoU on tree class (index 1) only.  
**File:** `BWTreeNet/GuiTest/loss/lossfunction.py`

### 4. BatchNorm grid artefact at inference
**Problem:** With batch size 2 and patch-based sliding window inference (1000×1000 patches), `BatchNorm2d` computes different normalisation statistics per patch. This causes visible grid lines at patch boundaries in inference outputs.  
**Workaround used (v7–v9):** Running inference in `model.train()` mode rather than `model.eval()` mode. This uses batch statistics rather than running stats, which is more consistent but still varies between patches.  
**Proper fix (v10+):** Replaced all 13 `BatchNorm2d` layers with `GroupNorm(8, C)` in `BWTreeNet.py`. GroupNorm normalises within each sample independently, producing consistent outputs regardless of patch content or position. No train/eval mode difference.  
**File:** `BWTreeNet/GuiTest/model/BWTreeNet.py`

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
| v10 | GroupNorm, all data | 0.6727 | TBD — grid artefact should be gone |
| v10b | Continue from v10 best, LR=5e-4 | in progress | TBD |

---

## Dataset

| City | Modern tiles | Historical label tiles | Notes |
|---|---|---|---|
| Malmö | 597 | 0 (901 polygons reserved for validation) | GeoJSON 2022 labels |
| Gothenburg | 2707 | 36 (1960s clip, 2000×2000px) | Boverket binary raster + manual 1960s labels |
| Stockholm | 1157 | 36 (1960s clip, 2000×2000px) | Boverket binary raster + manual 1960s labels |
| **Total** | **4461** | **72** | |

All data EPSG:3006 (SWEREF99 TM), 0.5m resolution, 1000×1000px tiles.

---

## Inference

**Script:** `scripts/08_inference.py`  
**Strategy:** Sliding window, 1000×1000px patches, 200px overlap  
**Model must be fixed at 1000×1000 input** — hardcoded LayerNorm in SharpConnect expects 250×250 after downsampling  
**Current best weights:** `models/bwtreenet_v10_best.pt`  
**BatchNorm recalibration:** No longer needed with GroupNorm (v10+). Use `model.eval()` for inference.

**Results (Malmö 1960s, v7 train-mode):**
- Tree cover range: 8–21% across 12 tiles
- Modern Malmö reference: ~14% canopy cover
- Grid artefact present but detections spatially coherent

**Building footprint masking:**
- `data/raw/malmo/buildings/byggnad_sverige.gpkg` available for Malmö
- Post-processing script written, removes building-overlapping predictions
- Gothenburg and Stockholm building footprints not yet sourced

---

## Validation

**Manual labels:** `data/processed/malmo/validation/labels_malmo_1959.shp`  
- 901 polygons, ~1km² central Malmö, 1959 imagery  
- **Not used for training — reserved for accuracy assessment only**  
- Accuracy assessment script not yet written — priority for post-summer

---

## Known Issues / Post-Summer Agenda

1. **Accuracy assessment** — write `09_accuracy_assessment.py` against 901 Malmö labels. Priority.
2. **GroupNorm inference quality** — run v10/v10b inference and visually confirm grid artefact is gone
3. **Stockholm and Gothenburg building footprints** — source from Lantmäteriet or OSM for masking
4. **More historical labels** — additional clips from 1970s and 1990s would improve multi-epoch robustness. Use different geographic areas to avoid spatial overfitting.
5. **Fourier/texture features** — add FFT-derived texture channel as additional input. Promising for broccoli-texture canopy detection in B&W imagery. Methodological comparison experiment.
6. **Full city inference** — run inference on all 3 cities × 4 time periods (1960s, 1970s, 1990s, modern) once v10b is confirmed working
7. **Change maps** — compute pixel-level canopy change between epochs, aggregate to neighbourhood/district level for stakeholder workshops
8. **NAISS Arrhenius** — account approved (NAISS 2026/4-1108), 200 GPU-h/month on GH200. SSH key registered, awaiting account activation. Use for v11+ training and full city inference.

---

## Key Paths

```
~/bw_treenet/
├── BWTreeNet/GuiTest/          — patched BWTreeNet source (GroupNorm, fixed loss, fixed softmax)
├── data/raw/malmo|gtb|sth/     — raw imagery per city per epoch
├── data/processed/             — canopy rasters, training tiles, historical label tiles
├── models/                     — saved checkpoints (not in git)
├── scripts/                    — all pipeline scripts
│   ├── 06_train.py             — main training script (v10b config)
│   └── 08_inference.py         — sliding window inference
└── results/                    — training logs, inference outputs
```

---

## Compute

**Local:** Endeavour OS, RTX 4070 Ti Super 16GB, 32GB RAM  
**NAISS Arrhenius:** GH200 96GB HBM3, ~4–6x faster than local GPU  
**GitHub:** github.com/blazklobucar/bw_treenet