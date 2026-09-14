# bw_treenet — production stage handover

Context carry-over for continuing work on the production/dissemination phase.
Written 11 September 2026.

---

## Project

Deep learning pipeline mapping historical urban tree canopy change in **Malmö,
Gothenburg and Stockholm**, 1960s to present, from archival panchromatic aerial
photography. SLU Alnarp, Formas-funded to end 2027.
Repo: `github.com/blazklobucar/bw_treenet`

**Model:** BWTreeNet (Gui et al. 2025, IEEE TGRS), adapted from Swiss historical
*forest* imagery to Swedish *urban* settings. Single-band, 0.5 m, EPSG:3006.

**Production model: v19** (`models/bwtreenet_v19_best.pt`), threshold 0.6.

---

## Where things stand

The detection phase is **complete**. Inference has been run across all nine
city-era combinations and aggregate canopy figures produced.

### Results — canopy cover (%) within the 2023 tätort extent

| | 1960s | 1970s | 1990s | 1960→1990 |
|---|---|---|---|---|
| Malmö | 6.64 | 5.65 | 8.40 | +1.76 |
| Gothenburg | 17.71 | 21.23 | 16.08 | −1.63 |
| Stockholm | 27.20 | 31.64 | 21.19 | −6.00 |

Detail in `results/canopy_change_v19.csv` (valid area, canopy area, frame counts,
coverage gaps). Per-frame rasters in `results/inference_v19/<city>_<era>/`.

**All three cities show a 1970s peak then decline.** Unresolved whether this is
real or an era-specific imagery bias. This needs settling before it goes in a
paper — a sensible check is a few known-stable sites (old parks, cemeteries)
read across eras.

### Model accuracy (held-out, masked to tätort)

| Benchmark | IoU @0.5 | IoU @0.6 |
|---|---|---|
| Malmö 1970 (unseen city and era) | 0.580 | 0.569 |
| Stockholm 1970 | 0.655 | 0.659 |

---

## What was established (don't re-litigate)

- **Method tuning is exhausted.** A controlled 2×2 ablation (Luminance Enhancer ×
  degradation augmentation) plus earlier negatives (Fourier texture channel,
  class weighting, 512 px retiling) moved IoU by at most +0.005.
- **The Luminance Enhancer must be kept.** Removing it cost ~0.02 IoU, contrary
  to the original author's advice.
- **Era-matched labels are the only lever that measurably works.** +40 labelled
  tiles gave +0.013 IoU (v19). A further +30 (v20) went backwards — possibly
  label inconsistency in the second round, possibly variance; never resolved, and
  there is no variance estimate since no configuration was trained twice.
- **Unmasked water dominated apparent error** on coastal imagery: 56% of false
  positives. Masking to tätort moved precision 0.533 → 0.723 with no retraining.

Full detail in `NOTES.md` under "Publishable findings".

---

## Known limitations of the production output

- **Water inside the tätort boundary is NOT masked.** Tätort is a settlement
  definition, not a land/water boundary, so harbours and canals fall inside it.
  Roughly 15% of model-selected Malmö clips were water false positives. Adding
  the Lantmäteriet hydrography mask (`data/hydro/{1,2,3}/mv_south.shp`, folders
  are 1=STH, 2=Malmö, 3=GTB) is the obvious next fix and needs no retraining.
- **False positives on textured open fields** inside the study area.
- Performance is best in residential areas, degrading toward the urban fringe.
  Rural false negatives fall mostly *outside* tätort and so outside the analysis.
- **Coverage gaps from corrupt source files** — LZW decode failures
  (`Using code not yet in table`). GTB 1960s: 2 frames plus `gtb_clip`.
  STH 1960: 2 frames. STH 1970: 2 frames. Files are full-size with intact
  headers, so it is pixel-data corruption, not truncation; re-download from
  Lantmäteriet Geotorget is the fix (credentials exist).
- **Two output rasters were written corrupt** by the cluster and re-run.

---

## Study-area convention (state this in any write-up)

All figures are computed inside the **2023 SCB tätort boundary, held constant
across every era**. This is deliberate: it measures canopy change within today's
urban extent, so change reflects canopy rather than boundary drift. It also
excludes the rural regime where the model is weakest.

---

## Environment and infrastructure

**Cluster:** NAISS Arrhenius, project `naiss2026-4-1108`, storage 500 GB
(~63 GB used), 100k file quota. GPU partition, account flag
`-A naiss2026-4-1108-gpu`.

```bash
interactive -A naiss2026-4-1108-gpu --partition gpu --gpus 1 -t 02:00:00
module load GPU/Miniforge/26.3.2-2-eb
eval "$(conda shell.bash hook)"
conda activate bwtreenet
cd /nobackup/proj/disk/naiss2026-4-1108/personal/bklobucar/bw_treenet
```

The login node has no Python and no GDAL command-line tools — anything beyond
`ls`/`grep` needs a compute node.

**Local:** EndeavourOS, RTX 4070 Ti Super (16 GB — too small for training at
1000 px; inference is fine). Conda env `bwtreenet`. QGIS: use the Flatpak
(`flatpak run org.qgis.qgis`), not the pacman 4.2.1 build which crashes.

**NAISS asked that data be packed** rather than stored as many small files.
`data/train_tiles.h5` exists for this; the 512 px tile set that caused the quota
breach was deleted.

---

## Three gotchas that break the model silently

1. **Input must be raw 0–255.** The Luminance Enhancer expects the raw byte
   range. Dividing by 255 gives garbage with no error.
2. **Input must be exactly 1000×1000 px.** A LayerNorm in SharpConnect has
   hardcoded spatial dimensions. `Forest_dataset` returns tiles at native size
   (its resize calls are commented out), so mixed tile sizes fail at collation.
3. **Checkpoint loading needs a filter** or `load_state_dict` throws on the
   `.expand()` sharpening kernels:
   ```python
   sd = {k: v for k, v in sd.items()
         if "SharpFeature.weight" not in k and "SharpOri.weight" not in k}
   model.load_state_dict(sd, strict=False)
   ```
   Also: the model applies softmax internally — do not softmax again.

Startup must print **"Luminance Enhancer loaded and frozen"**. The LE load is
wrapped in `if os.path.exists()`, so a wrong path fails silently and degrades
results with no error.

---

## Key scripts

| Script | Purpose |
|---|---|
| `07_train_v18.py` | training; flags `--degrade`, `--drop-le`, `--smoke`, `--batch` |
| `10_score_models.py` | score models on masked Malmö 1970 |
| `11_score_sth.py` | score on STH 1970 clip |
| `19_production_inference.py` | city-era inference, tätort masking, RGB→gray |
| `20_aggregate_canopy.py` | pool per-frame rasters into city-era statistics |
| `21_build_qgis_package.py` | mosaic outputs into a portable QGIS package |

**The cluster copy of `07_train_v18.py` carries fixes the local copy lacks**
(path resolution, checkpoint filter, dtype fix). Do not scp the local version
over it.

---

## Publication plan

1. **Methods** (ISPRS / RSE) — adapting BWTreeNet to historical urban imagery.
   Draftable now; all findings are in `NOTES.md`.
2. **Results** (Landscape and Urban Planning) — 60 years of canopy change in
   three cities. Open question: whether to fold in SCB socioeconomic data.
3. **Validation** (Remote Sensing) — apply the shipped model to recent RGB
   converted to grayscale, compare against Boverket's canopy product. Ian Brown
   (Stockholm University) is running post-1990s imagery; handover package sent.

Qualitative work (Delphi, municipal workshops with Thomas Randrup and Märit
Jansson) follows once maps exist.

---

## Immediate next steps

1. **Add hydrography masking** to the production outputs — cheapest remaining
   accuracy gain, no retraining.
2. **Resolve the 1970s peak** — is it real or imagery bias?
3. **Re-download the corrupt source frames** from Geotorget.
4. **Build the dissemination map** (ArcGIS Online was the plan; QGIS Cloud and
   static XYZ tiles are alternatives).
5. **Draft Paper 1.**

## Open threads

- Formas Explore application with Jonas Bohlin (SLU, forestry remote sensing) —
  potential joint project spanning forest and urban domains, would fund a PhD.
- Alnarp Innovation Prize, deadline 30 September 2026.
- NAISS Large Fall 2026 compute call, closes 16 October 2026.
- Correspondence with Yuanyuan Gui (BWTreeNet author) — has given substantive
  advice; the Luminance Enhancer negative result has not yet been reported back
  to her, and co-authorship on Paper 1 is an open question.
