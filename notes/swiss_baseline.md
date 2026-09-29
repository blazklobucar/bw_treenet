# Swiss baseline: findings (28 Sep 2026)

## Released weights
- GitHub release v1.1.0 of GrayVictoria/BWTreeNet: BWTreeNet_weights.zip (379 MB)
- Two checkpoints, 263 MB each: BWTreeNet_SwissHistorical_1946.pth, BWTreeNet_SwissHistorical_1980s.pth
- Different md5s (third_party/gui_upstream/swiss_weights_md5.txt); files in weights/swiss/ (gitignored)
- Upstream README (4 Jun 2026): works well at 1 m; authors unsure about imagery resampled to 1 m; input single-band uint8 0-255
- LuminanceEnhancer/weights/Epoch99.pth (49 KB) is the LE only

## Architecture vs v19 (models/bwtreenet_v19_best.pt)
- Same 200 network parameter keys, no shape mismatches
- Swiss: 365 keys = 200 params + 165 BatchNorm buffers (55 BN layers); stats are real (num_batches_tracked 1,243,520)
- v19: 228 keys = 200 params + LE keys; no BN buffers
- v19 LE weights identical to Epoch99.pth

## Our model file vs upstream (BWTreeNet/GuiTest/model/BWTreeNet.py)
- nn.BatchNorm2d -> nn.GroupNorm(8, C) throughout; all our checkpoints depend on this
- nn.Softmax() -> nn.Softmax(dim=1)
- forward(): x/255 -> LE (frozen) -> x*255 -> 255 - x (upstream inversion kept; confirmed)
- Swiss weights must NOT be loaded into our class (keys match, runs silently with wrong normalisation).
  Use third_party/gui_upstream/BWTreeNet.py.
- TODO: record why GroupNorm was chosen (Paper 1 methods)

## Upstream inference (SwissTest.py + SwissTestParameters.py, as shipped)
- model.eval(); BN uses stored statistics
- normalization_strategy = True: per-patch histogram equalisation (util/util.py EqualizeHist; bins = patch max + 1, output 0-255)
- No LE call in SwissTest.py; EnhanceImage imported (usage: see grep)
- ratio_output_strategy = False: pred.argmax(1), equivalent to 0.5 threshold (v19 production uses 0.6)
- flip_strategy = True: 5-way TTA. Apparent bug: image_ori overwritten with the input image before the vote;
  vote sets >2 to 1 but never <=2 to 0. Verify empirically (TTA on vs off, one tile) before reporting to Gui.
- window 1000, stride 600, abandon/delete_edge on: large-image tiling only

## Baseline design
- Benchmarks: Malmo 1970, Stockholm 1970 (held out); tatort + hydrography masks as in 10_score_models.py
- Configs: v19; Swiss 1980s and 1946, each at native 0.5 m and at 1 m (1 km window at 1 m, crop centre, nearest upsample)
- Swiss preprocessing: primary = histogram equalisation as shipped; secondary = LE input
- TTA off for all models (documented deviation for Swiss)
- Thresholds 0.5 and 0.6 plus 0.3-0.8 sweep (best-threshold numbers flagged optimistic)
- Metrics: pooled IoU, precision, recall, area ratio, per-tile IoU (paired, no significance claims)

## Open items
- EnhanceImage role in SwissTest.py
- TTA bug check
- Write scripts/24_baseline_swiss.py

## Results (29 Sep 2026, scripts/24_baseline_swiss.py, TTA off, tatort-masked)
Sanity: v19 reproduces Malmo 0.580 @0.5 / 0.569 @0.6. Equalisation from upstream util.py.

| config | Malmo IoU@0.6 | Malmo area ratio | STH IoU@0.6 | STH area ratio |
|---|---|---|---|---|
| v19 | 0.569 | 0.935 | 0.659 | 1.105 |
| swiss46_eq_05m | 0.622 | 1.061 | 0.658 | 1.062 |
| swiss46_eq_1m | 0.611 | 0.909 | 0.678 | 0.957 |
| swiss80_eq_05m | 0.507 | 1.249 | 0.536 | 1.341 |
| swiss80_eq_1m | 0.552 | 1.082 | 0.636 | 1.199 |
| swiss80_le_05m | 0.543 | 1.188 | 0.514 | 1.559 |

- Swiss 1946 zero-shot matches/beats v19; wins 6 of 8 Malmo tiles with canopy (all 5 largest)
- 1946 > 1980s weights everywhere; 1980s overpredicts
- Equalisation > LE input for Swiss weights
- Swiss IoU nearly flat across thresholds 0.3-0.8 (saturated probabilities); v19 best at 0.40 on Malmo (0.585)
- v19 at 0.6 underestimates Malmo area (0.935) but overestimates STH (1.105); Swiss46 0.5 m is ~1.06 in both
- Malmo tile 00000_00000 scores 0 for all models: check the reference
- Malmo tiles look like a contiguous 3x3 grid: stitch for the 1 m arm instead of reflect-padding

## Next
- Fine-tune from Swiss 1946 weights (upstream BatchNorm architecture) on Swedish labels;
  2 seeds each scratch vs fine-tuned (variance estimate); likely new production candidate
- Paper 1 cannot claim v19 beats the published model; reframe around transfer + adaptation

## Leakage and selection check (29 Sep 2026)
- ALL of v18a-d, v19, v20 warm-started from v12 (slurm logs: "[model] warm-started from v12";
  run scripts have no --from-scratch). Earlier notes saying "random initialisation" are WRONG.
- 07_train_v18.py uses malmo/tiles_1970 (the Malmo benchmark) as val_loader: it drove LR schedule,
  early stopping and best-checkpoint choice. Malmo 1970 is a validation set, not a test set.
- Malmo benchmark: 9 tiles overlap by 50% (offsets are pixels); union = 1.0 km2, 2000x2000 px.
  Pooled metrics count central pixels up to 4x (same for all models). Score as one mosaic in future.
- Malmo benchmark overlap with training: 25% malmo/tiles_1970new (SAME ERA), 75% malmo/tiles (modern),
  3% tiles_1959.
- STH clip (1.0 km2): 0% same-era overlap; 100% overlap with sth/tiles (modern) and sth/tiles_1960.
  Not used for model selection. -> cleanest test available.
- On STH: v19 0.659 vs Swiss46 0.658 (0.5 m) / 0.678 (1 m). v19's Malmo advantage over the tie may
  reflect selection + leakage.
- Fine-tuning design: validation must be neither benchmark; exclude same-era training tiles that
  overlap either benchmark.
- v12 trained FROM SCRATCH (slurm_v12_20565/26068: "no pretrained weights found"). Lineage: v12 scratch -> v18a-d, v19, v20 warm-start from v12.
- Only malmo/tiles_1970new/616_37_00_1973_clip_03000_03000.tif overlaps the Malmo benchmark (same era). Exclude in new runs.

## GTB 1970 check and fine-tuning setup (29 Sep 2026)
- Zero-shot on all 19 GTB 1970 tiles (IoU@0.5, pooled, no tatort mask): Swiss46 0.614, v19 0.591,
  although GTB 1970 was in v19's TRAINING data.
- gtb/tiles_1970: 9 tiles, 50% overlap, 5 with zero canopy -> held out entirely. Validation = gtb/tiles_1970new (10 tiles, 15-61% canopy).
- Smoke val IoU ~0.15-0.18 was the near-empty first 8 tiles, not a bug; BN freeze did not change it.
- Upstream EqualizeHist crashes on uint8 input containing 255 (image.max()+1 overflows): pass int64.
  Fixed in 24_baseline_swiss.py (results unaffected: no 255 in any benchmark window). Report to Gui with TTA bug.
- 25_train_finetune.py generated by make_25_train_finetune.py (07 loop verbatim). Runs: ft46_s1/s2 (lr 1e-4), sc_s1/s2 (lr 1e-3),
  upstream BatchNorm arch, equalisation, --degrade, batch 8. Optional: --freeze-bn variant.

## Fine-tuning results (29 Sep 2026, results/finetune_eval_ft)
- ft46_s1: best val 0.6775 at epoch 5, early stop at 15 (~1 h). Val flat after epoch 1.
- STH IoU@0.5: swiss46 0.658 | ft46_s1 0.668 | ft46_s2 0.647 | v19 0.655. Malmo: 0.623 | 0.628 | 0.611 | 0.580.
- Seed spread (0.021 on STH) > every between-model gap -> fine-tuning not distinguishable from zero-shot.
- Implication: v18 single-run ablation differences (~0.02) are within seed noise.
- Fine-tuning softens probabilities (threshold matters again); area ratio seed-dependent (0.92-1.02) vs swiss46 steady ~1.06.
- Production candidate: zero-shot swiss46 (pending scratch runs and ensemble check).
- Ensemble ft46_s1+s2 (mean prob) @0.5: STH IoU 0.666 area 1.025; Malmo 0.626 area 1.035. No IoU gain beyond noise;
  area bias ~3% vs swiss46 ~6%, both consistent across cities.
- Leaning: zero-shot swiss46 @0.5 as production (reproducible from public weights, threshold-insensitive);
  decide after scratch runs + 1960s/1990s visual check.
- Ensemble re-inspected with basemap: NO water false positives (fine-tuning fixed that), but still labels
  smaller shrubs as canopy. Ranking on definition: v19 > ft46 ensemble > Swiss46 zero-shot.
- Scratch runs (upstream arch, equalise, no Swiss init): best val 0.613 (s1, ep 27) / 0.611 (s2, ep 31)
  vs ft46_s1 0.678.
- TEST (results/finetune_eval_all), IoU@0.5 STH: sc_s1 0.451, sc_s2 0.518 (vs ft 0.668/0.647, swiss46 0.658, v19 0.655);
  Malmo: sc 0.353/0.409 (vs ft 0.628/0.611, swiss46 0.623, v19 0.580).
  -> pretraining worth +0.15-0.20 IoU on test cities (only +0.065 on GTB val): scratch generalises poorly.
  -> scratch unstable: seed spread 0.07 on STH, area ratio 0.96-1.45.
  -> v19 (no Swiss init) reaches 0.655: its long v12 lineage / LE / GroupNorm / GTB1970 in training /
     Malmo selection carry what short scratch training lacks. Scratch arm is a control for fine-tuning,
     not a stand-in for v19.

## Visual era check (29 Sep 2026, QGIS, 8 windows; Swiss46 red, v19 yellow, ft46 ensemble cyan)
- Swiss46 labels vegetation in general: overpredicts shrub/low vegetation and forest edges (GTB 1960s/1970s/1990s),
  and water surfaces (GTB 1990s). v19 excludes small vegetation and water -> matches our tree-canopy definition.
- Malmo 1960 (leaf-off imagery): v19 much better on urban trees; Swiss46 misses leaf-off deciduous trees.
  v19 has more false positives on open fields.
- Fine-tuned ensemble: no water false positives, but still labels smaller shrubs as canopy.
  Ranking on definition: v19 > ft46 ensemble > Swiss46 zero-shot.
- Benchmarks (Malmo/STH 1970) underrepresent these cases -> similar IoU hides a definitional mismatch.
- DECISION: keep v19 as production.
- Paper 1 angle: pretrained model matches IoU on urban test tiles but transfers a different canopy definition;
  local labels teach tree canopy (incl. leaf-off) vs vegetation. Needs a targeted test set to quantify.
- (QGIS display issue was a corrupted project file, not the exports; GeoTIFFs are correctly in EPSG:3006.)
