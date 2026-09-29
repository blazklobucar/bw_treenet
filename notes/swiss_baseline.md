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
