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
