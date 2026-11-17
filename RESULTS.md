# Results summary (all numbers from `results/eval_test*.json`, produced by `src/evaluate.py`; tables by `src/make_report.py`)

Setup: pretrained AdaIN as base. Test = 15 DAVIS-2017 videos (never used for any decision) x 6 unseen WikiArt styles (artist-disjoint from training). 480p, 8-bit outputs.
Warp error = masked Lai-et-al.-style warping MSE, RAFT-Large flow on the ORIGINAL frames, occlusion mask; lower = smoother. Original video floor: 0.00118 (1 frame), 0.00361 (5 frames).
Style distance = AdaIN-VGG mean/std distance to the style image (lower = closer). Content LPIPS: lower = closer to the content frame.
CI = bootstrap 95% over the 15 videos (styles averaged inside a video). Speed: RTX 4060 laptop GPU, 480p, median of 5 repeats (`src/time_methods.py`); the number includes moving frames CPU<->GPU.

| method | warp err 1fr [95% CI] | warp err 5fr | content LPIPS | style dist | ms/frame | flow at inference |
|---|---|---|---|---|---|---|
| B0 pretrained AdaIN | 0.01056 [0.00863, 0.01270] | 0.01750 | 0.5646 | 0.2366 | 59 | no |
| B0 + flow blend 0.1 | 0.00811 [0.00664, 0.00975] | 0.01567 | 0.5604 | 0.2734 | 286 | yes |
| B0 + flow blend 0.2 | 0.00613 [0.00503, 0.00738] | 0.01406 | 0.5568 | 0.3592 | 286 | yes |
| B0 + flow blend 0.3 | 0.00454 [0.00373, 0.00547] | 0.01259 | 0.5537 | 0.4860 | 286 | yes |
| B0 + flow blend 0.45 | 0.00273 [0.00223, 0.00329] | 0.01053 | 0.5506 | 0.7397 | 286 | yes |
| B0 + flow blend 0.6 | 0.00145 [0.00119, 0.00175] | 0.00852 | 0.5505 | 1.0580 | 286 | yes |
| control (fine-tuned, no temporal loss) | 0.01052 [0.00860, 0.01266] | 0.01753 | 0.5594 | 0.2411 | 59 | no |
| B1 lt=150 (seed 0) | 0.00818 [0.00663, 0.00991] | 0.01444 | 0.5521 | 0.2525 | 59 | no |
| B1 lt=150 (seed 1) | 0.00794 [0.00643, 0.00962] | 0.01410 | 0.5487 | 0.2568 | 59 | no |
| B1 lt=500 | 0.00621 [0.00497, 0.00762] | 0.01169 | 0.5473 | 0.3006 | 59 | no |
| B1 lt=1500 | 0.00428 [0.00332, 0.00545] | 0.00875 | 0.5422 | 0.4656 | 59 | no |
| B2 lt=150 (seed 0) | 0.00791 [0.00640, 0.00960] | 0.01401 | 0.5535 | 0.2624 | 64 | no |
| B2 lt=150 (seed 1) | 0.00814 [0.00662, 0.00985] | 0.01441 | 0.5522 | 0.2538 | 64 | no |

Chart: `results/tradeoff_test.png`. Tables: `results/table_test.md/.csv`. Qualitative frames/videos: `results/qualitative/`.
Flow cost as implemented = 227 ms/frame (2 RAFT-Large passes + mask), so flow blending ~286 ms/frame vs 59 for a per-frame model.

## What the evidence supports
1. **Temporal loss (B1) works.** vs control, paired over 15 videos: warp err 1fr -22.3% (seed 0) / -24.5% (seed 1), 5fr -17.6% / -19.6%; better on 15/15 videos; style distance +4.8% / +6.5%; content LPIPS slightly better (-1.3% / -1.9%).
2. **The control equals B0 on warp error** (-0.4%, CI includes 0): the gain is not "extra training".
3. **Raising lt trades style for smoothness monotonically** (lt 150 -> 500 -> 1500: warp 0.0082 -> 0.0062 -> 0.0043, style 0.253 -> 0.301 -> 0.466). There is no free lunch; the weight is a user-facing knob.
4. **B1 vs flow blending at matched warp error**: B1 lt150 (0.0082, style 0.253) vs blend 0.1 (0.0081, 0.273); lt500 (0.0062, 0.301) vs blend 0.2 (0.0061, 0.359); lt1500 (0.0043, 0.466) vs blend 0.3 (0.0045, 0.486). B1 has lower style distance at each matched point, also lower 5-frame error, and needs no flow at inference (~59 vs ~286 ms/frame).
   Caveat: the blend strengths 0.1/0.2/0.45 were added after the main test run to draw a fair curve; blend is a very simple post-process, not a strong published method.
5. **B2 (our attention module) shows no benefit over B1**: 3% better with seed 0, 2.5% worse with seed 1 on both val and test; its gate stays ~0.007.

## What it does NOT show
* Compared with ReReVST and MCCNet in our pipeline (section 6) but NOT with CCPL / AdaAttN / others. Numbers quoted in those papers use other protocols and are not comparable (see LITERATURE.md).
* Style/content metrics are proxies tied to AdaIN's own VGG; the warp error can be lowered by blur/ghosting (blend 0.6 shows this). No human ratings collected yet (HUMAN_STUDY.md).
* One base model, one training recipe, 2 seeds for lt=150 only, 6 styles per evaluation, DAVIS only.
* Fine-tuning DAVIS-only decoder slightly worsens style vs B0 (control +1.9% style distance, CI above 0).
* Round-1 selection rule was changed after seeing validation results (PROTOCOL.md, Amendment 1); the extra seed runs and extra blend strengths were also added after seeing results. All are logged.

---
# 6. Published models run through OUR pipeline (same 15 test videos x 6 unseen styles, same flow/masks/metrics)
Models: ReReVST (Wang et al., TIP 2020; official code + final weights, inference mirrors the authors' own script incl. global feature sharing) and
MCCNet (Deng et al., AAAI 2021; official decoder/module weights; VGG file byte-identical to ours; the authors' `test_video.py` cannot run as released, so we
reimplemented its inference core: relu4_1 -> MCC module -> decoder, frame by frame). Default settings, no tuning. Style image given at 256x256 like for all methods.

| method | warp err 1fr [95% CI] | warp err 5fr | content LPIPS | style dist | ms/frame | peak GPU |
|---|---|---|---|---|---|---|
| B1 lt=500 (ours) | 0.00621 [0.00497, 0.00762] | 0.01169 | 0.5473 | 0.3006 | 59 | <1 GB |
| B1 lt=1500 (ours) | 0.00428 [0.00332, 0.00545] | 0.00875 | 0.5422 | 0.4656 | 59 | <1 GB |
| B0 + flow blend 0.45 | 0.00273 [0.00223, 0.00329] | 0.01053 | 0.5506 | 0.7397 | 286 | - |
| MCCNet (published) | 0.00572 [0.00443, 0.00716] | 0.01094 | 0.5327 | 0.5663 | 61 | 0.5 GB |
| ReReVST (published) | 0.00258 [0.00186, 0.00352] | 0.00672 | 0.5289 | 1.1056 | 171 | 5.5 GB |

Paired over the 15 test videos (bootstrap 95% CI of the relative difference):
* **B1 lt=1500 vs MCCNet:** warp 1fr -25% [-33, -18] (lower on 13/15 videos), 5fr -20%, style distance -18% [-23, -13] (14/15), content LPIPS +1.8% [-0.2, +3.8] (not distinguishable). By these metrics B1 lt=1500 is both smoother and closer to the style than MCCNet.
* **B1 lt=500 vs MCCNet:** warp +8.5% (MCCNet smoother on 13/15), style distance -47% (B1 closer on 15/15).
* **ReReVST vs MCCNet:** warp -55% (smoother on 15/15), style distance +95% (worse on 15/15).
* **ReReVST vs flow blend 0.45:** about equal 1-frame warp (-5% [-18, +13]), but much smoother over 5 frames (-36% [-41, -32], 15/15), better content LPIPS (-4%), and a higher style distance (+50%).

What this means, honestly:
1. **ReReVST is the smoothest learned method we tested, especially over longer gaps (5 frames: 0.0067 vs our best 0.0088).** Our B1 does NOT beat it on smoothness.
2. **ReReVST pays for it in style distance (1.11 vs 0.24-0.47 for ours).** Caution: our style metric uses AdaIN's own VGG mean/std statistics, and ReReVST deliberately uses a *relaxed* style loss (its supplement), so this metric is expected to disfavour it. We cannot say from the metric alone which looks better; that needs the human study.
3. **Our B1 family sits better than MCCNet in the smoothness/style trade-off** (see `results/tradeoff_test.png`); MCCNet is about as fast as ours (61 vs 59 ms/frame), ReReVST is ~3x slower (171 ms) and needs 5.5 GB.
4. Not claimed: that B1 is better than ReReVST. Not tested: CCPL, AdaAttN, other published models.

# 7. The earlier project's per-style TransformerNet models (your friends' checkpoints), our metrics, our test videos
Normalisation handled as in the old `run_temporal.py` (ImageNet-normalised in/out); frames reflect-padded to a multiple of 4. The old fine-tuning used the same 60 DAVIS training videos; our 15 test videos come from the old validation set, so they were never seen. Each model is evaluated only on its own style image; style distances are only comparable within one style, and the per-style models were trained with a different (VGG16 Gram) objective than the AdaIN-VGG distance used here, so their style distances are not a fair comparison. Full table: `results/table_old_models.md`.

| style | method | warp err 1fr | warp err 5fr | content LPIPS | style dist |
|---|---|---|---|---|---|
| cuphead | old original (10000 it) | 0.01045 | 0.02112 | 0.6006 | 2.069 |
| cuphead | old "temporal_best" | 0.00940 | 0.01909 | 0.5921 | 2.749 |
| cuphead | old temporal_m2 / m3 / m2_e5 | 0.0166 / 0.0166 / 0.0164 | 0.0276 / 0.0276 / 0.0278 | 0.639 / 0.639 / 0.646 | 1.16 / 1.16 / 1.66 |
| cuphead | old m2 + flow blend 0.35 (our mask) | 0.00619 | 0.01853 | 0.6207 | 1.521 |
| cuphead | our B0 / B1 lt150 / B1 lt500 | 0.0293 / 0.0228 / 0.0175 | 0.0450 / 0.0375 / 0.0308 | 0.611 / 0.608 / 0.608 | 0.73 / 0.77 / 0.85 |
| mosaic | old original | 0.02371 | 0.03593 | 0.6385 | 1.020 |
| mosaic | our B0 / B1 lt150 / B1 lt500 | 0.0313 / 0.0259 / 0.0194 | 0.0435 / 0.0381 / 0.0303 | 0.622 / 0.623 / 0.631 | 0.65 / 0.75 / 1.07 |
| starry night | old original | 0.03696 | 0.04523 | 1.0250 | 3.564 |
| starry night | our B0 / B1 lt150 / B1 lt500 | 0.0198 / 0.0160 / 0.0126 | 0.0285 / 0.0243 / 0.0201 | 0.672 / 0.649 / 0.664 | 0.54 / 0.66 / 0.72 |
(Original video floor: 0.00118 / 0.00361.)

What this shows:
* **Only one of the old project's "temporal" fine-tunes made the model steadier.** `temporal_best`: warp error -10% vs the old original on Cuphead (paired 95% CI -12.8..-7.7%, lower on 15/15 videos), content LPIPS -1.4%. `m2`, `m2_e5`, `m3` are clearly WORSE: warp error +57..+59% (CI +47..+70%, worse on 15/15), content LPIPS +6..+7.5%, and they visibly changed the look (heavier crackle texture). `m2` and `m3` are different files but practically the same model (per-video warp error differs by <= 1.3e-5).
  Likely reason (from their training logs, not tested): the content/style losses were ~10^6 while the temporal term was ~0.06, i.e. the temporal term was numerically negligible, so those runs mostly drifted the style.
* **Per-style models are not uniformly steadier or flickerier than AdaIN:** Cuphead: the old model is much steadier than our AdaIN-based methods (0.010 vs 0.029 / 0.0175 at lt=500); Mosaic: our B1 lt=500 (0.0194) is steadier than the old model (0.0237), B0 is not (0.0313); Starry Night: our methods are steadier (0.013-0.020 vs 0.037).
* What per-style models cannot do: they need one trained network per style; ours takes any style image. The numbers do not measure that.
* Caveat for the starry-night LPIPS (1.02): the model changes the image strongly (checked visually: correct, strong Van-Gogh look; `results/old_models_preview_camel.jpg`); LPIPS to the content frame is not comparable across styles.

# 8. Speed (RTX 4060 laptop GPU, 480p, median of 5 repeats after warm-up, frames start on CPU)
Old per-style TransformerNet 27 ms/frame; ours (AdaIN decoder, B0/control/B1) 59; B2 64; MCCNet 61; ReReVST 171 (peak 5.5 GB); flow blending 286 (flow+mask = 227 ms). Scripts: `src/time_methods.py`, `src/time_external.py`.
