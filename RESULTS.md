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
* Not compared with ReReVST/MCCNet/CCPL in our pipeline (no published-method baseline yet). Published numbers use other protocols and are not comparable (see LITERATURE.md).
* Style/content metrics are proxies tied to AdaIN's own VGG; the warp error can be lowered by blur/ghosting (blend 0.6 shows this). No human ratings collected yet (HUMAN_STUDY.md).
* One base model, one training recipe, 2 seeds for lt=150 only, 6 styles per evaluation, DAVIS only.
* Fine-tuning DAVIS-only decoder slightly worsens style vs B0 (control +1.9% style distance, CI above 0).
* Round-1 selection rule was changed after seeing validation results (PROTOCOL.md, Amendment 1); the extra seed runs and extra blend strengths were also added after seeing results. All are logged.
