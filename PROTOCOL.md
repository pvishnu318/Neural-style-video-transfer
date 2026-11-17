# Experiment protocol (written BEFORE any fine-tuned model was evaluated)

Base model: pretrained AdaIN (naoto0804/pytorch-AdaIN weights). Everything uses images in [0,1], no hidden normalisation.

## Methods
| id | what it is | flow at inference |
|---|---|---|
| B0 | pretrained AdaIN, every frame independent | no |
| B0+blendA | B0 + post-hoc flow-warped blending of the previous output, strength A (occlusion-masked) | yes (RAFT, ~135 ms/frame at 480p on the 4060) |
| control | AdaIN decoder fine-tuned with content+style loss only (same data, steps, optimiser as B1/B2) | no |
| B1 | fine-tuned with content+style + flow-warped temporal loss (weight lt) | no |
| B2 | B1 recipe + LocalAttnFusion module (flow-free local cross-frame attention + gate) | no |

Why `control`: DAVIS-only fine-tuning changes the decoder even without a temporal term. B1 is compared with `control`
(and B0) so that "extra training" is not mistaken for "temporal loss"; B2 is compared with B1 so that "the module" is isolated.

## Data (disjoint)
* videos: DAVIS-2017 train (60) for training; official val (30) split 15/15 into **val** (decisions) and **test** (final, untouched until the end). Lists in `splits/`.
* styles (WikiArt shard, split by artist): train 480 / val 30 / test 40, no artist overlap. Evaluation uses 6 evenly spaced styles of the corresponding split.
* The 3 original project styles (cuphead, mosaic, starry night) are used only for qualitative showcase / human study, never for selection.

## Metrics (see `src/metrics.py`)
Temporal: masked warping error MSE/L1, k=1 and k=5, flow estimated on the ORIGINAL frames (RAFT-Large), forward-backward occlusion mask, 8-bit outputs.
Quality: content LPIPS (AlexNet), AdaIN style distance (mean/std) and Gram distance. Warp error alone can be gamed by blur / washed-out style,
so results are always read as a trade-off. Reference row = warp error of the original video (metric floor).
Aggregation: average over styles within a video, then bootstrap 95% CI over the 15 videos (video = independent unit).

## Fixed training recipe (all runs identical except the stated switch)
2000 steps, batch 4 clips x 3 frames, 256x256 crops of native-resolution 480p frames, Adam lr 5e-5 cosine to 5e-6, seed 0,
lambda_content 1, lambda_style 10 (AdaIN defaults). Only the decoder (and B2 module) is trained; the VGG encoder is frozen.

## Selection rule (decided in advance, applied on VAL only)
B1 temporal weight lt in {150, 500, 1500}. Pick the lt with the lowest val warp_mse_k1 among those that satisfy BOTH
 style_ms <= 1.10 x B0.style_ms  and  content_lpips <= B0.content_lpips + 0.02.
If none satisfies both, pick the one with the smallest violation and say so. B2 uses the same lt as the selected B1.
No further hyper-parameter tuning against val or test. Test is evaluated once, at the end, with all methods.

## Known limitations (stated up front)
single training seed; style images are 256x256 WikiArt thumbnails; warp error / style distance use networks tied to the training objective;
no human study yet; no comparison with ReReVST/MCCNet yet (their pretrained models are not on this machine).

---
## Amendment 1 (2026-11-16, after seeing ROUND-1 validation results; test split still untouched)

**What happened in round 1** (480 train styles; archived in `results/round1_480styles/`, `checkpoints/round1_480styles/`):
| model | warp_mse_k1 | content_lpips | style_ms |
|---|---|---|---|
| B0 | 0.00735 | 0.551 | 0.348 |
| control | 0.00722 | 0.550 | 0.527 |
| B1 lt150 | 0.00529 | 0.538 | 0.461 |
| B1 lt500 | 0.00376 | 0.521 | 0.539 |
| B1 lt1500 | 0.00250 | 0.512 | 0.684 |

The original selection rule (style_ms <= 1.10 x B0) could not be met by ANY fine-tuned model, including the control (0.527). Applied literally, the
"smallest violation" fallback would have picked lt=150. Diagnostic (`src/diag_style_overfit.py`, 4 val videos): the control had style_ms 0.340 on TRAIN styles
(B0: 0.392) but 0.464 on unseen val styles (B0: 0.313) -> the decoder over-fits the 480 training styles; the pretrained decoder saw tens of thousands.

**Changes (made after seeing val results, so disclosed as such):**
1. Training styles 480 -> 5000 (525 artists, <=20 paintings/artist, from the same WikiArt shard; val/test style sets are byte-identical to round 1; artist-disjoint).
2. Selection rule is now relative to the `control` (same data, steps, optimiser; isolates the temporal term) instead of B0:
   pick the B1 lt with the lowest val warp_mse_k1 among those with style_ms <= 1.10 x control.style_ms AND content_lpips <= control.content_lpips + 0.02.
   B0 is still reported for context (the cost of fine-tuning vs. the original pretrained model is part of the honest comparison).
3. Everything else (steps, lr, lt grid {150,500,1500}, seeds, metrics) unchanged. Round-1 numbers stay on record.

---
## Round-2 validation results (5000 training styles; val split = 15 videos x 6 unseen styles; test untouched)
| model | warp_mse_k1 | warp_mse_k5 | content_lpips | style_ms |
|---|---|---|---|---|
| original video (metric floor) | 0.00089 | 0.00344 | - | - |
| B0 (pretrained, from eval_val.json) | 0.00735 | - | 0.551 | 0.348 |
| control | 0.00712 | 0.01291 | 0.543 | 0.357 |
| B1 lt150 | 0.00531 | 0.00986 | 0.534 | 0.377 |
| B1 lt500 | 0.00384 | 0.00752 | 0.518 | 0.444 |
| B1 lt1500 | 0.00256 | 0.00540 | 0.509 | 0.569 |

Overfitting gap closed: control style_ms 0.357 vs B0 0.348 (round 1: 0.527).
**Selection (Amendment-1 rule, applied mechanically):** style_ms limit = 1.10 x 0.3574 = 0.393 -> only lt=150 passes (0.377). Selected **lt = 150**.
Observation (not a result to sell): the temporal loss trades style for consistency monotonically; lt500/1500 get lower warp error but visibly pay in style_ms.
B2 is trained with lt=150 (`run_queue2.sh`), evaluated on val, then everything is evaluated ONCE on test.

## B2 validation result (lt=150, seed 0)
B2: warp_mse_k1 0.00516 (B1 lt150: 0.00531), warp_mse_k5 0.00960 (0.00986), content_lpips 0.5325 (0.5336), style_ms 0.371 (0.377).
Paired over 15 val videos, B2-B1 warp_mse_k1 = -0.00015 (95% CI -0.00018..-0.00012), B2 better in 15/15 videos (-2.8%); LPIPS/style differences are within noise.
BUT: the attention gate shrank from ~0.025 (init) to 0.008 during training, i.e. the module contributes very little; and with one seed per model a -2.8% gap cannot be separated from run-to-run variance.
Added (no tuning, disclosed): a second seed (seed=1) of B1 lt150 and B2 lt150 (`run_queue3.sh`) to estimate that variance before any claim about B2.

## Seed-variance result (val)  [decides what we may claim about B2]
| lt=150 | seed 0 | seed 1 |
|---|---|---|
| B1 warp_mse_k1 | 0.00531 | 0.00516 |
| B2 warp_mse_k1 | 0.00516 | 0.00531 |
The B2-B1 difference flips sign between seeds (-2.8% / +2.8%) -> it is seed noise. Per-video paired CIs (15/15 videos) do NOT capture seed variance and must not be read as significance.
Conclusion fixed before the test run: B2 (LocalAttnFusion) is NOT shown to improve on B1; its gate stays ~0.007-0.008. Report it as a negative/neutral result.

## FINAL test evaluation (run once): B0, B0+blend0.3/0.6, control, B1 lt150 (seeds 0,1), B1 lt500, B1 lt1500, B2 lt150 (seeds 0,1), with --timing. Script: run_test.sh.
