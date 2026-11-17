# Video neural style transfer: temporal consistency on top of AdaIN

Arbitrary-style video stylisation (input: a content video + a style image). Base model: pretrained AdaIN. We fine-tune the decoder with a
flow-warped temporal loss (B1) and test a small flow-free attention module (B2), and compare against post-hoc flow blending, the original
per-style models of the earlier project, and (in our pipeline) the released ReReVST and MCCNet models.

**Start here:** [`RESULTS.md`](RESULTS.md) (final test numbers and what they do / do not show), [`PROTOCOL.md`](PROTOCOL.md) (rules fixed before the
experiments + every later amendment), [`LITERATURE.md`](LITERATURE.md) (related work, with verification status per paper), [`HUMAN_STUDY.md`](HUMAN_STUDY.md).

Status is tracked honestly in those files. The repository is updated automatically while work is in progress.

## Layout
- `src/` code: `adain.py`, `flowutil.py` (RAFT, warping, occlusion masks), `metrics.py`, `evaluate.py`, `train.py`, `temporal_module.py` (B2), `external_methods.py` / `eval_external.py` (published models in our pipeline), `eval_old_models.py`, `make_report.py`
- `tests/test_metrics.py` synthetic checks of the metrics (warp, RAFT, occlusion mask)
- `splits/` DAVIS-2017 train / val / test lists (test is used once, at the end)
- `results/` evaluation JSONs, tables, trade-off chart, qualitative frame figures; `logs/` training and evaluation logs
- `run_*.sh` the exact job queues that produced the results

## Not in the repository (too large, or someone else's)
DAVIS videos, WikiArt style sets, the original pretrained AdaIN weights, the discarded round-1 checkpoints, third-party code (ReReVST, MCCNet; their own licences apply) and the human-study clips.
Pretrained AdaIN weights: `vgg_normalised.pth`, `decoder.pth` from https://github.com/naoto0804/pytorch-AdaIN (put in `weights/`).

## Trained models (in `checkpoints/`, 14 MB each)
`control.pt` (fine-tuned, no temporal loss), `b1_lt150.pt` / `b1_lt150_s1.pt` (temporal loss weight 150, seeds 0 and 1), `b1_lt500.pt`, `b1_lt1500.pt`,
`b2_lt150.pt` / `b2_lt150_s1.pt` (with the attention module; no gain over B1). Each file holds the fine-tuned decoder (`decoder`), the attention module for B2 (`module`)
and the training arguments (`args`); the frozen VGG encoder is the original `vgg_normalised.pth`. Use e.g.
`python src/evaluate.py --split val --methods ckpt:b1_lt500:checkpoints/b1_lt500.pt` (B2: `attn:name:path`).

## Reproduce (Windows, one RTX-class GPU, Python venv with torch/torchvision, lpips, opencv)
`python tests/test_metrics.py`, then see `run_queue*.sh`, `run_test.sh`, `src/make_report.py`.
