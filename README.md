# Video neural style transfer: temporal consistency on top of AdaIN

Arbitrary-style video stylisation (input: a content video + a style image). Base model: pretrained AdaIN. We fine-tune the decoder with a
flow-warped temporal loss (B1) and test a small flow-free attention module (B2), and compare against post-hoc flow blending, the original
per-style models of the earlier project, and (in our pipeline) the released ReReVST and MCCNet models.

**Start here:** [`RESULTS.md`](RESULTS.md) (final test numbers and what they do / do not show), [`PROTOCOL.md`](PROTOCOL.md) (rules fixed before the
experiments + every later amendment), [`LITERATURE.md`](LITERATURE.md) (related work, with verification status per paper), [`HUMAN_STUDY.md`](HUMAN_STUDY.md).

Status is tracked honestly in those files; results still being produced are marked as such.

## Layout
- `src/` code: `adain.py`, `flowutil.py` (RAFT, warping, occlusion masks), `metrics.py`, `evaluate.py`, `train.py`, `temporal_module.py` (B2), `external_methods.py` / `eval_external.py` (published models in our pipeline), `eval_old_models.py`, `make_report.py`
- `tests/test_metrics.py` synthetic checks of the metrics (warp, RAFT, occlusion mask)
- `splits/` DAVIS-2017 train / val / test lists (test is used once, at the end)
- `results/` evaluation JSONs, tables, trade-off chart, qualitative frame figures; `logs/` training and evaluation logs
- `run_*.sh` the exact job queues that produced the results

## Not in the repository (too large, or someone else's)
DAVIS videos, WikiArt style sets, model weights/checkpoints, third-party code (ReReVST, MCCNet; their own licences apply) and the human-study clips.
Pretrained AdaIN weights: `vgg_normalised.pth`, `decoder.pth` from https://github.com/naoto0804/pytorch-AdaIN (put in `weights/`).

## Reproduce (Windows, one RTX-class GPU, Python venv with torch/torchvision, lpips, opencv)
`python tests/test_metrics.py`, then see `run_queue*.sh`, `run_test.sh`, `src/make_report.py`.
