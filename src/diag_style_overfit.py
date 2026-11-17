"""Diagnostic: does fine-tuning overfit the TRAIN styles? Compare style distance on train-split vs val-split styles (4 val videos)."""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, lpips
from evaluate import pick_styles, evaluate_method
from adain import AdaINNet
from flowutil import Flow
from metrics import Evaluator
from data import read_split, ROOT
from methods import PerFrameAdaIN, load_adain
net = load_adain(); lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
ev = Evaluator(net, Flow("cuda"), lp, ROOT / "data" / "flow_cache")
seqs = read_split("val")[:4]
for label, path in [("B0", None), ("control", "checkpoints/control.pt")]:
    m = PerFrameAdaIN(load_adain(path) if path else net, label)
    for sp in ("train", "val"):
        rows = evaluate_method(m, ev, seqs, pick_styles(sp, 6), max_frames=0, verbose=False)
        print(f"{label:8s} on {sp:5s} styles: style_ms={np.mean([r['style_ms'] for r in rows]):.3f}  style_gram={np.mean([r['style_gram'] for r in rows]):.6f}  content_lpips={np.mean([r['content_lpips'] for r in rows]):.3f}", flush=True)
