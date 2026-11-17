"""Clean speed measurement (run with the GPU otherwise idle). 480p, 60 frames of 'blackswan', median of 5 repeats after warm-up.
Per-frame decoders (B0 / control / B1) share one architecture -> timed once as 'perframe'. B2 adds the attention module.
Flow blend = per-frame model + one RAFT-Large call per frame (+ warp/blend)."""
import json, sys, time, warnings, statistics
from pathlib import Path
import torch
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lpips
from flowutil import Flow, flow_and_mask
from metrics import Evaluator
from data import load_sequence, load_style, style_paths, ROOT
from methods import load_adain, load_attn, PerFrameAdaIN, AttnVideo

frames = load_sequence("blackswan", 60); style = load_style(style_paths("test")[0], 256)
net = load_adain(); flow = Flow("cuda")
def bench(fn, reps=5, per=None):
    per = per or frames.shape[0]            # number of items processed per fn() call
    fn(); fn()                                        # warm-up
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000 / per)
    return statistics.median(ts), min(ts), max(ts)
res = {}
pf = PerFrameAdaIN(net, "pf"); res["perframe"] = bench(lambda: pf.stylize("blackswan", frames, style))
n2, m2 = load_attn("checkpoints/b2_lt150.pt"); b2 = AttnVideo(n2, m2, "b2"); res["b2"] = bench(lambda: b2.stylize("blackswan", frames, style))
a, b = frames[1:2].cuda(), frames[0:1].cuda()
def fl():
    for _ in range(20): flow_and_mask(flow, a, b)
res["flow_and_mask_per_frame"] = bench(fl, 5, per=20)   # 2 RAFT-Large passes (fw + bw) + mask
out = {k: {"ms_median": v[0], "ms_min": v[1], "ms_max": v[2]} for k, v in res.items()}
t = {"B0": out["perframe"]["ms_median"], "control": out["perframe"]["ms_median"], "b1_lt150": out["perframe"]["ms_median"], "b1_lt150_s1": out["perframe"]["ms_median"],
     "b1_lt500": out["perframe"]["ms_median"], "b1_lt1500": out["perframe"]["ms_median"], "b2_lt150": out["b2"]["ms_median"], "b2_lt150_s1": out["b2"]["ms_median"]}
for al in ["0.1", "0.2", "0.3", "0.45", "0.6"]:
    t[f"B0+blend{al}"] = out["perframe"]["ms_median"] + out["flow_and_mask_per_frame"]["ms_median"]
json.dump({**{k: {"ms_total": v} for k, v in t.items()}, "_raw": out}, open(ROOT / "results" / "timing.json", "w"), indent=1)
print(json.dumps(out, indent=1)); print("GPU:", torch.cuda.get_device_name(0))
