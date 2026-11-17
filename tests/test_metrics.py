"""Sanity tests of the measuring tool itself (run before trusting any number):  python tests/test_metrics.py"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import torch, numpy as np
from flowutil import Flow, warp, occlusion_mask, flow_and_mask
from metrics import warp_error, quantize8
from data import load_sequence
dev = "cuda"
flow = Flow(dev)
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name, info)

seq = load_sequence("blackswan", 12).to(dev)          # [T,3,480,854]
base = seq[5:6]

# T1: warp is exact for an integer translation with an analytic flow.
dx, dy = 7, 3
cur = base[..., dy:, dx:]                              # cur(x) = base(x + d)  -> content moved by -d
prev = base[..., :-dy, :-dx]
h, w = cur.shape[-2:]
f = torch.zeros(1, 2, h, w, device=dev); f[:, 0] = -dx * 0 + 0                       # cur(x)=prev(x) here (same crop offset trick)
# build the real test: prev = base, cur = base shifted right by d  => cur(x) = base(x - d) => flow_bw(cur->prev) = -d
prev = base[..., :, :]; cur = torch.zeros_like(base); cur[..., :, dx:] = base[..., :, :-dx]; cur[..., dy:, :] = cur[..., :-dy, :].clone()
cur[..., :dy, :] = 0
fb = torch.zeros(1, 2, *base.shape[-2:], device=dev); fb[:, 0] = -dx; fb[:, 1] = -dy
wp, inside = warp(prev, fb)
inner = (slice(None), slice(None), slice(dy + 2, None), slice(dx + 2, None))
err = (wp - cur)[inner].abs().max().item()
check("T1 warp reproduces an integer shift (interior max abs err)", err < 1e-3, f"err={err:.2e} (tolerance 1e-3 = 1/4 of one 8-bit level; float32 coordinate rounding)")
check("T1b in-frame mask drops exactly the pixels sampled outside", abs(inside[0, 0, :, :dx].sum().item()) == 0 and inside[0, 0, dy:, dx:].min().item() == 1)

# T2: RAFT recovers a known shift on a real frame.
est = flow(cur, prev)                                   # flow(cur -> prev)
med = est[0, :, 40:-40, 40:-40].reshape(2, -1).median(1).values.cpu().numpy()
check("T2 RAFT recovers known shift (expect dx=-7, dy=-3)", abs(med[0] + dx) < 0.5 and abs(med[1] + dy) < 0.5, f"median flow={med}")

# T3: warp error on the ORIGINAL frames is small; on unrelated content it is large.
def seq_err(a, b):
    bw, m = flow_and_mask(flow, a, b)
    return warp_error(a, b, bw, m)
e_true = seq_err(seq[6:7], seq[5:6])
other = load_sequence("bmx-trees", 3).to(dev)[1:2]
e_other = seq_err(seq[6:7], other)
check("T3 consecutive frames << unrelated frames", e_true[0] < 0.2 * e_other[0], f"true={e_true[0]:.5f} unrelated={e_other[0]:.5f}")

# T4: occlusion mask. Static textured background + a square moving 24px: mask must exclude the
#     square's leading/trailing bands but keep far-away background.
g = torch.Generator(device=dev).manual_seed(0)
bg = torch.nn.functional.interpolate(torch.rand(1, 3, 60, 107, device=dev, generator=g), size=(480, 856), mode="bicubic").clamp(0, 1)
sq = torch.rand(1, 3, 120, 120, device=dev, generator=g)
def frame(x0):
    im = bg.clone(); im[..., 180:300, x0:x0 + 120] = sq; return im
a_, b_ = frame(300), frame(324)                          # square moves right by 24 px
bw, m = flow_and_mask(flow, b_, a_)
far = m[0, 0, 20:100, 20:200].mean().item()
lead = m[0, 0, 200:280, 324 + 120:324 + 120 + 10].mean().item()
check("T4 occlusion mask keeps far background", far > 0.9, f"kept={far:.2f}")
check("T4 mask removes the newly revealed (disoccluded) band behind the moving square", m[0, 0, 190:290, 300:324].mean().item() < 0.2,
      f"revealed band kept={m[0,0,190:290,300:324].mean().item():.2f}")

# T5: the metric ranks a consistent "stylisation" below a flickering one (real flow, real frames).
frames = seq[:10]
def run(fn):
    es = []
    for t in range(1, 10):
        bw, m = flow_and_mask(flow, frames[t:t+1], frames[t-1:t])
        es.append(warp_error(fn(frames[t:t+1], t), fn(frames[t-1:t], t - 1), bw, m)[0])
    return float(np.mean(es))
consistent = run(lambda x, t: quantize8(1 - x))                           # fixed per-pixel colour map
gen = torch.Generator(device=dev).manual_seed(1)
flicker = run(lambda x, t: quantize8(x + 0.15 * torch.randn(x.shape, device=dev, generator=gen)))  # independent noise per frame
check("T5 consistent colour-map << per-frame noise flicker", consistent < 0.3 * flicker, f"consistent={consistent:.5f} flicker={flicker:.5f}")
print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
