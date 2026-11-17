"""Evaluate methods on a DAVIS split x style split. Writes per-(sequence,style) rows and a summary with bootstrap CIs.

    python src/evaluate.py --split val  --methods B0 B0+blend0.3 ...      (used for choosing settings)
    python src/evaluate.py --split test --methods ...                     (final numbers; do not tune on this)
"""
import argparse, json, sys, time, warnings
from pathlib import Path
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lpips
from adain import AdaINNet
from flowutil import Flow
from metrics import Evaluator, quantize8
from data import read_split, load_sequence, style_paths, load_style, ROOT
from methods import PerFrameAdaIN, FlowBlend, AttnVideo, load_adain, load_attn

METRICS = ["warp_mse_k1", "warp_l1_k1", "warp_mse_k5", "warp_l1_k5", "content_lpips", "style_ms", "style_gram"]


def pick_styles(split, n):
    ps = style_paths(split)
    idx = np.linspace(0, len(ps) - 1, n).round().astype(int)
    return [ps[i] for i in idx]


def evaluate_method(method, ev, seqs, styles, style_side=256, max_frames=0, verbose=True):
    rows = []
    for seq in seqs:
        frames = load_sequence(seq, max_frames)
        for sp in styles:
            style = load_style(sp, style_side)
            sf = ev.net.encode_all(style.to(ev.device))
            out = quantize8(method.stylize(seq, frames, style))
            r = {"seq": seq, "style": sp.name, "frames": int(frames.shape[0])}
            r.update(ev.temporal(seq, frames, out))
            r.update(ev.quality(frames, out, sf, stride=2))
            rows.append(r)
        if verbose:
            print(f"  [{method.name}] {seq}: " + ", ".join(f"{m}={np.mean([x[m] for x in rows if x['seq']==seq]):.5f}" for m in ("warp_mse_k1", "content_lpips", "style_ms")), flush=True)
    return rows


def reference_row(ev, seqs):
    """Warp error of the ORIGINAL video (noise floor of the metric: natural photometric change + flow error)."""
    out = {}
    for k in (1, 5):
        v = []
        for seq in seqs:
            fr = load_sequence(seq)
            v.append(ev.temporal(seq, fr, fr, ks=(k,))[f"warp_mse_k{k}"])
        out[f"warp_mse_k{k}"] = float(np.mean(v))
    return out


def summarize(rows, n_boot=2000, seed=0):
    """Average styles within a sequence first (sequence = independent unit), then bootstrap over sequences."""
    seqs = sorted({r["seq"] for r in rows})
    per_seq = {m: np.array([np.mean([r[m] for r in rows if r["seq"] == s]) for s in seqs]) for m in METRICS}
    rng = np.random.default_rng(seed)
    summ = {}
    for m, v in per_seq.items():
        boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
        summ[m] = {"mean": float(v.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
    return summ, seqs, per_seq


def timing(method, ev, frames_n=60, seq="blackswan"):
    """ms/frame at 480p incl. warm-up exclusion. Flow cost is added for methods that need flow at inference."""
    frames = load_sequence(seq, frames_n)
    style = load_style(style_paths("test")[0], 256)
    method.stylize(seq, frames[:20], style); method.stylize(seq, frames[:20], style)   # warm-up (cudnn autotune etc.)
    torch.cuda.synchronize(); t = time.perf_counter()
    method.stylize(seq, frames, style)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / frames.shape[0] * 1000
    flow_ms = 0.0
    if getattr(method, "uses_flow_at_inference", False):
        a, b = frames[1:2].cuda(), frames[0:1].cuda()
        ev.flow(a, b); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(10): ev.flow(a, b)
        torch.cuda.synchronize(); flow_ms = (time.perf_counter() - t) / 10 * 1000
    return {"ms_per_frame_model": ms, "ms_per_frame_flow": flow_ms, "ms_per_frame_total": ms + flow_ms}


def build_methods(names, ev):
    ms = {}
    for n in names:
        if n == "B0":
            ms[n] = PerFrameAdaIN(ev.net, "B0")
        elif n.startswith("B0+blend"):
            ms[n] = FlowBlend(PerFrameAdaIN(ev.net, "B0"), ev, float(n.replace("B0+blend", "")), n)
        elif n.startswith("ckpt:"):                      # ckpt:<name>:<path-to-decoder-or-model>  (fine-tuned per-frame decoders)
            _, nm, path = n.split(":", 2)
            ms[nm] = PerFrameAdaIN(load_adain(path), nm)
        elif n.startswith("attn:"):                      # attn:<name>:<path>
            _, nm, path = n.split(":", 2)
            net2, mod = load_attn(path)
            ms[nm] = AttnVideo(net2, mod, nm)
        else:
            raise ValueError(n)
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--n-styles", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    net = load_adain()
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    ev = Evaluator(net, Flow(dev), lp, ROOT / "data" / "flow_cache", dev)
    seqs = read_split(a.split)
    styles = pick_styles(a.split, a.n_styles)
    print(f"split={a.split}: {len(seqs)} sequences x {len(styles)} unseen styles  | styles: {[s.name for s in styles]}")
    out = Path(a.out or ROOT / "results" / f"eval_{a.split}.json")
    allres = json.load(open(out)) if out.exists() else {}
    allres.setdefault("_meta", {}).update({"split": a.split, "sequences": seqs, "styles": [s.name for s in styles],
                                           "max_frames": a.max_frames})
    if "reference_original_video" not in allres:
        allres["reference_original_video"] = reference_row(ev, seqs)
    for name, m in build_methods(a.methods, ev).items():
        t0 = time.time()
        rows = evaluate_method(m, ev, seqs, styles, max_frames=a.max_frames)
        summ, _, _ = summarize(rows)
        allres[name] = {"summary": summ, "rows": rows}
        if a.timing:
            allres[name]["timing"] = timing(m, ev)
        json.dump(allres, open(out, "w"), indent=1)
        print(f"== {name} ({time.time()-t0:.0f}s): " + " | ".join(f"{k}={v['mean']:.5f}" for k, v in summ.items()), flush=True)
    print("reference (original video):", allres["reference_original_video"])


if __name__ == "__main__":
    main()
