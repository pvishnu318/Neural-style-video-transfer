"""Evaluate published models (see external_methods.py) with the SAME protocol as evaluate.py.

    python src/eval_external.py --split test --methods rerevst --out results/eval_test_external.json [--timing] [--max-frames N]
"""
import argparse, json, sys, time, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lpips
from flowutil import Flow
from metrics import Evaluator
from data import read_split, load_sequence, load_style, style_paths, ROOT
from methods import load_adain
from evaluate import evaluate_method, summarize, pick_styles, reference_row


def build(name):
    from external_methods import ReReVST, MCCNet
    if name == "mccnet":
        return MCCNet()
    if name == "rerevst":
        return ReReVST(True)
    if name == "rerevst_noglobal":
        m = ReReVST(False); m.name = "ReReVST_noglobal"; return m
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--n-styles", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    net = load_adain(); lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    ev = Evaluator(net, Flow(dev), lp, ROOT / "data" / "flow_cache", dev)
    seqs, styles = read_split(a.split), pick_styles(a.split, a.n_styles)
    print(f"split={a.split}: {len(seqs)} sequences x {len(styles)} styles {[s.name for s in styles]}", flush=True)
    out = Path(a.out); res = json.load(open(out)) if out.exists() else {}
    res.setdefault("_meta", {}).update({"split": a.split, "sequences": seqs, "styles": [s.name for s in styles], "max_frames": a.max_frames})
    for n in a.methods:
        m = build(n); t0 = time.time()
        rows = evaluate_method(m, ev, seqs, styles, max_frames=a.max_frames)
        summ, _, _ = summarize(rows)
        res[m.name] = {"summary": summ, "rows": rows}
        if a.timing:                                   # ms/frame at 480p (blackswan, 60 frames), warm-up excluded, median of 3
            fr = load_sequence("blackswan", 60); st = load_style(style_paths("test")[0], 256)
            m.stylize("blackswan", fr[:16], st); ts = []
            for _ in range(3):
                torch.cuda.synchronize(); t = time.perf_counter(); m.stylize("blackswan", fr, st); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000 / 60)
            res[m.name]["timing"] = {"ms_per_frame_total": sorted(ts)[1], "repeats": ts, "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20}
        json.dump(res, open(out, "w"), indent=1)
        print(f"== {m.name} ({time.time()-t0:.0f}s): " + " | ".join(f"{k}={v['mean']:.5f}" for k, v in summ.items()), flush=True)


if __name__ == "__main__":
    main()
