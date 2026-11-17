"""Clean speed numbers for published models and the old per-style model (run with the GPU otherwise idle).
480p, 60 frames of 'blackswan', warm-up excluded, median of 5 repeats; frames start on the CPU like in time_methods.py."""
import json, sys, time, statistics, warnings
from pathlib import Path
import torch
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_sequence, load_style, style_paths, ROOT
from external_methods import ReReVST, MCCNet
from eval_old_models import OldPerStyle, OLD

frames = load_sequence("blackswan", 60); style = load_style(style_paths("test")[0], 256)
def bench(m, reps=5):
    m.stylize("blackswan", frames[:20], style); m.stylize("blackswan", frames[:20], style)
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter(); m.stylize("blackswan", frames, style); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1000 / frames.shape[0])
    return {"ms_total": statistics.median(ts), "ms_min": min(ts), "ms_max": max(ts)}
out = {}
for name, make in [("MCCNet", MCCNet), ("ReReVST", lambda: ReReVST(True)),
                   ("OLD:per-style TransformerNet", lambda: OldPerStyle(OLD / "checkpoints" / "cuphead_10000.pth", "old"))]:
    torch.cuda.reset_peak_memory_stats(); m = make(); out[name] = bench(m); out[name]["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
    print(name, out[name], flush=True); del m; torch.cuda.empty_cache()
json.dump(out, open(ROOT / "results" / "timing_external.json", "w"), indent=1)
