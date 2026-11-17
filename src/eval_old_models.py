"""Evaluate the friends' ORIGINAL per-style TransformerNet checkpoints (old repo) with OUR metrics on OUR test split, and
our new methods on the SAME three styles so the comparison is like-for-like.

Old model I/O (verified in old `scripts/run_temporal.py`): input = ImageNet-normalised RGB, output = normalised RGB -> de-normalise -> [0,1].
TransformerNet needs H,W multiples of 4; we reflect-pad and crop (the old script resampled the output instead).
Each per-style checkpoint is evaluated only with its own style image (cuphead / mosaic / starry_night = the old project's styles).
"""
import importlib.util, json, sys, time, warnings
from pathlib import Path
import torch, torch.nn.functional as F
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lpips
from adain import pad_to_multiple, crop_to
from flowutil import Flow
from metrics import Evaluator
from data import read_split, ROOT
from methods import load_adain, PerFrameAdaIN, FlowBlend
from evaluate import evaluate_method, summarize, reference_row, METRICS

OLD = ROOT.parent / "Fast-Neural-Style-Transfer"
spec = importlib.util.spec_from_file_location("old_models", OLD / "models.py"); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class OldPerStyle:
    def __init__(self, ckpt, name, device="cuda"):
        self.name, self.device, self.uses_flow_at_inference = name, device, False
        self.net = old.TransformerNet().to(device).eval()
        sd = torch.load(ckpt, map_location=device, weights_only=True)
        self.net.load_state_dict(sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd, strict=True)
        self.mean, self.std = MEAN.to(device), STD.to(device)

    @torch.no_grad()
    def stylize(self, seq, frames, style):                     # `style` unused: the style is baked into the checkpoint
        out = []
        for t in range(frames.shape[0]):
            x = frames[t:t + 1].to(self.device); h, w = x.shape[-2:]
            ph, pw = (-h) % 4, (-w) % 4
            xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
            y = self.net((xp - self.mean) / self.std) * self.std + self.mean
            out.append(y[..., :h, :w].clamp(0, 1).cpu())
        return torch.cat(out)


def run(name, method, ev, seqs, style_path, store, key):
    t0 = time.time()
    rows = evaluate_method(method, ev, seqs, [style_path], verbose=False)
    summ, _, _ = summarize(rows)
    store.setdefault(key, {})[name] = {"summary": summ, "rows": rows}
    print(f"== [{key}] {name} ({time.time()-t0:.0f}s): " + " | ".join(f"{k}={v['mean']:.5f}" for k, v in summ.items() if k in ("warp_mse_k1", "warp_mse_k5", "content_lpips", "style_ms")), flush=True)


def main():
    dev = "cuda"
    net = load_adain(); lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    ev = Evaluator(net, Flow(dev), lp, ROOT / "data" / "flow_cache", dev)
    seqs = read_split("test")
    out_path = ROOT / "results" / "eval_test_oldmodels.json"
    res = json.load(open(out_path)) if out_path.exists() else {}
    res["reference_original_video"] = reference_row(ev, seqs)
    ck = OLD / "checkpoints"
    sty = {s: ROOT / "data" / "styles" / "showcase" / f"{s}.jpg" for s in ("cuphead", "mosaic", "starry_night")}
    base = PerFrameAdaIN(net, "x")
    for s in ("cuphead", "mosaic", "starry_night"):
        # --- old project models (own style only)
        old_models = {"cuphead": ["cuphead_10000", "cuphead_temporal_best", "cuphead_temporal_m2", "cuphead_temporal_m2_e5", "cuphead_temporal_m3"],
                      "mosaic": ["mosaic_10000"], "starry_night": ["starry_night_10000"]}[s]
        for ckn in old_models:
            run("OLD:" + ckn, OldPerStyle(ck / f"{ckn}.pth", ckn), ev, seqs, sty[s], res, s)
            json.dump(res, open(out_path, "w"), indent=1)
        if s == "cuphead":      # the old project's inference-time "M2": fine-tuned model + flow-warped blend, alpha 0.35 (our occlusion mask)
            m = FlowBlend(OldPerStyle(ck / "cuphead_temporal_m2.pth", "m2"), ev, 0.35, "old_M2_blend0.35")
            run("OLD:cuphead_temporal_m2+blend0.35", m, ev, seqs, sty[s], res, s); json.dump(res, open(out_path, "w"), indent=1)
        # --- our methods on the same style
        ours = {"B0": PerFrameAdaIN(net, "B0"), "control": PerFrameAdaIN(load_adain("checkpoints/control.pt"), "control"),
                "B1_lt150": PerFrameAdaIN(load_adain("checkpoints/b1_lt150.pt"), "B1_lt150"),
                "B1_lt500": PerFrameAdaIN(load_adain("checkpoints/b1_lt500.pt"), "B1_lt500"),
                "B0+blend0.3": FlowBlend(base, ev, 0.3, "blend")}
        for nm, m in ours.items():
            run(nm, m, ev, seqs, sty[s], res, s)
            json.dump(res, open(out_path, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
