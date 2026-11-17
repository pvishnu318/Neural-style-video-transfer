"""Qualitative outputs on TEST videos with the 3 showcase styles (never used for selection).
Per (video, style): a figure with rows = methods, cols = frame t, frame t+1, flow-compensated error |O_{t+1} - W(O_t)| x4 on valid pixels
(the quantity the warping-error metric averages; black = consistent, bright = flicker), and an mp4 per method."""
import sys, warnings
from pathlib import Path
import numpy as np, torch, cv2
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lpips
from flowutil import Flow, warp
from metrics import Evaluator, quantize8
from data import load_sequence, load_style, ROOT
from methods import load_adain, PerFrameAdaIN, FlowBlend, AttnVideo, load_attn

VIDS = ["camel", "bmx-trees", "libby"]
STYLES = ["cuphead", "mosaic", "starry_night"]
T0 = 30
out_root = ROOT / "results" / "qualitative"; out_root.mkdir(exist_ok=True, parents=True)
net = load_adain()
lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval()
ev = Evaluator(net, Flow("cuda"), lp, ROOT / "data" / "flow_cache", "cuda")
base = PerFrameAdaIN(net, "B0")
methods = {
    "B0 (pretrained)": base,
    "control": PerFrameAdaIN(load_adain("checkpoints/control.pt"), "control"),
    "B1 lt150": PerFrameAdaIN(load_adain("checkpoints/b1_lt150.pt"), "b1"),
    "B1 lt500": PerFrameAdaIN(load_adain("checkpoints/b1_lt500.pt"), "b1b"),
    "B0 + flow blend 0.3": FlowBlend(base, ev, 0.3, "blend"),
}
for v in VIDS:
    frames = load_sequence(v)
    for sn in STYLES:
        style = load_style(ROOT / "data" / "styles" / "showcase" / f"{sn}.jpg", 256)
        rows = []
        for mn, m in methods.items():
            out = quantize8(m.stylize(v, frames, style))
            slug = mn.lower().replace(" (pretrained)", "").replace(" + flow blend ", "_blend").replace(" ", "_").replace("=", "")
            vw = cv2.VideoWriter(str(out_root / f"{v}_{sn}_{slug}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (out.shape[-1], out.shape[-2]))
            for f in out:
                vw.write(cv2.cvtColor((f.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            vw.release()
            a, b = out[T0], out[T0 + 1]
            d = ev.flows(v, frames, 1); i = T0           # index i <-> frame pair (T0, T0+1)
            w, inside = warp(a[None].cuda(), d["flow"][i:i + 1].cuda().float())
            valid = (d["mask"][i:i + 1].cuda().float() * inside)[0]
            diff = ((b.cuda() - w[0]).abs().mean(0, keepdim=True) * valid).repeat(3, 1, 1).mul(4).clamp(0, 1).cpu()
            tile = torch.cat([a, b, diff], 2)
            tile = (tile.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            tile = cv2.resize(tile, (tile.shape[1] // 2, tile.shape[0] // 2), interpolation=cv2.INTER_AREA)
            cv2.rectangle(tile, (0, 0), (250, 26), (0, 0, 0), -1)
            cv2.putText(tile, mn, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            rows.append(tile)
        cv2.imwrite(str(out_root / f"{v}_{sn}_frames_warp_error.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
        print("done", v, sn, flush=True)
