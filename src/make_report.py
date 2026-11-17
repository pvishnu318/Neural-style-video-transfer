"""Final tables (markdown + csv) and the consistency-vs-style trade-off chart from results/eval_test*.json."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent / "results"
d = json.load(open(R / "eval_test.json"))
extra = R / "eval_test_blend_extra.json"
if extra.exists():
    e = json.load(open(extra)); d.update({k: v for k, v in e.items() if not k.startswith("_") and k != "reference_original_video"})
for fn in ("eval_test_external.json", "eval_test_external_mccnet.json"):          # published models run through OUR pipeline
    fp = R / fn
    if fp.exists():
        d.update({k: v for k, v in json.load(open(fp)).items() if not k.startswith("_") and k != "reference_original_video"})
ref = d["reference_original_video"]
tim = json.load(open(R / "timing.json")) if (R / "timing.json").exists() else {}
if (R / "timing_external.json").exists():
    tim.update(json.load(open(R / "timing_external.json")))
names = {"ReReVST": "ReReVST (published, TIP 2020)", "MCCNet": "MCCNet (published, AAAI 2021)", "B0": "B0 pretrained AdaIN", "control": "control (fine-tuned, no temporal loss)", "b1_lt150": "B1 lt=150 (seed 0)",
         "b1_lt150_s1": "B1 lt=150 (seed 1)", "b1_lt500": "B1 lt=500", "b1_lt1500": "B1 lt=1500",
         "b2_lt150": "B2 lt=150 (seed 0)", "b2_lt150_s1": "B2 lt=150 (seed 1)"}
for k in d:
    if k.startswith("B0+blend"): names[k] = f"B0 + flow blend {k.replace('B0+blend','')}"
order = ["B0"] + sorted([k for k in d if k.startswith("B0+blend")], key=lambda k: float(k.replace("B0+blend", ""))) + \
        ["control", "b1_lt150", "b1_lt150_s1", "b1_lt500", "b1_lt1500", "b2_lt150", "b2_lt150_s1", "MCCNet", "ReReVST"]
order = [k for k in order if k in d]
tim.setdefault("MCCNet", tim.get("MCCNet", {})); 
cols = [("warp_mse_k1", "warp err (1 fr)"), ("warp_mse_k5", "warp err (5 fr)"), ("content_lpips", "content LPIPS"), ("style_ms", "style dist"), ("style_gram", "gram dist")]
lines = ["| method | " + " | ".join(c[1] for c in cols) + " | ms/frame | needs flow at inference |", "|" + "---|" * (len(cols) + 3)]
lines.append(f"| original video (floor) | {ref['warp_mse_k1']:.5f} | {ref['warp_mse_k5']:.5f} | - | - | - | - | - |")
csv = ["method," + ",".join(c[0] for c in cols) + ",k1_ci_lo,k1_ci_hi,ms_per_frame"]
for k in order:
    s = d[k]["summary"]; ms = tim.get(k, {}).get("ms_total") or d[k].get("timing", {}).get("ms_per_frame_total")
    cell = lambda m: f"{s[m]['mean']:.5f}" if m in ("warp_mse_k1", "warp_mse_k5", "style_gram") else f"{s[m]['mean']:.4f}"
    k1 = s["warp_mse_k1"]["ci95"]
    lines.append(f"| {names[k]} | {cell('warp_mse_k1')} [{k1[0]:.5f}, {k1[1]:.5f}] | {cell('warp_mse_k5')} | {cell('content_lpips')} | {cell('style_ms')} | {cell('style_gram')} | "
                 f"{'%.0f' % ms if ms else 'n/a'} | {'yes (RAFT)' if k.startswith('B0+blend') else 'no'} |")
    csv.append(f"{k}," + ",".join(f"{s[c[0]]['mean']:.6f}" for c in cols) + f",{k1[0]:.6f},{k1[1]:.6f},{ms if ms else ''}")
(R / "table_test.md").write_text("\n".join(lines) + "\n"); (R / "table_test.csv").write_text("\n".join(csv) + "\n")
print("\n".join(lines))

# trade-off chart: x = warping error (lower = smoother), y = style distance (lower = closer to style image)
INK, MUT, BG = "#0b0b0b", "#52514e", "#fcfcfb"
blend = ["B0"] + sorted([k for k in d if k.startswith("B0+blend")], key=lambda k: float(k.replace("B0+blend", "")))
b1 = ["control", "b1_lt150", "b1_lt500", "b1_lt1500"]
lab_of = {"B0": "B0", "control": "control", "b1_lt150": "lt150", "b1_lt500": "lt500", "b1_lt1500": "lt1500"}
def xy(k): return d[k]["summary"]["warp_mse_k1"]["mean"], d[k]["summary"]["style_ms"]["mean"]
def draw(ax, zoom):
    for keys, col, mk, lab, off in [(blend, "#eb6834", "o", "B0 + flow blend, strength 0.1-0.6 (RAFT at inference, ~286 ms/frame)", (6, 7)),
                                    (b1, "#2a78d6", "s", "control, B1 lt=150/500/1500 (no flow at inference, ~59 ms/frame)", (-4, -15))]:
        xs, ys = zip(*[xy(k) for k in keys]); ax.plot(xs, ys, "-", color=col, lw=2, zorder=2, label=lab)
        ax.scatter(xs, ys, s=44, color=col, marker=mk, edgecolor=BG, linewidth=1.5, zorder=3)
        for k in keys:
            x, y = xy(k)
            t = lab_of.get(k, k.replace("B0+blend", ""))
            if zoom and (x > 0.0125 or y > 0.56): continue
            if not zoom and k in ("control", "b1_lt150"): continue
            ax.annotate(t, (x, y), xytext=off, textcoords="offset points", color=INK, fontsize=8)
    for k, mk, c, lab in [("b1_lt150_s1", "D", "#2a78d6", "B1 lt=150, seed 1"), ("b2_lt150", "^", "#6250d6", "B2 lt=150, seed 0"), ("b2_lt150_s1", "^", "#6250d6", "B2 lt=150, seed 1")]:
        x, y = xy(k); ax.scatter(x, y, s=58, color=c, marker=mk, edgecolor=BG, linewidth=1.5, zorder=4, label=lab if k != "b2_lt150_s1" else "B2 lt=150, seed 1")
    for k, mk, c, lab in [("MCCNet", "P", "#e87ba4", "MCCNet (published)"), ("ReReVST", "X", "#1baf7a", "ReReVST (published)")]:
        if k in d:
            x, y = xy(k)
            if zoom and not (0.0038 <= x <= 0.0112 and y <= 0.62):
                continue
            ax.scatter(x, y, s=70, color=c, marker=mk, edgecolor=BG, linewidth=1.2, zorder=4, label=lab); ax.annotate(k, (x, y), xytext=(6, 6), textcoords="offset points", color=INK, fontsize=8)
    x, y = xy("B0"); ax.scatter(x, y, s=70, color=INK, zorder=5)
    ax.set_facecolor(BG)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#c9c8c2")
    ax.tick_params(colors=MUT); ax.grid(True, color="#e6e5e0", lw=.7, zorder=0)
    ax.set_xlabel("warping error, 1 frame (lower = smoother)", color=MUT); ax.set_ylabel("style distance (lower = closer to style)", color=MUT)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor=BG, gridspec_kw={"width_ratios": [1, 1.05]})
draw(ax1, False); ax1.set_xscale("log"); ax1.set_title("All methods (log x). Original video = dashed line", color=INK, fontsize=10.5, loc="left")
ax1.axvline(ref["warp_mse_k1"], color=MUT, lw=1, ls=(0, (4, 3)))
draw(ax2, True); ax2.set_xlim(0.0038, 0.0112); ax2.set_ylim(0.225, 0.62); ax2.set_title("Zoom: where the fine-tuned models live", color=INK, fontsize=10.5, loc="left")
h, l = ax1.get_legend_handles_labels(); seen = {}
for hh, ll in zip(h, l): seen.setdefault(ll, hh)
fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=8.5, labelcolor=INK)
fig.suptitle("Smoothness vs style on 15 test videos x 6 unseen styles - lower-left is better", color=INK, fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.2, 1, 0.95)); fig.savefig(R / "tradeoff_test.png", dpi=160, facecolor=BG); print("saved chart")


# ---- old project's per-style models (own style) + our methods on the SAME three styles
op = R / "eval_test_oldmodels.json"
if op.exists():
    o = json.load(open(op)); oref = o.pop("reference_original_video", ref)
    L = ["| style | method | warp err 1fr [95% CI] | warp err 5fr | content LPIPS | style dist |", "|---|---|---|---|---|---|"]
    for sty, ms_ in o.items():
        for nm, v in ms_.items():
            s_ = v["summary"]; k1 = s_["warp_mse_k1"]["ci95"]
            L.append(f"| {sty} | {nm} | {s_['warp_mse_k1']['mean']:.5f} [{k1[0]:.5f}, {k1[1]:.5f}] | {s_['warp_mse_k5']['mean']:.5f} | {s_['content_lpips']['mean']:.4f} | {s_['style_ms']['mean']:.4f} |")
    NL = chr(10)
    L.append(NL + "Original video floor: warp err 1fr %.5f, 5fr %.5f. Style distance is only comparable WITHIN one style (rows with the same style)." % (oref["warp_mse_k1"], oref["warp_mse_k5"]))
    (R / "table_old_models.md").write_text(NL.join(L) + NL); print(NL.join(L))
