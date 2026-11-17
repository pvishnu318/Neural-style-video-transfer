"""Fine-tune the AdaIN decoder (and optionally the B2 attention module) on DAVIS-train clips x train styles.

    modes
      control : content+style loss only            (same data/steps/optimiser as b1; isolates 'extra training' from 'temporal loss')
      b1      : + flow-warped temporal loss          (per-frame decoder at inference, no extra inference cost)
      b2      : + temporal loss + LocalAttnFusion    (recurrent feature fusion, no flow at inference)
Loss = lc*content + ls*style + lt*temporal  with AdaIN's own definitions for content/style (lc=1, ls=10).
The share of every weighted term is logged so no term can silently be negligible.
"""
import argparse, json, sys, time, warnings
from pathlib import Path
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adain import AdaINNet, adain, mean_std
from flowutil import Flow, flow_and_mask, warp
from data import read_split, ClipDataset, ROOT
from temporal_module import LocalAttnFusion


def style_loss(feats_out, feats_style):
    s = 0.0
    for fo, fs in zip(feats_out, feats_style):
        mo, so = mean_std(fo); ms, ss = mean_std(fs)
        s = s + F.mse_loss(mo, ms) + F.mse_loss(so, ss)
    return s


def masked_mse(a, b, m):
    return (((a - b) ** 2).mean(1, keepdim=True) * m).sum() / m.sum().clamp_min(1.0)


def forward_clip(net, module, clip, style):
    """clip [B,T,3,H,W], style [B,3,S,S] -> list of outputs (unclamped) and AdaIN content targets."""
    B, T = clip.shape[:2]
    sf_all = net.encode_all(style)
    sf = sf_all[-1]
    outs, targets, gates = [], [], []
    prev_fc, prev_f = None, None
    for t in range(T):
        fc = net.encode(clip[:, t])
        tgt = adain(fc, sf)
        if module is not None and prev_f is not None:
            f, g = module(tgt, fc, prev_fc, prev_f)
            gates.append(g.mean().item())
        else:
            f = tgt
        outs.append(net.decode(f)); targets.append(tgt)
        prev_fc, prev_f = fc, f
    return outs, targets, sf_all, gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["control", "b1", "b2"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--lt", type=float, default=0.0, help="temporal loss weight (ignored for control)")
    ap.add_argument("--lc", type=float, default=1.0)
    ap.add_argument("--ls", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--clip-len", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--init", default=None, help="optional checkpoint to continue from")
    a = ap.parse_args()
    if a.mode == "control":
        a.lt = 0.0
    torch.manual_seed(a.seed)
    dev = "cuda"
    net = AdaINNet(ROOT / "weights/vgg_normalised.pth", ROOT / "weights/decoder.pth").to(dev)
    module = LocalAttnFusion().to(dev) if a.mode == "b2" else None
    if a.init:
        ck = torch.load(a.init, map_location=dev, weights_only=True)
        net.decoder.load_state_dict(ck["decoder"])
        if module is not None and "module" in ck:
            module.load_state_dict(ck["module"])
    flow = Flow(dev) if a.lt > 0 else None
    params = list(net.decoder.parameters()) + (list(module.parameters()) if module is not None else [])
    opt = torch.optim.Adam(params, lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr * 0.1)
    ds = ClipDataset(read_split("train"), "train", a.clip_len, a.crop)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, persistent_workers=a.workers > 0)
    out_dir = ROOT / "checkpoints"; out_dir.mkdir(exist_ok=True)
    log_path = ROOT / "logs" / f"train_{a.name}.jsonl"
    log_path.write_text("")
    json.dump(vars(a), open(ROOT / "logs" / f"train_{a.name}.args.json", "w"), indent=1)
    print("trainable params: %.3fM" % (sum(p.numel() for p in params) / 1e6), flush=True)
    net.train()
    if module is not None:
        module.train()
    acc, t0, step = {}, time.time(), 0
    for clip, style in dl:
        step += 1
        clip, style = clip.to(dev, non_blocking=True), style.to(dev, non_blocking=True)
        outs, targets, sf_all, gates = forward_clip(net, module, clip, style)
        lcont = sum(F.mse_loss(net.encode(o), t) for o, t in zip(outs, targets)) / len(outs)
        lsty = sum(style_loss(net.encode_all(o), sf_all) for o in outs) / len(outs)
        ltemp = torch.zeros((), device=dev)
        if a.lt > 0:
            terms = []
            for t in range(1, clip.shape[1]):
                bw, m = flow_and_mask(flow, clip[:, t], clip[:, t - 1])      # flow on the ORIGINAL frames
                wp, inside = warp(outs[t - 1].clamp(0, 1), bw)
                terms.append(masked_mse(outs[t].clamp(0, 1), wp, m * inside))
            ltemp = sum(terms) / len(terms)
        loss = a.lc * lcont + a.ls * lsty + a.lt * ltemp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step(); sched.step()
        rec = {"content": lcont.item(), "style": lsty.item(), "temporal": ltemp.item(), "total": loss.item(),
               "w_content": a.lc * lcont.item(), "w_style": a.ls * lsty.item(), "w_temporal": a.lt * ltemp.item()}
        if gates:
            rec["gate_mean"] = sum(gates) / len(gates)
        if not torch.isfinite(loss):
            print("non-finite loss, stopping", rec); break
        for k, v in rec.items():
            acc[k] = acc.get(k, 0.0) + v
        if step % a.log_every == 0:
            row = {"step": step, "seconds": time.time() - t0, "lr": sched.get_last_lr()[0]}
            row.update({k: v / a.log_every for k, v in acc.items()}); acc = {}
            tot = row["w_content"] + row["w_style"] + row["w_temporal"]
            row["share_temporal"] = row["w_temporal"] / tot
            open(log_path, "a").write(json.dumps(row) + "\n")
            print(f"step {step}/{a.steps} total={row['total']:.3f} content={row['w_content']:.3f} style={row['w_style']:.3f} "
                  f"temporal(raw)={row['temporal']:.5f} (weighted {row['w_temporal']:.3f}, {100*row['share_temporal']:.1f}% of loss)"
                  + (f" gate={row['gate_mean']:.3f}" if "gate_mean" in row else "") + f" {row['seconds']:.0f}s", flush=True)
        if step >= a.steps:
            break
    ck = {"decoder": net.decoder.state_dict(), "args": vars(a)}
    if module is not None:
        ck["module"] = module.state_dict()
    torch.save(ck, out_dir / f"{a.name}.pt")
    print("saved", out_dir / f"{a.name}.pt", flush=True)


if __name__ == "__main__":
    main()
