"""Evaluation metrics for video style transfer.

TEMPORAL  (lower = more consistent)
    warp_mse / warp_l1 : warping error of Lai et al. (ECCV 2018) style. For frame t and reference frame t-k:
        E = sum_x M(x) * d(O_t(x), W(O_{t-k})(x)) / sum_x M(x)
      * flow comes from RAFT run on the ORIGINAL frames (never on the stylized ones)
      * M = forward-backward-consistent, non-motion-boundary, in-frame pixels (occlusions removed)
      * O are 8-bit quantised outputs in [0,1]; d = per-pixel mean over RGB of squared (mse) / absolute (l1) error
      * k=1 -> short-term, k=5 -> long-term
    Because blur or a constant image trivially lowers warp error, ALWAYS read it together with the
    quality metrics below.
QUALITY   (lower = better)
    content_lpips : LPIPS(AlexNet) between the stylized frame and its own content frame
    style_ms      : AdaIN style loss (mean/std distance, relu1_1..relu4_1) to the style image
    style_gram    : Gram-matrix distance (relu1_1..relu4_1) to the style image
  style_* use the AdaIN VGG-19 encoder (the network AdaIN itself is trained with), so they track what the model optimises;
  they are NOT a substitute for human judgement.
"""
import torch
import torch.nn.functional as F

from flowutil import warp, flow_and_mask
from adain import mean_std


def quantize8(x):
    """Clamp to [0,1] and round to 8-bit, exactly what saving a PNG would do."""
    return (x.clamp(0, 1) * 255).round() / 255


def warp_error(out_cur, out_prev, flow_bw, mask):
    """-> (mse, l1, valid_fraction) as python floats for one frame pair."""
    w, inside = warp(out_prev, flow_bw)
    m = mask * inside
    n = m.sum().clamp_min(1.0)
    d = out_cur - w
    mse = ((d ** 2).mean(1, keepdim=True) * m).sum() / n
    l1 = (d.abs().mean(1, keepdim=True) * m).sum() / n
    return mse.item(), l1.item(), m.mean().item()


def gram(f):
    b, c, h, w = f.shape
    x = f.reshape(b, c, h * w)
    return x.bmm(x.transpose(1, 2)) / (c * h * w)


def style_distances(feats_out, feats_style):
    ms, gr = 0.0, 0.0
    for fo, fs in zip(feats_out, feats_style):
        mo, so = mean_std(fo)
        ms_, ss_ = mean_std(fs)
        ms += F.mse_loss(mo, ms_) + F.mse_loss(so, ss_)
        gr += F.mse_loss(gram(fo), gram(fs))
    return float(ms), float(gr)


class Evaluator:
    def __init__(self, net, flow, lpips_model, cache_dir, device="cuda"):
        self.net, self.flow, self.lp, self.device = net, flow, lpips_model, device
        self.cache = cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)

    def flows(self, seq, frames, k):
        """Cached (flow_bw fp16 [T-k,2,H,W], mask uint8 [T-k,1,H,W]) between frame t and t-k of the ORIGINAL video."""
        f = self.cache / f"{seq}_n{frames.shape[0]}_k{k}.pt"      # frame count in the key: a partial video must never reuse a full-video cache
        if f.exists():
            d = torch.load(f, weights_only=True)
            assert d["flow"].shape[0] == frames.shape[0] - k, f"corrupt flow cache {f}"
            return d
        fl, mk = [], []
        for t in range(k, frames.shape[0]):
            cur = frames[t:t + 1].to(self.device)
            prev = frames[t - k:t - k + 1].to(self.device)
            bw, m = flow_and_mask(self.flow, cur, prev)
            fl.append(bw.half().cpu())
            mk.append((m > 0.5).to(torch.uint8).cpu())
        d = {"flow": torch.cat(fl), "mask": torch.cat(mk)}
        torch.save(d, f)
        return d

    @torch.no_grad()
    def temporal(self, seq, frames_orig, outputs, ks=(1, 5)):
        """outputs: [T,3,H,W] float in [0,1] (already 8-bit quantised). Returns dict of means."""
        res = {}
        for k in ks:
            if frames_orig.shape[0] <= k:
                continue
            d = self.flows(seq, frames_orig, k)
            mse, l1, vf = [], [], []
            for i, t in enumerate(range(k, outputs.shape[0])):
                a, b, c = warp_error(outputs[t:t + 1].to(self.device), outputs[t - k:t - k + 1].to(self.device),
                                     d["flow"][i:i + 1].to(self.device).float(),
                                     d["mask"][i:i + 1].to(self.device).float())
                mse.append(a); l1.append(b); vf.append(c)
            res[f"warp_mse_k{k}"] = sum(mse) / len(mse)
            res[f"warp_l1_k{k}"] = sum(l1) / len(l1)
            res[f"valid_frac_k{k}"] = sum(vf) / len(vf)
        return res

    @torch.no_grad()
    def quality(self, frames_orig, outputs, style_feats, stride=1):
        lp, ms, gr = [], [], []
        for t in range(0, outputs.shape[0], stride):
            o = outputs[t:t + 1].to(self.device)
            c = frames_orig[t:t + 1].to(self.device)
            lp.append(self.lp(o * 2 - 1, c * 2 - 1).item())
            x = o
            h, w = x.shape[-2:]
            ph, pw = (-h) % 8, (-w) % 8
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="reflect")
            a, b = style_distances(self.net.encode_all(x), style_feats)
            ms.append(a); gr.append(b)
        n = len(lp)
        return {"content_lpips": sum(lp) / n, "style_ms": sum(ms) / n, "style_gram": sum(gr) / n}
