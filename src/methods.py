"""Stylization methods. Every method exposes  stylize(seq_name, frames[T,3,H,W] cpu, style[1,3,h,w]) -> [T,3,H,W] cpu in [0,1]
(un-quantised; the evaluator quantises to 8 bit)."""
import torch

from adain import AdaINNet, adain, pad_to_multiple, crop_to
from flowutil import warp

W_VGG = "weights/vgg_normalised.pth"
W_DEC = "weights/decoder.pth"


def load_adain(decoder_path=W_DEC, device="cuda"):
    """decoder_path: pretrained decoder.pth, or one of our checkpoints (dict with key 'decoder')."""
    net = AdaINNet(W_VGG, None).to(device)
    sd = torch.load(decoder_path, map_location=device, weights_only=True)
    net.decoder.load_state_dict(sd["decoder"] if "decoder" in sd else sd)
    return net.eval()


def load_attn(path, device="cuda"):
    from temporal_module import LocalAttnFusion
    net = load_adain(path, device)
    mod = LocalAttnFusion().to(device)
    mod.load_state_dict(torch.load(path, map_location=device, weights_only=True)["module"])
    return net, mod.eval()


class PerFrameAdaIN:
    """B0 / control / B1: every frame stylised independently (the decoder may have been fine-tuned)."""

    def __init__(self, net, name, device="cuda"):
        self.net, self.name, self.device = net, name, device
        self.uses_flow_at_inference = False

    @torch.no_grad()
    def stylize(self, seq, frames, style):
        sf = self.net.encode(style.to(self.device))
        out = []
        for t in range(frames.shape[0]):
            x, hw = pad_to_multiple(frames[t:t + 1].to(self.device))
            y = crop_to(self.net.decode(adain(self.net.encode(x), sf)), hw)
            out.append(y.clamp(0, 1).cpu())
        return torch.cat(out)


class FlowBlend:
    """Post-hoc temporal smoothing of any per-frame method (the 'M2' idea of the old project, but with proper masks):
        out_t = (1 - a*M) * raw_t + a*M * W(out_{t-1})
    M = occlusion mask from flow computed on the ORIGINAL frames."""

    def __init__(self, base, evaluator, alpha, name):
        self.base, self.ev, self.alpha, self.name = base, evaluator, alpha, name
        self.uses_flow_at_inference = True
        self.device = base.device

    @torch.no_grad()
    def stylize(self, seq, frames, style):
        raw = self.base.stylize(seq, frames, style)
        d = self.ev.flows(seq, frames, 1)
        out = [raw[0:1].to(self.device)]
        for i, t in enumerate(range(1, frames.shape[0])):
            bw = d["flow"][i:i + 1].to(self.device).float()
            m = d["mask"][i:i + 1].to(self.device).float()
            prev_w, inside = warp(out[-1], bw)
            a = self.alpha * m * inside
            out.append(((1 - a) * raw[t:t + 1].to(self.device) + a * prev_w).clamp(0, 1))
        return torch.cat(out).cpu()


class AttnVideo:
    """B2: recurrent local-attention feature fusion; no optical flow at inference."""

    def __init__(self, net, module, name, device="cuda"):
        self.net, self.module, self.name, self.device = net, module, name, device
        self.uses_flow_at_inference = False

    @torch.no_grad()
    def stylize(self, seq, frames, style):
        sf = self.net.encode(style.to(self.device))
        out, prev_fc, prev_f = [], None, None
        for t in range(frames.shape[0]):
            x, hw = pad_to_multiple(frames[t:t + 1].to(self.device))
            fc = self.net.encode(x)
            tgt = adain(fc, sf)
            f = tgt if prev_f is None else self.module(tgt, fc, prev_fc, prev_f)[0]
            out.append(crop_to(self.net.decode(f), hw).clamp(0, 1).cpu())
            prev_fc, prev_f = fc, f
        return torch.cat(out)
