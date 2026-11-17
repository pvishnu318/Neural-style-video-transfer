"""Adapters that run PUBLISHED video style-transfer models inside OUR evaluation pipeline (same videos, styles, flow, masks, metrics).
Third-party code lives in third_party/ (their licences apply; not redistributed by us).

ReReVST: Wang et al., IEEE TIP 2020 (official code github.com/daooshee/ReReVST-Code, final model style_net-TIP-final.pth).
  Mirrors the authors' own `generate_real_video.py`: reflect-pad by 64 px (+ to a multiple of 64), Sequence-Level Global Feature Sharing
  (every 8th frame + last frame), then per-frame transfer. Input/outputs are BGR; we convert from/to our RGB [0,1].
"""
import sys
from pathlib import Path
import cv2, numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent


class ReReVST:
    def __init__(self, use_global=True, device="cuda"):
        d = str(ROOT / "third_party" / "rerevst" / "test")
        if d not in sys.path:
            sys.path.insert(0, d)
        from framework import Stylization                          # their code
        self.fw = Stylization(str(ROOT / "third_party" / "rerevst_weights" / "models" / "style_net-TIP-final.pth"), True, use_global)
        self.use_global, self.name, self.device, self.uses_flow_at_inference = use_global, "ReReVST", device, False

    @staticmethod
    def _bgr(x):                                                    # [3,H,W] float 0..1 RGB -> HxWx3 uint8 BGR
        return cv2.cvtColor((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)

    def stylize(self, seq, frames, style):
        fw = self.fw
        fw.prepare_style(self._bgr(style[0].cpu()))
        T, H, W = frames.shape[0], frames.shape[-2], frames.shape[-1]
        newH = H + 128; newH += (64 - newH % 64) % 64
        newW = W + 128; newW += (64 - newW % 64) % 64
        pad = lambda im: cv2.copyMakeBorder(im, 64, newH - 64 - H, 64, newW - 64 - W, cv2.BORDER_REFLECT)
        imgs = [self._bgr(frames[t]) for t in range(T)]
        if self.use_global:
            fw.clean()
            for i in range(((T - 1) // 8)):
                fw.add(imgs[i * 8])
            fw.add(imgs[-1]); fw.compute()
        out = []
        for t in range(T):
            y = fw.transfer(pad(imgs[t]))[64:64 + H, 64:64 + W, :]     # float BGR 0..255
            y = cv2.cvtColor(np.clip(y, 0, 255).astype(np.float32), cv2.COLOR_BGR2RGB) / 255.0
            out.append(torch.from_numpy(y).permute(2, 0, 1).float())
        return torch.stack(out)


class MCCNet:
    """MCCNet (Deng et al., AAAI 2021). Official code github.com/diyiiyiii/MCCNet; weights decoder_iter_160000 / mcc_module_iter_160000 from the repo,
    vgg_normalised.pth from the authors' Drive link (byte-identical to the AdaIN VGG we already use, sha256 checked).
    Their released test_video.py cannot run (calls torch.uint8(...) / cv2.cvColor and its frame branch never executes), so we reimplement ITS inference
    core: relu4_1 features -> MCC_Module(content, style) -> decoder, one frame at a time (the method is flow-free and frame-independent by design).
    Input RGB [0,1]; style as given by our pipeline (256x256); H,W reflect-padded to multiples of 8 and cropped back."""

    def __init__(self, device="cuda"):
        import torch.nn as nn
        d = str(ROOT / "third_party" / "mccnet")
        if d not in sys.path:
            sys.path.insert(0, d)
        import net as mnet
        self.device, self.name, self.uses_flow_at_inference = device, "MCCNet", False
        vgg, dec = mnet.vgg, mnet.decoder
        network = mnet.Net(vgg, dec)
        vgg.load_state_dict(torch.load(ROOT / "third_party" / "mccnet_weights" / "vgg_normalised.pth", weights_only=True))
        dec.load_state_dict(torch.load(ROOT / "third_party" / "mccnet" / "decoder_iter_160000.pth", weights_only=True))
        network.mcc_module.load_state_dict(torch.load(ROOT / "third_party" / "mccnet" / "mcc_module_iter_160000.pth", weights_only=True))
        self.net = network.to(device).eval(); self.vgg = vgg.to(device).eval(); self.dec = dec.to(device).eval()
        L = list(vgg.children())
        self.encs = [nn.Sequential(*L[:4]), nn.Sequential(*L[4:11]), nn.Sequential(*L[11:18]), nn.Sequential(*L[18:31])]   # up to relu4_1
        for e in self.encs:
            e.to(device).eval()

    def _f4(self, x):
        for e in self.encs:
            x = e(x)
        return x

    @torch.no_grad()
    def stylize(self, seq, frames, style):
        sf = self._f4(style.to(self.device))
        out = []
        for t in range(frames.shape[0]):
            x = frames[t:t + 1].to(self.device); h, w = x.shape[-2:]
            ph, pw = (-h) % 8, (-w) % 8
            xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
            feat = self.net.mcc_module.MCCN(self._f4(xp), sf)      # == MCC_Module on the relu4_1 features (content_feats[-2] / style_feats[-2])
            y = self.dec(feat)[..., :h, :w]
            out.append(y.clamp(0, 1).cpu())
        return torch.cat(out)
