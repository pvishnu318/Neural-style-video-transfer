"""AdaIN (Huang & Belongie, ICCV 2017) in PyTorch, loading the pretrained weights of
naoto0804/pytorch-AdaIN (vgg_normalised.pth + decoder.pth).

CONVENTION (used by every script in this project):
    * images are RGB float tensors in [0, 1], shape [B, 3, H, W]
    * there is NO other normalisation: the VGG file already contains the mean/std
      normalisation inside its first 1x1 convolution.
    * the decoder output is NOT clamped (clamp only when saving / measuring images)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _vgg():
    pad = lambda: nn.ReflectionPad2d((1, 1, 1, 1))
    pool = lambda: nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True)
    cfg = [("c", 3, 64), ("c", 64, 64), "p", ("c", 64, 128), ("c", 128, 128), "p",
           ("c", 128, 256), ("c", 256, 256), ("c", 256, 256), ("c", 256, 256), "p",
           ("c", 256, 512), ("c", 512, 512), ("c", 512, 512), ("c", 512, 512), "p",
           ("c", 512, 512), ("c", 512, 512), ("c", 512, 512), ("c", 512, 512)]
    layers = [nn.Conv2d(3, 3, (1, 1))]
    for item in cfg:
        if item == "p":
            layers.append(pool())
        else:
            _, i, o = item
            layers += [pad(), nn.Conv2d(i, o, (3, 3)), nn.ReLU()]
    return nn.Sequential(*layers)


def _decoder():
    pad = lambda: nn.ReflectionPad2d((1, 1, 1, 1))
    up = lambda: nn.Upsample(scale_factor=2, mode="nearest")
    def c(i, o, relu=True):
        return [pad(), nn.Conv2d(i, o, (3, 3))] + ([nn.ReLU()] if relu else [])
    return nn.Sequential(
        *c(512, 256), up(), *c(256, 256), *c(256, 256), *c(256, 256), *c(256, 128), up(),
        *c(128, 128), *c(128, 64), up(), *c(64, 64), *c(64, 3, relu=False))


def mean_std(feat, eps=1e-5):
    b, c = feat.shape[:2]
    flat = feat.reshape(b, c, -1)
    mean = flat.mean(2).view(b, c, 1, 1)
    std = (flat.var(2, unbiased=False) + eps).sqrt().view(b, c, 1, 1)
    return mean, std


def adain(content, style):
    """Align the channel-wise mean/std of `content` features to those of `style`."""
    cm, cs = mean_std(content)
    sm, ss = mean_std(style)
    return (content - cm) / cs * ss + sm


class AdaINNet(nn.Module):
    def __init__(self, vgg_path, decoder_path=None):
        super().__init__()
        vgg = _vgg()
        missing, _unexpected = vgg.load_state_dict(
            torch.load(vgg_path, map_location="cpu", weights_only=True), strict=False)
        assert not missing, f"VGG weights do not match the architecture: {missing}"
        ls = list(vgg.children())
        # relu1_1, relu2_1, relu3_1, relu4_1
        self.stages = nn.ModuleList([nn.Sequential(*ls[:4]), nn.Sequential(*ls[4:11]),
                                     nn.Sequential(*ls[11:18]), nn.Sequential(*ls[18:31])])
        for p in self.stages.parameters():
            p.requires_grad_(False)
        self.decoder = _decoder()
        if decoder_path is not None:
            self.decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=True))
        self.stages.eval()

    def train(self, mode=True):
        super().train(mode)
        self.stages.eval()          # encoder is always frozen
        return self

    def encode_all(self, x):
        outs = []
        for s in self.stages:
            x = s(x)
            outs.append(x)
        return outs

    def encode(self, x):
        for s in self.stages:
            x = s(x)
        return x

    def decode(self, t):
        return self.decoder(t)

    @torch.no_grad()
    def stylize_image(self, content, style_feat, alpha=1.0):
        fc = self.encode(content)
        t = adain(fc, style_feat)
        t = alpha * t + (1 - alpha) * fc
        return self.decode(t)


def pad_to_multiple(x, m=8):
    """Reflect-pad H and W up to a multiple of m. Returns padded tensor and (h, w) to crop back."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (h, w)


def crop_to(x, hw):
    return x[..., :hw[0], :hw[1]]
