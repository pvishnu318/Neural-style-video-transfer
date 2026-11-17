"""Data access: DAVIS frames, style images, splits."""
import random
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DAVIS = ROOT / "data" / "DAVIS" / "JPEGImages" / "480p"
STYLES = ROOT / "data" / "styles"
SPLITS = ROOT / "splits"


def read_split(name):
    return [x.strip() for x in (SPLITS / f"{name}.txt").read_text().splitlines() if x.strip()]


def _read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_tensor(rgb_uint8):
    return torch.from_numpy(np.ascontiguousarray(rgb_uint8)).permute(2, 0, 1).float().div(255.0)


def seq_frame_paths(seq):
    return sorted((DAVIS / seq).glob("*.jpg"))


def load_sequence(seq, max_frames=0):
    """All frames of a DAVIS sequence as a [T,3,H,W] float tensor in [0,1] (CPU)."""
    paths = seq_frame_paths(seq)
    if max_frames:
        paths = paths[:max_frames]
    return torch.stack([to_tensor(_read_rgb(p)) for p in paths])


def style_paths(split):
    d = STYLES / split
    return sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))


def load_style(path, short_side=None):
    """Style image as [1,3,H,W] in [0,1]. If short_side is given the image is resized to that short side."""
    rgb = _read_rgb(path)
    if short_side:
        h, w = rgb.shape[:2]
        s = short_side / min(h, w)
        rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return to_tensor(rgb)[None]


class ClipDataset(torch.utils.data.Dataset):
    """Random T-frame clips from DAVIS with ONE shared random crop per clip (no resizing, so pixel motion
    statistics are the same as at inference time) and one random style from the given style split."""

    def __init__(self, sequences, style_split="train", clip_len=3, crop=256, length=100000, style_side=256,
                 max_gap=1):
        self.clip_len, self.crop, self.length, self.max_gap = clip_len, crop, length, max_gap
        self.paths = {s: seq_frame_paths(s) for s in sequences}
        self.seqs = [s for s in sequences if len(self.paths[s]) >= clip_len * max_gap + 1]
        self.styles = style_paths(style_split)
        self.style_side = style_side
        assert self.seqs and self.styles

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = random.Random()                      # fresh randomness per sample (also per worker)
        seq = rng.choice(self.seqs)
        paths = self.paths[seq]
        gap = rng.randint(1, self.max_gap)
        start = rng.randint(0, len(paths) - 1 - gap * (self.clip_len - 1))
        frames = [_read_rgb(paths[start + gap * k]) for k in range(self.clip_len)]
        if rng.random() < 0.5:                     # horizontal flip, same for all frames
            frames = [f[:, ::-1] for f in frames]
        h, w = frames[0].shape[:2]
        y, x = rng.randint(0, h - self.crop), rng.randint(0, w - self.crop)
        clip = torch.stack([to_tensor(f[y:y + self.crop, x:x + self.crop]) for f in frames])
        style = load_style(rng.choice(self.styles), self.style_side)[0]
        # fixed-size style tensor so batches stack: random crop / resize to style_side x style_side
        sh, sw = style.shape[-2:]
        if min(sh, sw) < self.style_side:
            style = torch.nn.functional.interpolate(style[None], size=(self.style_side, self.style_side),
                                                    mode="bilinear", align_corners=False)[0]
        else:
            sy, sx = rng.randint(0, sh - self.style_side), rng.randint(0, sw - self.style_side)
            style = style[:, sy:sy + self.style_side, sx:sx + self.style_side]
        return clip, style
