"""Optical flow (RAFT-Large), backward warping and occlusion masks.

Conventions
-----------
* images: [B,3,H,W] float in [0,1]
* flow(a -> b) = RAFT(a, b): displacement field defined on a's pixel grid with
  a(x) ~= b(x + flow(x)). Units: pixels.
* To warp the PREVIOUS stylized frame onto the CURRENT frame we need, for every pixel x of the
  current frame, where it was in the previous frame:  flow_bw = flow(cur -> prev), and then
  warped_prev(x) = prev(x + flow_bw(x)).
* The flow is ALWAYS estimated on the ORIGINAL (unstylized) frames, never on stylized frames.
"""
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


class Flow:
    def __init__(self, device="cuda", iters=12):
        self.device, self.iters = device, iters
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, a, b):
        """flow(a -> b), same spatial size as the inputs."""
        h, w = a.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        if ph or pw:
            a = F.pad(a, (0, pw, 0, ph), mode="replicate")
            b = F.pad(b, (0, pw, 0, ph), mode="replicate")
        flow = self.model(a * 2 - 1, b * 2 - 1, num_flow_updates=self.iters)[-1]
        return flow[..., :h, :w]


def warp(img, flow):
    """Backward warp: out(x) = img(x + flow(x)). Returns (warped, in_frame_mask[B,1,H,W])."""
    b, _, h, w = img.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=img.device, dtype=img.dtype),
                            torch.arange(w, device=img.device, dtype=img.dtype), indexing="ij")
    gx = xx[None] + flow[:, 0]
    gy = yy[None] + flow[:, 1]
    inside = ((gx >= 0) & (gx <= w - 1) & (gy >= 0) & (gy <= h - 1)).unsqueeze(1).to(img.dtype)
    grid = torch.stack((2 * gx / max(w - 1, 1) - 1, 2 * gy / max(h - 1, 1) - 1), dim=-1)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out, inside


def _grad_sq(f):
    """Squared spatial gradient magnitude of a 2-channel field (|du/dx|^2+|du/dy|^2+|dv/dx|^2+|dv/dy|^2)."""
    dx = f[..., :, 1:] - f[..., :, :-1]
    dy = f[..., 1:, :] - f[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return (dx ** 2).sum(1, keepdim=True) + (dy ** 2).sum(1, keepdim=True)


def occlusion_mask(flow_bw, flow_fw):
    """Valid-pixel mask in the CURRENT frame (1 = reliable, 0 = occluded / unreliable).

    flow_bw = flow(cur -> prev) on the current grid;  flow_fw = flow(prev -> cur) on the previous grid.
    Forward-backward consistency and motion-boundary tests of Ruder et al. (2016), as used by
    Lai et al. (2018) for the warping-error metric. Also excludes pixels that leave the frame.
    """
    fw_on_cur, inside = warp(flow_fw, flow_bw)               # flow_fw sampled where cur pixels were in prev
    mag2 = (flow_bw ** 2).sum(1, keepdim=True) + (fw_on_cur ** 2).sum(1, keepdim=True)
    inconsistent = ((flow_bw + fw_on_cur) ** 2).sum(1, keepdim=True) > 0.01 * mag2 + 0.5
    boundary = _grad_sq(flow_bw) > 0.01 * (flow_bw ** 2).sum(1, keepdim=True) + 0.002
    return (~inconsistent & ~boundary).to(flow_bw.dtype) * inside


@torch.no_grad()
def flow_and_mask(flow_net, cur, prev):
    """(flow_bw, valid_mask) for the pair, estimated on the ORIGINAL frames."""
    bw = flow_net(cur, prev)
    fw = flow_net(prev, cur)
    return bw, occlusion_mask(bw, fw)
