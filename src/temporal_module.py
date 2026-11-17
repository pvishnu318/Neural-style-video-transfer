"""B2: flow-free local cross-frame attention with a learned gate, applied in AdaIN feature space (1/8 resolution).

For frame t the AdaIN output feature  T_t = AdaIN(enc(I_t), enc(S))  is blended with information fetched from the
PREVIOUS stylised feature F_{t-1} (the module's own previous output, so it is recurrent):

    Q = q(enc(I_t))      K = k(enc(I_{t-1}))     (content features decide WHERE the previous frame matches)
    attention over a (2r+1)x(2r+1) window around every position (local, so no optical flow is needed at inference)
    A = out( sum_w softmax_w(Q.K_w / sqrt(d)) * v(F_{t-1,w}) )
    g = sigmoid( gate([proj(T_t), proj(A), peak similarity]) )          # per-position confidence, init ~0.05
    F_t = T_t + g * (A - T_t)

RAFT optical flow is used only in the TRAINING loss (to supervise temporal consistency), never at inference.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalAttnFusion(nn.Module):
    def __init__(self, ch=512, d=64, dv=128, r=3, gate_bias=-3.0):
        super().__init__()
        self.r, self.d = r, d
        self.q = nn.Conv2d(ch, d, 1)
        self.k = nn.Conv2d(ch, d, 1)
        self.v = nn.Conv2d(ch, dv, 1)
        self.out = nn.Conv2d(dv, ch, 1)
        self.gt = nn.Conv2d(ch, 32, 1)
        self.ga = nn.Conv2d(ch, 32, 1)
        self.gate = nn.Conv2d(65, 1, 1)
        nn.init.constant_(self.gate.bias, gate_bias)

    def forward(self, t_cur, fc_cur, fc_prev, f_prev):
        """t_cur: AdaIN feature of frame t; fc_cur/fc_prev: content encoder features of t / t-1; f_prev: module output at t-1."""
        b, c, h, w = t_cur.shape
        k = 2 * self.r + 1
        q = self.q(fc_cur).reshape(b, self.d, 1, h * w)
        kn = F.unfold(self.k(fc_prev), k, padding=self.r).reshape(b, self.d, k * k, h * w)
        sim = (q * kn).sum(1) / math.sqrt(self.d)                      # [b, k*k, hw]
        att = sim.softmax(1)
        vn = F.unfold(self.v(f_prev), k, padding=self.r).reshape(b, -1, k * k, h * w)
        a = (att.unsqueeze(1) * vn).sum(2).reshape(b, -1, h, w)
        a = self.out(a)
        peak = sim.max(1).values.reshape(b, 1, h, w)
        g = torch.sigmoid(self.gate(torch.cat([self.gt(t_cur), self.ga(a), peak], 1)))
        return t_cur + g * (a - t_cur), g
