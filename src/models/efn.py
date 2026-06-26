"""Energy Flow Network / Deep Sets: permutation-invariant set regression.

Each particle's feature vector is mapped by a shared per-particle MLP (Phi),
the latent vectors are masked-summed over the event, and an event MLP (F) maps
the pooled latent to the targets. Permutation invariant by construction.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(sizes, act=nn.GELU, last_act=True):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


class EnergyFlowNetwork(nn.Module):
    def __init__(self, n_feat: int, n_out: int, phi=(128, 128, 128),
                 f=(128, 128), latent=128, pool: str = "sum"):
        super().__init__()
        self.phi = _mlp((n_feat, *phi, latent))
        self.f = _mlp((latent, *f), last_act=True)
        self.head = nn.Linear(f[-1], n_out)
        self.pool = pool

    def forward(self, x, mask):
        # x: [B,P,F]  mask: [B,P]
        h = self.phi(x) * mask.unsqueeze(-1)        # [B,P,L]
        summed = h.sum(dim=1)                        # [B,L]
        if self.pool == "mean":
            n = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            summed = summed / n
        return self.head(self.f(summed))            # [B,n_out]
