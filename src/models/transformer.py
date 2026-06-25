"""Particle-cloud Transformer: self-attention over the soft-particle set.

A lightweight, permutation-invariant Transformer encoder (no positional
encoding) with attention masking over padded particles, mean-pooled over real
particles, then an MLP head to the targets. The heavier analogue of the EFN.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ParticleTransformer(nn.Module):
    def __init__(self, n_feat: int, n_out: int, d_model: int = 128, nhead: int = 8,
                 layers: int = 4, dim_ff: int = 256, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_feat, d_model), nn.GELU(),
                                   nn.Linear(d_model, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model),
                                  nn.GELU(), nn.Linear(d_model, n_out))

    def forward(self, x, mask):
        # x: [B,P,F]  mask: [B,P] (1=real, 0=pad)
        h = self.embed(x)
        pad = mask < 0.5  # True where padded -> ignored by attention
        h = self.encoder(h, src_key_padding_mask=pad)
        h = h * mask.unsqueeze(-1)
        n = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = h.sum(dim=1) / n
        return self.head(pooled)
