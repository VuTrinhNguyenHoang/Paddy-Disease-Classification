import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..pe import PositionalEncoding2D, LearnablePositionalEncoding2D

class MultiHeadLinearAttention2D(nn.Module):
    def __init__(self, c_in, c_out, heads, pe_learnable=False):
        super().__init__()
        assert c_out % heads == 0

        self.heads = heads
        self.dh = c_out // heads
        self.scale = 1.0 / math.sqrt(self.dh)

        self.q_proj = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.k_proj = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.v_proj = nn.Conv2d(c_in, c_out, 1, bias=False)

        self.pe = PositionalEncoding2D(c_out) if not pe_learnable else LearnablePositionalEncoding2D(c_out)

    def kernel_feature(self, x):
        return F.elu(x) + 1
    
    def forward(self, x):
        B, _, H, W = x.shape
        M = H * W

        # Projections
        q = self.q_proj(x).flatten(2).view(B, self.heads, self.dh, M)
        k = self.k_proj(x).flatten(2).view(B, self.heads, self.dh, M)
        v = self.v_proj(x).flatten(2).view(B, self.heads, self.dh, M)

        # Positional encoding (B, C, M)
        r = self.pe(B, H, W).view(B, self.heads, self.dh, M)
        q = q + r
        k = k + r

        # Kernel feature (ensures positivity) + scaling
        q = self.kernel_feature(q) * self.scale
        k = self.kernel_feature(k) * self.scale

        # Precompute KV (B, h, dh, dh)
        KV = torch.einsum("bhcm,bhdm->bhcd", k, v)

        # Output = (KV) * q  -> (B, h, dh, M)
        out = torch.einsum("bhcd,bhcm->bhdm", KV, q)

        # Normalization denominator (stability with clamp)
        k_sum = k.sum(dim=-1)  # (B, h, dh)
        denom = torch.einsum("bhcm,bhc->bhm", q, k_sum)  # (B, h, M)
        denom = denom.clamp_min(1e-6).unsqueeze(2)  # (B, h, 1, M)

        out = out / denom
        out = out.reshape(B, -1, M).view(B, -1, H, W)

        # if torch.isnan(out).any():
        #     raise RuntimeError("NaN detected in MultiHeadLinearAttention2D output")
        return out