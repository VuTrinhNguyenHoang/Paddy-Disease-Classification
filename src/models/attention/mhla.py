import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadLinearAttention2D(nn.Module):
    def __init__(self, c_in, c_out, heads):
        super().__init__()
        assert c_out % heads == 0

        self.heads = heads
        self.dh = c_out // heads
        self.scale = 1.0 / math.sqrt(self.dh)

        self.q_proj = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.k_proj = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.v_proj = nn.Conv2d(c_in, c_out, 1, bias=False)

    def kernel_feature(self, x):
        return F.elu(x) + 1
    
    def forward(self, x):
        B, _, H, W = x.shape
        M = H * W

        q = self.q_proj(x).flatten(2).view(B, self.heads, self.dh, M)
        k = self.k_proj(x).flatten(2).view(B, self.heads, self.dh, M)
        v = self.v_proj(x).flatten(2).view(B, self.heads, self.dh, M)

        q = self.kernel_feature(q) * self.scale
        k = self.kernel_feature(k) * self.scale

        KV = torch.einsum("bhcm,bhdm->bhcd", k, v)

        out = torch.einsum("bhcd,bhcm->bhdm", KV, q)

        k_sum = k.sum(dim=-1)
        denom = torch.einsum("bhcm,bhc->bhm", q, k_sum)
        denom = denom.clamp_min(1e-6).unsqueeze(2)

        out = out / denom
        out = out.reshape(B, -1, M).view(B, -1, H, W)

        return out