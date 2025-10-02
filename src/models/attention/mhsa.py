import torch
import torch.nn as nn
import math

from ..pe import PositionalEncoding2D, LearnablePositionalEncoding2D

class MultiHeadSelfAttention2D(nn.Module):
    def __init__(self, c_in, c_out, heads, pe_learnable=False):
        super().__init__()
        assert c_out % heads == 0

        self.heads = heads
        self.dh = c_out // heads
        self.scale = math.sqrt(self.dh)

        self.v_proj = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.z_proj = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.q_proj = nn.Conv2d(c_in, c_out, 1, bias=False)

        self.pe = PositionalEncoding2D(c_out) if pe_learnable == False else LearnablePositionalEncoding2D(c_out)

    def forward(self, x):
        B, _, H, W = x.shape
        M = H * W

        v = self.v_proj(x).flatten(2)
        z = self.z_proj(x).flatten(2)
        q = self.q_proj(x).flatten(2)
        r = self.pe(B, H, W)

        def split_heads(t):
            return t.view(B.self.heads, self.dh, M)
        
        vh, zh, qh, rh = map(split_heads, (v, z, q, r))

        A_vz = torch.einsum("bhcm,bhcn->bhmn", vh, zh)
        A_vr = torch.einsum("bhcm,bhcn->bhmn", vh, rh)

        A = torch.softmax((A_vz + A_vr) / self.scale, dim=-1)
        out_h = torch.einsum("bhmn,bhdn->bhdm", A, qh)

        out = out_h.reshape(B, -1, M).view(B, -1, H, W)
        return out