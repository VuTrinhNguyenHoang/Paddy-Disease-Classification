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

        self.q_proj = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.k_proj = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.v_proj = nn.Conv2d(c_in, c_out, 1, bias=False)

        self.pe = PositionalEncoding2D(c_out) if not pe_learnable else LearnablePositionalEncoding2D(c_out)

    def forward(self, x):
        B, _, H, W = x.shape
        M = H * W

        q = self.q_proj(x).flatten(2)  # (B, C, M)
        k = self.k_proj(x).flatten(2)
        v = self.v_proj(x).flatten(2)

        # Positional encoding (B, C, M)
        r = self.pe(B, H, W)

        # Inject positional encoding (can experiment: only add to q or k)
        q = q + r
        k = k + r

        def split_heads(t):
            return t.view(B, self.heads, self.dh, M)  # (B, h, dh, M)

        qh, kh, vh = map(split_heads, (q, k, v))

        logits = torch.einsum("bhcm,bhcn->bhmn", qh, kh) / self.scale
        A = torch.softmax(logits.float(), dim=-1).type_as(qh)

        out_h = torch.einsum("bhmn,bhdn->bhdm", A, vh)

        out = out_h.reshape(B, -1, M).view(B, -1, H, W)

        # Optional runtime check (can disable for speed)
        # if torch.isnan(out).any():
        #     raise RuntimeError("NaN detected in MultiHeadSelfAttention2D output")
        return out