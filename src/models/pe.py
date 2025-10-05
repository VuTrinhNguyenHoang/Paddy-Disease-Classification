import torch, math
import torch.nn as nn

class PositionalEncoding2D(nn.Module):
    def __init__(self, dim, max_h=256, max_w=256, temperature=10000):
        super().__init__()
        assert dim % 4 == 0
        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w
        self.temperature = temperature

        pe = self._build_pe()
        self.register_buffer("pe", pe, persistent=False)

    def _build_pe(self):
        dim_quarter = self.dim // 4

        y_pos = torch.arange(self.max_h, dtype=torch.float32).unsqueeze(1)  # (H,1)
        x_pos = torch.arange(self.max_w, dtype=torch.float32).unsqueeze(1)  # (W,1)

        div_term = torch.exp(
            torch.arange(dim_quarter, dtype=torch.float32) * -(math.log(self.temperature) / dim_quarter)
        )  # (dim_q,)

        # (H, dim_q)
        y_scaled = y_pos * div_term.unsqueeze(0)  # broadcast
        # (W, dim_q)
        x_scaled = x_pos * div_term.unsqueeze(0)

        sin_y = torch.sin(y_scaled).unsqueeze(2).repeat(1, 1, self.max_w)  # (H, dim_q, W)
        cos_y = torch.cos(y_scaled).unsqueeze(2).repeat(1, 1, self.max_w)
        # (H, W, dim_q)
        sin_x = torch.sin(x_scaled).unsqueeze(0).repeat(self.max_h, 1, 1)
        cos_x = torch.cos(x_scaled).unsqueeze(0).repeat(self.max_h, 1, 1)

        pe = torch.zeros(self.dim, self.max_h, self.max_w, dtype=torch.float32)
        # Rearrange to (C, H, W)
        pe[0:dim_quarter] = sin_y.permute(1, 0, 2)          # (dim_q, H, W)
        pe[dim_quarter:2*dim_quarter] = cos_y.permute(1, 0, 2)
        # For x we need (dim_q, H, W); current sin_x/cos_x is (H, W, dim_q)
        pe[2*dim_quarter:3*dim_quarter] = sin_x.permute(2, 0, 1)
        pe[3*dim_quarter:] = cos_x.permute(2, 0, 1)
        return pe

    def forward(self, B, H, W):
        pe = self.pe[:, :H, :W].reshape(self.dim, H * W).unsqueeze(0).repeat(B, 1, 1)
        return pe
    
class LearnablePositionalEncoding2D(nn.Module):
    def __init__(self, dim, max_h=256, max_w=256):
        super().__init__()
        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w
        
        # Learnable parameters
        self.pe = nn.Parameter(torch.randn(dim, max_h, max_w) * 0.02)
    
    def forward(self, B, H, W):
        pe = self.pe[:, :H, :W]
        pe = pe.reshape(self.dim, H * W)
        pe = pe.unsqueeze(0).repeat(B, 1, 1)
        return pe
