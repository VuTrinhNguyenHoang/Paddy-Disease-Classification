import torch, math
import torch.nn as nn

class PositionalEncoding2D(nn.Module):
    def __init__(self, dim, max_h=256, max_w=256, temperature=10000):
        super().__init__()
        assert dim % 4 == 0
        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w
 
        pe = self._generate_pe(dim, max_h, max_w, temperature)
        self.register_buffer("pe", pe, persistent=False)

    def _generate_pe(self, dim, max_h, max_w, temperature):
        pe = torch.zeros(dim, max_h, max_w)

        d_model = dim // 2

        y_pos = torch.range(max_h, dtype=torch.float32).unsqueeze(1)
        x_pos = torch.range(max_w, dtype=torch.float32).unsqueeze(0)

        div_term = torch.exp(torch.range(0, d_model, 2, dtype=torch.float32) * 
                             -(math.log(temperature) / d_model))
        
        pe[0::4] = torch.sin(y_pos * div_term.view(-1, 1, 1))[:d_model//2]
        pe[1::4] = torch.cos(y_pos * div_term.view(-1, 1, 1))[:d_model//2]
        
        pe[2::4] = torch.sin(x_pos * div_term.view(-1, 1, 1))[:d_model//2]
        pe[3::4] = torch.cos(x_pos * div_term.view(-1, 1, 1))[:d_model//2]
        
        return pe

    def forward(self, B, H, W): 
        pe = self.pe[:, :H, :W]
        pe = pe.reshape(self.dim, H * W)
        pe = pe.unsqueeze(0).repeat(B, 1, 1)
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
