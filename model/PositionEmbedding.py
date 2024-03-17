import torch
import torch.nn as nn
import math

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, T=1000):
        super(PositionEmbedding, self).__init__()
        pe = torch.zeros(T, d_model)
        position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos = torch.arange(T).float()
        emb = pos[:, None] * div_term[None, :]
        pe = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        if d_model % 2 != 0:
          pe = torch.cat([pe, torch.sin(position * div_term[-1]).unsqueeze(-1)], dim=-1)
        pe = pe.view(T, d_model)

        self.timeembedding = nn.Sequential(
            nn.Embedding.from_pretrained(pe),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model * 4),
        )

    def forward(self, x):
        emb = self.timeembedding(x)
        return emb