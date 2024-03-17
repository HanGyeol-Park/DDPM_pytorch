import torch.nn as nn
from model.MultiHeadAttention import MultiHeadAttentionBlock

class ResnetBlock(nn.Module):
  def __init__(self, in_ch, hidden_dim, attention = False):
    super(ResnetBlock, self).__init__()
    self.attention = attention
    self.block1 = nn.Sequential(
        nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1),
        nn.SiLU()
    )
    self.mhattention = nn.Sequential(
        MultiHeadAttentionBlock(hidden_dim, 8),
    )
    self.block2 = nn.Sequential(
        nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1),
        nn.BatchNorm2d(hidden_dim),
        nn.Dropout(p = 0.1)
    )
    self.temb_proj = nn.Sequential(
        nn.Linear(in_ch, hidden_dim),
        nn.SiLU(),
    )

  def forward(self, x, temb):
    y = self.block1(x)
    y = y + self.temb_proj(temb)[:,:,None,None]
    if self.attention == True:
      y = self.mhattention(y)
    z = self.block2(y)
    x = x + z
    return x