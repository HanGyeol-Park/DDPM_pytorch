import torch
import torch.nn as nn
import torch.nn.functional as F
from model.PositionEmbedding import PositionEmbedding
from model.MultiHeadAttention import MultiHeadAttentionBlock
from model.ResNet import ResnetBlock
from model.DDPM_total import ConV
from model.Sampling import upSample, downSample

device = "cuda" if torch.cuda.is_available() else "cpu"
# Unavailable code is for channel fixed ddpm
class UNet(nn.Module):
  def __init__(self, hidden_dim, num_sampling, ch, pos):
    super(UNet, self).__init__()
    self.hidden_dim = hidden_dim
    self.block0 = nn.Sequential(
        nn.Conv2d(ch, hidden_dim, 1),
        nn.SiLU(),
    )
    self.block1 = nn.ModuleList()
    self.block2 = nn.ModuleList()
    self.block3 = nn.ModuleList()
    self.block4 = nn.Sequential(
        nn.SiLU(),
        nn.Conv2d(hidden_dim, ch, 1),
    )
    self.timeembedding = PositionEmbedding(pos, 1000).to(device)

    att = True
    for i in range(num_sampling):
      att = not att
      self.block1.append(ResnetBlock(pos * 4, hidden_dim * (2 ** i), att).to(device))
      self.block1.append(ResnetBlock(pos * 4, hidden_dim * (2 ** i), att).to(device))
      self.block1.append(downSample(hidden_dim * (2 ** i)).to(device))
      """self.block1.append(ResnetBlock(pos * 4, hidden_dim, att).to(device))
      self.block1.append(ResnetBlock(pos * 4, hidden_dim, att).to(device))
      self.block1.append(downSample(hidden_dim).to(device))"""

    for i in range(1):
      self.block2.append(ResnetBlock(pos * 4, hidden_dim * (2 ** num_sampling)).to(device))
      """self.block2.append(ResnetBlock(pos * 4, hidden_dim).to(device))"""

    for i in range(num_sampling):
      att = not att
      self.block3.append(ResnetBlock(pos * 4, hidden_dim * (2 ** (num_sampling - i)), att).to(device))
      self.block3.append(upSample(hidden_dim * (2 ** (num_sampling - i))).to(device))
      self.block3.append(ConV(hidden_dim * (2 ** (num_sampling - i)), hidden_dim * (2 ** (num_sampling - i - 1))).to(device))
      """self.block3.append(ResnetBlock(pos * 4, hidden_dim, att).to(device))
      self.block3.append(upSample(hidden_dim).to(device))
      self.block3.append(ConV(hidden_dim  * 2, hidden_dim).to(device))"""

  def forward(self, x, t):
    temb = self.timeembedding(t)
    xs = []

    x = self.block0(x)

    for block in self.block1:
      if isinstance(block, downSample):
        xs.append(x)
      x = block(x, temb)

    for block in self.block2:
      x = block(x, temb)

    for block in self.block3:
      x = block(x, temb)
      if isinstance(block, upSample):
        x = torch.cat([x, xs.pop()], dim=1)

    x = self.block4(x)

    return x