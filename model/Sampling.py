import torch.nn as nn

"""
# Upsampling and Downsampling that channles fixed.
class upSample(nn.Module):
  def __init__(self, dim):
    super(upSample, self).__init__()
    self.dim = dim
    self.upsample = nn.Sequential(
      nn.Upsample(scale_factor = 2, mode = "nearest"),
      nn.Conv2d(dim, dim, 3, padding = 1)
  )

  def forward(self, x, temb):
    return self.upsample(x)

class downSample(nn.Module):
  def __init__(self, dim):
    super(downSample, self).__init__()
    self.dim = dim
    self.downsample = nn.Sequential(
      Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1 = 2, p2 = 2),
      nn.Conv2d(dim * 4, dim, 1)
  )

  def forward(self, x, temb):
    return self.downsample(x)

"""

class upSample(nn.Module):
  def __init__(self, dim):
    super(upSample, self).__init__()
    self.dim = dim
    self.upsample = nn.Sequential(
        nn.ConvTranspose2d(dim, dim // 2, kernel_size = 4, stride = 2, dilation = 1, padding = 1),
        nn.SiLU(),
    )

  def forward(self, x, temb):
    return self.upsample(x)


class downSample(nn.Module):
  def __init__(self, dim):
    super(downSample, self).__init__()
    self.dim = dim
    self.downsample = nn.Sequential(
        nn.Conv2d(dim, dim * 2, kernel_size = 3, stride = 2, padding = 1),
        nn.SiLU(),
        nn.Conv2d(dim * 2, dim * 2, 1),
        nn.SiLU(),
    )

  def forward(self, x, temb):
    return self.downsample(x)