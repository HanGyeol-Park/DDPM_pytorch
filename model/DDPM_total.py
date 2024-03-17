import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import math

T = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = (1. - betas).to(device)
alphas_bar = torch.cumprod(alphas, dim=0).to(device)
alphas_bar_prev = F.pad(alphas_bar[:-1], (1, 0), value=1.).to(device)

reciprocal_alphas_sqrt1 = torch.sqrt(1. / alphas_bar).to(device)
reciprocal_alphas_sqrt2 = torch.sqrt(1. / alphas_bar - 1.).to(device)
posterior_mean_coeff1 = torch.sqrt(alphas_bar_prev) * betas / (1. - alphas_bar).to(device)
posterior_mean_coeff2 = torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar).to(device)

ddpm_coeff1 = torch.sqrt(1. / alphas).to(device)
ddpm_coeff2 = (betas / torch.sqrt(alphas * (1. - alphas_bar))).to(device)

posterior_variance = betas * ((1. - alphas_bar_prev) / (1. - alphas_bar)).to(device)

def gather_and_expand(coeff, t, xshape):
    B, *dims = xshape
    coeff_t = torch.gather(coeff, index=t, dim=0)
    return coeff_t.view([B] + [1]*len(dims))

def train(model, x_0):
    t = torch.randint(T, size=(x_0.shape[0], ), device=x_0.device)
    eps = torch.randn_like(x_0)

    x_t = gather_and_expand(torch.sqrt(alphas_bar).to(device), t, x_0.shape) * x_0 + \
          gather_and_expand(torch.sqrt(1. - alphas_bar).to(device), t, x_0.shape) * eps

    print(x_t.shape)
    loss = F.mse_loss(model(x_t, t), eps)
    return loss

def sample(model, x_T):
    x_t = x_T
    for time_step in reversed(range(T)):
        t = torch.full((x_T.shape[0], ), time_step, dtype=torch.long, device=device)
        eps = model(x_t, t)
        x0_predicted = gather_and_expand(reciprocal_alphas_sqrt1, t, eps.shape) * x_t - \
            gather_and_expand(reciprocal_alphas_sqrt2, t, eps.shape) * eps

        mean = gather_and_expand(posterior_mean_coeff1, t, eps.shape) * x0_predicted + \
            gather_and_expand(posterior_mean_coeff2, t, eps.shape) * x_t

        #DDPM paper's mean
        """mean = gather_and_expand(ddpm_coeff1, t, eps.shape) * x_t + \
            gather_and_expand(ddpm_coeff2, t, eps.shape) * eps"""

        z = torch.randn_like(x_t) if time_step else 0
        var = torch.sqrt(gather_and_expand(posterior_variance, t, eps.shape)) * z

        x_t = mean + var
    x_0 = x_t
    return x_0

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

class MultiHeadAttentionBlock(nn.Module):
  def __init__(self, dim, num_heads):
    super(MultiHeadAttentionBlock, self).__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.attenq = nn.Conv2d(dim, dim, 1)
    self.attenk = nn.Conv2d(dim, dim, 1)
    self.attenv = nn.Conv2d(dim, dim, 1)
    self.atteno = nn.Conv2d(dim, dim, 1)

  def forward(self, x):
    q = self.attenq(x)
    k = self.attenk(x)
    v = self.attenv(x)

    h = q.shape[2]
    w = q.shape[3]

    q = q.reshape(q.shape[0], self.num_heads, self.head_dim, q.shape[2] * q.shape[3])
    k = k.reshape(k.shape[0], self.num_heads, self.head_dim, k.shape[2] * k.shape[3]).transpose(2, 3)
    v = v.reshape(v.shape[0], self.num_heads, self.head_dim, v.shape[2] * v.shape[3])

    Scaled_dot_product = torch.matmul(q, k) / (self.head_dim ** 0.5)
    attention_weights = F.softmax(Scaled_dot_product, dim = -1)
    attention_output = torch.matmul(attention_weights, v)

    attention_output = attention_output.transpose(2, 3).reshape(attention_output.shape[0], self.dim, h, w)
    attention_output = self.atteno(attention_output)

    return attention_output

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

class ConV(nn.Module):
  def __init__(self, x, y):
    super(ConV, self).__init__()
    self.conv = nn.Conv2d(x, y, kernel_size = 1)
    self.convblock = nn.Sequential(
        nn.Conv2d(x, y, 3, padding = 1),
        nn.BatchNorm2d(x),
        nn.SiLU(),
        nn.Conv2d(y, y, 1),
        nn.SiLU(),
    )

  def forward(self, x, temb):
    return self.conv(x)

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