import torch
import torch.nn as nn
import torch.nn.functional as F

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