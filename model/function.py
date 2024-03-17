import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
T = 1000
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