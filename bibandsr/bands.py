"""
Frequency bands of image tensors, for losses split at a cutoff (in cycles per pixel; 1/8 is the LR Nyquist limit of
x4 SR). The mask is the one of data_tools/band_scale.py: a raised cosine from cutoff * (1 - width) to
cutoff * (1 + width), so that the losses and the analysis use the same bands.
"""

import math

import torch

def low_pass_mask(height, width_px, cutoff, width=0.1, device=None):
    """Smooth low-pass mask over torch.fft.fft2 coefficients: 1 below cutoff * (1 - width), 0 above cutoff * (1 + width)."""
    fy = torch.fft.fftfreq(height, device=device)[:, None]
    fx = torch.fft.fftfreq(width_px, device=device)[None, :]
    f = torch.sqrt(fy ** 2 + fx ** 2)
    lo, hi = cutoff * (1 - width), cutoff * (1 + width)
    t = ((f - lo) / (hi - lo)).clamp(0.0, 1.0)
    return 0.5 + 0.5 * torch.cos(math.pi * t)

def low_band(x, cutoff, width=0.1):
    """The band of x (..., H, W) below the cutoff, computed in float32 (periodic boundaries)."""
    mask = low_pass_mask(x.shape[-2], x.shape[-1], cutoff, width, device=x.device)
    return torch.fft.ifft2(torch.fft.fft2(x.float()) * mask).real.to(x.dtype)

def project_high(x, cutoff, width=0.1):
    """Equal to x, but a gradient through it reaches x only in the band above the cutoff (stop-gradient on the low band)."""
    low = low_band(x, cutoff, width)
    return low.detach() + (x - low)

def low_mse(x, y, cutoff, width=0.1):
    """Mean squared error of x - y restricted to the band below the cutoff."""
    return low_band(x.float() - y.float(), cutoff, width).pow(2).mean()

def low_share(x, cutoff, width=0.1):
    """Share of the energy of x (any tensor of images) in the band below the cutoff."""
    x = x.float()
    return (low_band(x, cutoff, width).pow(2).sum() / x.pow(2).sum().clamp_min(1e-20)).item()
