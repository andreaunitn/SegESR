"""
Tests for the frequency bands of the BiBandSR losses (bibandsr/bands.py).

Run with: pytest tests/test_bibandsr_bands.py
"""

import numpy as np
import torch

from bibandsr.bands import low_band, low_mse, low_pass_mask, low_share, project_high
from data_tools.band_scale import high_pass

def test_mask_matches_the_analysis_mask():
    mask = low_pass_mask(48, 64, 0.125).numpy()
    np.testing.assert_allclose(mask, 1.0 - high_pass((48, 64), 0.125), atol=1e-6)

def test_low_band_separates_a_slow_and_a_fast_cosine():
    n = 64
    yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    slow = torch.cos(2 * torch.pi * 4 * xx / n)      # 1/16 cycle per pixel
    fast = torch.cos(2 * torch.pi * 16 * yy / n)     # 1/4 cycle per pixel
    out = low_band((slow + fast)[None, None], 0.125)[0, 0]
    torch.testing.assert_close(out, slow, atol=1e-5, rtol=0)
    assert abs(low_share(fast[None], 0.125)) < 1e-10
    torch.testing.assert_close(low_mse(slow + fast, fast, 0.125), slow.pow(2).mean(), atol=1e-6, rtol=0)

def test_project_high_keeps_the_value_and_removes_the_low_band_of_the_gradient():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    w = torch.randn(2, 3, 32, 32)
    y = project_high(x, 0.125)
    torch.testing.assert_close(y, x, atol=1e-5, rtol=0)
    (y * w).sum().backward()
    # The gradient is w without its low band
    torch.testing.assert_close(x.grad, w - low_band(w, 0.125), atol=1e-5, rtol=0)
    # Below the transition (mask exactly 1) the gradient has no energy left
    spectrum = torch.fft.fft2(x.grad).abs()
    assert spectrum[..., low_pass_mask(32, 32, 0.125) == 1].max() < 1e-4
