"""
RRDBNet (ESRGAN generator), as in BasicSR (basicsr/archs/rrdbnet_arch.py, Apache-2.0), for the regression SR
baselines of data_tools/regression_sr.py. Copied here to avoid depending on basicsr.

`load_rrdbnet` also reads the older ESRGAN / BSRGAN key names (RRDB_trunk, trunk_conv, upconv1, HRconv...).
"""

import re

import torch
from torch import nn
from torch.nn import functional as F

class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb3(self.rdb2(self.rdb1(x)))
        return out * 0.2 + x

class RRDBNet(nn.Module):
    """x4 RRDBNet: 23 RRDB blocks, then two nearest-neighbour x2 upsamplings."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))

# Old ESRGAN / BSRGAN names -> BasicSR names
_RENAMES = [
    (r"^RRDB_trunk\.", "body."),
    (r"\.RDB(\d)\.", r".rdb\1."),
    (r"^trunk_conv\.", "conv_body."),
    (r"^upconv1\.", "conv_up1."),
    (r"^upconv2\.", "conv_up2."),
    (r"^HRconv\.", "conv_hr."),
]

def convert_state_dict(state_dict):
    """BasicSR state dict from a checkpoint: unwraps 'params_ema' / 'params' and renames the old ESRGAN keys."""

    for key in ("params_ema", "params"):
        if key in state_dict:
            state_dict = state_dict[key]
            break
    converted = {}
    for name, value in state_dict.items():
        for pattern, replacement in _RENAMES:
            name = re.sub(pattern, replacement, name)
        converted[name] = value
    return converted

def load_rrdbnet(path, device="cpu"):
    model = RRDBNet()
    model.load_state_dict(convert_state_dict(torch.load(path, map_location="cpu")), strict=True)
    return model.eval().to(device)
