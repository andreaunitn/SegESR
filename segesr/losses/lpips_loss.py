import torch
import torch.nn as nn
import pyiqa

class LPIPSLoss(nn.Module):
    """
    Computes LPIPS loss between predicted and ground-truth images.
    Expects input tensors in the range [0.0, 1.0].
    """

    def __init__(self, device):
        super().__init__()
        self.metric = pyiqa.create_metric("lpips", device=device, as_loss=True)

    def forward(self, sr_rgb, gt_rgb):
        """
        Args:
            sr_rgb (torch.Tensor): Super-resolved image (B, 3, H, W) in [0, 1] with gradients active.
            gt_rgb (torch.Tensor): Ground-truth image (B, 3, H, W) in [0, 1].
        Returns:
            torch.Tensor: Scalar LPIPS loss.
        """

        return self.metric(sr_rgb.to(torch.float32), gt_rgb.to(torch.float32))