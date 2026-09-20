import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class SamPerceptualLoss(nn.Module):
    """
    Computes perceptual feature MSE loss using the SAM 2 image encoder.
    Expects input tensors in the range [0.0, 1.0].
    """

    def __init__(self, image_encoder, normalization=None):
        super().__init__()

        self.image_encoder = image_encoder
        self.image_encoder.eval()
        self.image_encoder.requires_grad_(False)

        if normalization is not None:
            self.normalization = normalization
        else:
            self.normalization = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

    def forward(self, sr_rgb, gt_rgb):
        """
        Args:
            sr_rgb (torch.Tensor): Super-resolved image (B, 3, H, W) in [0, 1] with gradients active.
            gt_rgb (torch.Tensor): Ground-truth image (B, 3, H, W) in [0, 1].
        Returns:
            torch.Tensor: Scalar MSE feature loss.
        """

        sr_input = self.normalization(sr_rgb)
        sr_embeds = self.image_encoder(sr_input)["vision_features"]

        with torch.no_grad():
            gt_input = self.normalization(gt_rgb)
            gt_embeds = self.image_encoder(gt_input)["vision_features"]

        return F.mse_loss(sr_embeds.float(), gt_embeds.float(), reduction="mean")