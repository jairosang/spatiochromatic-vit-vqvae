"""YCbCr, ITU-R BT.601 full range (the JPEG convention).

Both directions map [0, 1] -> [0, 1]: Y is luma, Cb/Cr are chroma shifted by +0.5.
Works on (C, H, W) and (B, C, H, W) tensors on any device.
"""
import torch

YCC_FROM_RGB = torch.tensor([
    [ 0.299,     0.587,     0.114   ],
    [-0.168736, -0.331264,  0.5     ],
    [ 0.5,      -0.418688, -0.081312],
], dtype=torch.float64)
RGB_FROM_YCC = torch.linalg.inv(YCC_FROM_RGB)
OFFSET = torch.tensor([0.0, 0.5, 0.5], dtype=torch.float64).view(3, 1, 1)


def _apply(matrix: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Multiply every pixel's 3-vector by `matrix`."""
    return torch.einsum("ij,...jhw->...ihw", matrix.to(image), image)


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    return _apply(YCC_FROM_RGB, image) + OFFSET.to(image)


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    return _apply(RGB_FROM_YCC, image - OFFSET.to(image))
