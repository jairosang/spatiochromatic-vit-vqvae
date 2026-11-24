"""Colour-space registry.

Every space has a forward conversion from RGB and an inverse back to RGB. Both take
and return float tensors in [0, 1], shaped (C, H, W) or (B, C, H, W). RGB has no
conversion (None).

Pairs are written as in the DecomposeNet baseline: "<in>2<out>", e.g. "rgb2ycbcr"
means the model sees RGB and reconstructs YCbCr.
"""
from typing import Callable, NamedTuple

import torch

from src.colour_spaces.ycbcr import rgb_to_ycbcr, ycbcr_to_rgb

ColourFn = Callable[[torch.Tensor], torch.Tensor]


class ColourSpace(NamedTuple):
    to_space: ColourFn | None   # RGB [0, 1] -> this space [0, 1]
    to_rgb: ColourFn | None     # this space [0, 1] -> RGB [0, 1]


COLOUR_SPACES: dict[str, ColourSpace] = {
    "rgb": ColourSpace(None, None),
    "ycbcr": ColourSpace(rgb_to_ycbcr, ycbcr_to_rgb),
}


def parse_colour_pair(pair: str) -> tuple[str, str]:
    """"rgb2ycbcr" -> ("rgb", "ycbcr"), checking both names exist."""
    names = pair.lower().split("2")
    if len(names) != 2 or any(name not in COLOUR_SPACES for name in names):
        valid = ", ".join(COLOUR_SPACES)
        raise ValueError(f"Invalid colour pair {pair!r}: expected '<in>2<out>' with spaces from: {valid}")
    return names[0], names[1]
