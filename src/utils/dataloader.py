"""Imagenette data pipeline."""
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets
from torchvision.transforms import v2


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
try:
    DATA_ROOT = Path(__file__).resolve().parents[2] / "data"   # <repo>/data
except NameError:                                              # interactive session, no __file__
    DATA_ROOT = Path.cwd() / "data"

IMAGE_SIZE = 224            # model input resolution
MEAN = (0.5, 0.5, 0.5)      # baseline normalisation -> pixels in [-1, 1]
STD = (0.5, 0.5, 0.5)
BATCH_SIZE = 32
NUM_WORKERS = 4
MAX_EPOCH_SAMPLES = 50_000  # None = full pass over the training set

# A colour-space conversion: RGB (C, H, W) in [0, 1] -> converted (C, H, W) in [0, 1].
ColourFn = Callable[[torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
# PIL image -> float tensor (C, H, W) in [0, 1], still at full resolution.
to_float = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

# Resize + crop + normalise. antialias=False because the baseline uses cv2
# bilinear interpolation, which does not antialias.
resize_and_normalise = v2.Compose([
    v2.Resize(IMAGE_SIZE + 32, interpolation=v2.InterpolationMode.BILINEAR, antialias=False),
    v2.CenterCrop(IMAGE_SIZE),
    v2.Normalize(MEAN, STD),
])


def preprocess(image: torch.Tensor, colour: ColourFn | None = None) -> torch.Tensor:
    """Full-resolution RGB in [0, 1] -> (optional colour conversion) -> model-ready tensor in [-1, 1]."""
    if colour is not None:
        image = colour(image)
    return resize_and_normalise(image)


def denormalise(image: torch.Tensor) -> torch.Tensor:
    """Undo the normalisation: [-1, 1] -> [0, 1]. Works on (C, H, W) and (B, C, H, W)."""
    mean = torch.tensor(MEAN, device=image.device).view(-1, 1, 1)
    std = torch.tensor(STD, device=image.device).view(-1, 1, 1)
    return image * std + mean


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ImagenetteDataset(datasets.Imagenette):
    """Imagenette returning (input, target, label).

    `in_colour` / `out_colour` are optional colour conversions (see `ColourFn`) for
    the model input and the reconstruction target. They are applied to the
    full-resolution image, before resizing, as in the baseline. With neither set,
    input and target are the same RGB image.

    Note: with num_workers > 0 on macOS, workers are started with "spawn", so the
    colour functions must be importable top-level functions (no lambdas or
    notebook-local defs).
    """

    def __init__(self, root: str | Path = DATA_ROOT, split: str = "train", in_colour: ColourFn | None = None, out_colour: ColourFn | None = None) -> None:
        super().__init__(root, split=split, size="full", transform=to_float)
        self.in_colour = in_colour
        self.out_colour = out_colour

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image, label = super().__getitem__(index)  # full-resolution RGB in [0, 1]

        image_in = preprocess(image, self.in_colour)
        if self.out_colour is self.in_colour:      # same conversion -> reuse, skip the second resize
            image_out = image_in
        else:
            image_out = preprocess(image, self.out_colour)

        return image_in, image_out, label

    def show(self, index: int) -> None:
        """Plot the target image of sample `index` (only meaningful when the target is RGB)."""
        import matplotlib.pyplot as plt

        _, image_out, label = self[index]
        plt.imshow(denormalise(image_out).clamp(0, 1).permute(1, 2, 0))
        plt.title(self.classes[label][0])
        plt.axis("off")
        plt.show()


# ---------------------------------------------------------------------------
# Train / val loaders
# ---------------------------------------------------------------------------
class ImagenetteDataLoader:
    """Builds the train and val datasets and their DataLoaders.

    Attributes: `train_data`, `val_data`, `train_loader`, `val_loader`.
    The train loader draws a fresh random subset of `max_epoch_samples` images
    each epoch (like the baseline's early `break`); the val loader is sequential.
    """

    def __init__(self, root: str | Path = DATA_ROOT, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS, max_epoch_samples: int | None = MAX_EPOCH_SAMPLES, in_colour: ColourFn | None = None, out_colour: ColourFn | None = None) -> None:
        self.train_data = ImagenetteDataset(root, "train", in_colour, out_colour)
        self.val_data = ImagenetteDataset(root, "val", in_colour, out_colour)

        n_train = len(self.train_data)
        epoch_size = n_train if max_epoch_samples is None else min(max_epoch_samples, n_train)
        train_sampler = RandomSampler(self.train_data, num_samples=epoch_size)

        loader_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

        self.train_loader = DataLoader(self.train_data, sampler=train_sampler, **loader_kwargs)
        self.val_loader = DataLoader(self.val_data, **loader_kwargs)


# ---------------------------------------------------------------------------
# Quick check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = ImagenetteDataLoader(num_workers=0)
    print(f"train: {len(data.train_data)} | val: {len(data.val_data)}")
    print(f"train batches/epoch: {len(data.train_loader)} | val batches: {len(data.val_loader)}")

    images_in, images_out, labels = next(iter(data.train_loader))
    print(f"input:  {tuple(images_in.shape)} {images_in.dtype} [{images_in.min():.2f}, {images_in.max():.2f}]")
    print(f"target: {tuple(images_out.shape)} {images_out.dtype} [{images_out.min():.2f}, {images_out.max():.2f}]")
    print(f"labels: {tuple(labels.shape)}")

    small = ImagenetteDataLoader(num_workers=0, max_epoch_samples=1000)
    print(f"capped epoch (1000 samples): {len(small.train_loader)} batches")
