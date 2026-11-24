"""Image data pipeline

Supported datasets and the folder layout expected under `root`:
  - "imagenette":  <root>/imagenette2/{train,val}/<wnid>/*.JPEG   (root defaults to <repo>/data)
  - "imagenet":    <root>/{train,val}/<wnid>/*.JPEG
  - "imagenet100": <root>/{train,val}/<wnid>/*.JPEG               (only the 100 class folders)
  - "coco":        <root>/{train2017,val2017}/*.jpg               (no class labels -> label = -1)
  The official ImageNet val images come in one flat folder; sort them into one folder per
  class first (e.g. with the standard valprep script), as ImageFolder needs that layout.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import datasets
from torchvision.datasets.folder import default_loader
from torchvision.transforms import v2

from src.colour_spaces import COLOUR_SPACES, ColourFn, parse_colour_pair


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

DATASETS = ("imagenette", "imagenet", "imagenet100", "coco")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


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
# Source datasets: each yields (full-resolution RGB tensor in [0, 1], label)
# ---------------------------------------------------------------------------
class FlatImageFolder(Dataset):
    """All images in a single folder, without labels (label = -1). Used for COCO."""

    def __init__(self, folder: str | Path, transform=to_float) -> None:
        self.paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {folder}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = default_loader(str(self.paths[index]))   # always RGB (COCO has some grayscale images)
        return self.transform(image), -1


def build_base(name: str, root: str | Path, split: str) -> Dataset:
    """Source dataset `name` for `split` ("train" or "val")."""
    root = Path(root)
    if name == "imagenette":
        return datasets.Imagenette(root, split=split, size="full", transform=to_float)
    if name in ("imagenet", "imagenet100"):
        return datasets.ImageFolder(root / split, transform=to_float)
    if name == "coco":
        return FlatImageFolder(root / f"{split}2017")
    raise ValueError(f"Unknown dataset {name!r}: expected one of {DATASETS}")


# ---------------------------------------------------------------------------
# Input/target pairs
# ---------------------------------------------------------------------------
class PairedColourDataset(Dataset):
    """Wraps a source dataset and returns (input, target, label).

    `colour` is a pair such as "rgb2ycbcr": the input is in the first space and the
    reconstruction target in the second. Conversions run on the full-resolution
    image, before resizing, as in the baseline.
    """

    def __init__(self, base: Dataset, colour: str = "rgb2rgb") -> None:
        self.base = base
        self.colour = colour
        self.in_space, self.out_space = parse_colour_pair(colour)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image, label = self.base[index]   # full-resolution RGB in [0, 1]

        image_in = preprocess(image, COLOUR_SPACES[self.in_space].to_space)
        if self.out_space == self.in_space:   # same space -> reuse, skip the second resize
            image_out = image_in
        else:
            image_out = preprocess(image, COLOUR_SPACES[self.out_space].to_space)

        return image_in, image_out, label

    def show(self, index: int) -> None:
        """Plot the target image of sample `index`, converted back to RGB."""
        import matplotlib.pyplot as plt

        _, image_out, label = self[index]
        image = denormalise(image_out)
        to_rgb = COLOUR_SPACES[self.out_space].to_rgb
        if to_rgb is not None:
            image = to_rgb(image)

        plt.imshow(image.clamp(0, 1).permute(1, 2, 0))
        classes = getattr(self.base, "classes", None)
        if classes and label >= 0:
            name = classes[label]
            plt.title(name[0] if isinstance(name, tuple) else name)   # Imagenette gives tuples of synonyms
        plt.axis("off")
        plt.show()


# ---------------------------------------------------------------------------
# Train / val loaders
# ---------------------------------------------------------------------------
class ImageData:
    """Builds the train and val datasets and their DataLoaders for one dataset and colour pair.

    Attributes: `train_data`, `val_data`, `train_loader`, `val_loader`.
    `root` defaults to <repo>/data for Imagenette and <repo>/data/<dataset> otherwise;
    pass it explicitly for data stored elsewhere (e.g. on a cluster).
    The train loader draws a fresh random subset of `max_epoch_samples` images
    each epoch (like the baseline's early `break`); the val loader is sequential.
    """

    def __init__(self, dataset: str = "imagenette", root: str | Path | None = None, colour: str = "rgb2rgb", batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS, max_epoch_samples: int | None = MAX_EPOCH_SAMPLES) -> None:
        if root is None:
            root = DATA_ROOT if dataset == "imagenette" else DATA_ROOT / dataset

        self.train_data = PairedColourDataset(build_base(dataset, root, "train"), colour)
        self.val_data = PairedColourDataset(build_base(dataset, root, "val"), colour)

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
# Quick check: python -m src.utils.dataloader  (from the repo root)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = ImageData("imagenette", colour="rgb2ycbcr", num_workers=0)
    print(f"train: {len(data.train_data)} | val: {len(data.val_data)}")
    print(f"train batches/epoch: {len(data.train_loader)} | val batches: {len(data.val_loader)}")

    images_in, images_out, labels = next(iter(data.train_loader))
    print(f"input  (rgb):   {tuple(images_in.shape)} {images_in.dtype} [{images_in.min():.2f}, {images_in.max():.2f}]")
    print(f"target (ycbcr): {tuple(images_out.shape)} {images_out.dtype} [{images_out.min():.2f}, {images_out.max():.2f}]")
    print(f"labels: {tuple(labels.shape)}")

    small = ImageData("imagenette", num_workers=0, max_epoch_samples=1000)
    print(f"capped epoch (1000 samples): {len(small.train_loader)} batches")
