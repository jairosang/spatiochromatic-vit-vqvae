import torch


def get_device() -> torch.device:
    """Pick the best available accelerator: CUDA (cluster), MPS (Apple Silicon), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
