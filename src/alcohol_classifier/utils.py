import torch


def _set_seed(seed: int) -> None:
    print(f"Setting seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_device(device: str) -> torch.device:
    if device != "auto":
        print(f"Using device: {device}")
        return torch.device(device)
    if torch.backends.mps.is_available():
        print("Using device: mps")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using device: cuda")
        return torch.device("cuda")

    print("Using device: cpu")
    return torch.device("cpu")
