"""
src/data.py
---
CIFAR-10 data loading and preprocessing
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 channel statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def get_transforms(train: bool) -> transforms.Compose:
    """
    Returns the torchvision transform pipeline for splits

    Train augmentations (RandomCrop + RandomHorizontalFlip) match what was used for AlexNet

    Test transforms are deterministic: resize (no-op on 32x32) + normalize
    """
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    
def get_dataloaders(params: dict) -> tuple[DataLoader, DataLoader]:
    """
    Downloads and returns (train_loader, test_loader)

    Parameters
    ----------
    params : dict
        The parsed params.yaml dict
        Reads from params["data"] and params["train"] sections

    Returns
    --------
    train_loader, test_loader : DataLoader
        Ready-to-iterate PyTorch DataLoaders
    """
    data_params = params["data"]
    train_params = params["train"]

    data_dir = data_params["data_dir"]
    batch_size = train_params["batch_size"]
    num_workers = data_params["num_workers"]
    device_str = train_params["device"]

    pin_memory = device_str == "cuda"

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True)
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, test_loader

def get_class_names() -> list[str]:
    """Returns the 10 CIFAR-10 class labels in label-index order"""
    return [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]


# STANDALONE TEST
def main():
    import os
    import yaml
    import logging
    from typing import cast, Sized

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    _logger = logging.getLogger(__name__)

    params_path = os.path.join(os.path.dirname(__file__), "..", "params.yaml")
    with open(params_path) as f:
        params = yaml.safe_load(f)

    _logger.info("Loading CIFAR-10...")
    train_loader, test_loader = get_dataloaders(params)

    # dataset sizes
    _logger.info(f"\n  Training samples : {len(cast(Sized, train_loader.dataset)):,}")
    _logger.info(f"  Test samples     : {len(cast(Sized, test_loader.dataset)):,}")
    _logger.info(f"  Batch size       : {params['train']['batch_size']}")
    _logger.info(f"  Train batches    : {len(train_loader)}")
    _logger.info(f"  Test batches     : {len(test_loader)}")

    # sample batch shape
    images, labels = next(iter(train_loader))
    _logger.info(f"\n  Sample batch — images : {tuple(images.shape)}")
    _logger.info(f"  Sample batch — labels : {tuple(labels.shape)}")
    _logger.info(f"  Pixel value range     : [{images.min():.3f}, {images.max():.3f}]")
    _logger.info(f"  Classes               : {get_class_names()}")

    # verify normalization
    per_channel_mean = images.mean(dim=(0, 2, 3))
    _logger.info(f"\n  Per-channel mean (should be ≈ 0): {per_channel_mean.tolist()}")

    _logger.info("\n✅ data.py smoke-test passed.")

if __name__ == "__main__":
    main()