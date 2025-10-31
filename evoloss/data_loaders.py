from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def _split_dataset(ds, val_ratio: float = 0.2) -> Tuple[Any, Any]:
    val_size = int(val_ratio * len(ds))
    train_size = len(ds) - val_size
    return random_split(ds, [train_size, val_size])


def load_fashion_mnist(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    data_dir = cfg.get("data_dir", "./data")
    batch_size = int(cfg.get("batch_size", cfg.get("dataset", {}).get("batch_size", 64)))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    # Лимит выборки при быстрой проверке
    samples = int(cfg.get("dataset", {}).get("samples", 0))
    if samples and samples > 0:
        # Делим на train/val по соотношению 80/20, но не превышаем размер набора
        total = len(ds)
        train_size = min(int(samples * 0.8), total)
        val_size = min(samples - int(samples * 0.8), total - train_size)
        # Если val_size получилось 0, выделим хотя бы 1% от train_size
        if val_size <= 0:
            val_size = max(1, int(train_size * 0.2))
            train_size = min(train_size, total - val_size)
        train_ds, val_ds = random_split(ds, [train_size, val_size])
    else:
        train_ds, val_ds = _split_dataset(ds, val_ratio=0.2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def load_cifar10_gray28(cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    CIFAR10 загрузчик, приводящий изображения к формату 1x28x28 (grayscale + resize),
    чтобы использовать существующую SimpleCNN без изменений.
    """
    data_dir = cfg.get("data_dir", "./data")
    batch_size = int(cfg.get("batch_size", cfg.get("dataset", {}).get("batch_size", 64)))
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    samples = int(cfg.get("dataset", {}).get("samples", 0))
    if samples and samples > 0:
        total = len(ds)
        train_size = min(int(samples * 0.8), total)
        val_size = min(samples - int(samples * 0.8), total - train_size)
        if val_size <= 0:
            val_size = max(1, int(train_size * 0.2))
            train_size = min(train_size, total - val_size)
        train_ds, val_ds = random_split(ds, [train_size, val_size])
    else:
        train_ds, val_ds = _split_dataset(ds, val_ratio=0.2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader