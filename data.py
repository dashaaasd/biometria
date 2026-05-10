import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from config import get_train_transform, get_val_transform


def compute_class_weights(dataset, num_classes, device):
    """
    Вычисляет веса классов обратно пропорционально их частоте.
    Используется для борьбы с дисбалансом классов в ExpW и GFFD.

    Формула: weight[c] = N_total / (num_classes * N_c)
    Редкие классы (disgust, fear) получают большой вес,
    частые (happy, neutral) — маленький.
    """
    # Считаем количество примеров каждого класса
    if hasattr(dataset, 'dataset'):
        # Subset — берём таргеты только по нашим индексам
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        targets = dataset.targets

    targets  = np.array(targets)
    counts   = np.bincount(targets, minlength=num_classes).astype(np.float32)
    total    = counts.sum()
    weights  = np.sqrt(total / (num_classes * counts))
    weights  = torch.tensor(weights, dtype=torch.float32).to(device)

    print(f"  Веса классов: {[f'{w:.2f}' for w in weights.cpu().numpy()]}")
    return weights


def build_dataloaders(cfg):
    train_path = os.path.join(cfg.DATA_PATH, 'train')
    test_path  = os.path.join(cfg.DATA_PATH, 'test')

    assert os.path.exists(train_path), f"Нет папки train: {train_path}"
    assert os.path.exists(test_path),  f"Нет папки test:  {test_path}"

    train_transform = get_train_transform(cfg.IMAGE_SIZE, cfg.IN_CHANNELS)
    val_transform   = get_val_transform(cfg.IMAGE_SIZE, cfg.IN_CHANNELS)

    # Два независимых ImageFolder с разными трансформами,
    # разбитые по одинаковым индексам — исправляет баг с val transform
    full_for_idx = ImageFolder(train_path)
    n_total      = len(full_for_idx)
    n_val        = int(n_total * cfg.VAL_SPLIT)
    n_train      = n_total - n_val

    g       = torch.Generator().manual_seed(cfg.SEED)
    indices = torch.randperm(n_total, generator=g).tolist()

    train_ds = Subset(ImageFolder(train_path, transform=train_transform), indices[:n_train])
    val_ds   = Subset(ImageFolder(train_path, transform=val_transform),   indices[n_train:])
    test_ds  = ImageFolder(test_path, transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
        persistent_workers=cfg.NUM_WORKERS > 0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
        persistent_workers=cfg.NUM_WORKERS > 0
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
        persistent_workers=cfg.NUM_WORKERS > 0
    )

    print(f"Датасет: {cfg.DATA_PATH}")
    print(f"Размер изображений: {cfg.IMAGE_SIZE}x{cfg.IMAGE_SIZE}")
    print(f"Train: {n_train} | Val: {n_val} | Test: {len(test_ds)}")
    print(f"Классы: {full_for_idx.classes}")

    return train_loader, val_loader, test_loader