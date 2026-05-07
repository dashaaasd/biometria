import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from config import get_train_transform, get_val_transform


def build_dataloaders(cfg):
    train_path = os.path.join(cfg.DATA_PATH, 'train')
    test_path  = os.path.join(cfg.DATA_PATH, 'test')

    assert os.path.exists(train_path), f"Нет папки train: {train_path}"
    assert os.path.exists(test_path),  f"Нет папки test:  {test_path}"

    train_transform = get_train_transform(cfg.IMAGE_SIZE)
    val_transform   = get_val_transform(cfg.IMAGE_SIZE)

    # Исправление: val_ds.dataset.transform = ... перезаписывает трансформ
    # для ВСЕГО full_train (Subset хранит ссылку, а не копию датасета).
    # Решение: два независимых ImageFolder с разными трансформами,
    # разбитые по одинаковым индексам.
    full_for_idx = ImageFolder(train_path)          # только для подсчёта размера
    n_total = len(full_for_idx)
    n_val   = int(n_total * cfg.VAL_SPLIT)
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(cfg.SEED)
    indices      = torch.randperm(n_total, generator=g).tolist()
    train_idx    = indices[:n_train]
    val_idx      = indices[n_train:]

    train_ds = Subset(ImageFolder(train_path, transform=train_transform), train_idx)
    val_ds   = Subset(ImageFolder(train_path, transform=val_transform),   val_idx)
    test_ds  = ImageFolder(test_path, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    print(f"Датасет: {cfg.DATA_PATH}")
    print(f"Размер изображений: {cfg.IMAGE_SIZE}×{cfg.IMAGE_SIZE}")
    print(f"Train: {n_train} | Val: {n_val} | Test: {len(test_ds)}")
    print(f"Классы: {full_for_idx.classes}")
    return train_loader, val_loader, test_loader
