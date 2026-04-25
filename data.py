import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from config import Config, get_train_transform, get_val_transform


def build_dataloaders(cfg):
    train_path = os.path.join(cfg.DATA_PATH, 'train')
    test_path = os.path.join(cfg.DATA_PATH, 'test')

    assert os.path.exists(train_path), f"Нет папки train: {train_path}"
    assert os.path.exists(test_path), f"Нет папки test: {test_path}"

    train_transform = get_train_transform(cfg.IMAGE_SIZE)
    val_transform = get_val_transform(cfg.IMAGE_SIZE)

    full_train = ImageFolder(train_path, transform=train_transform)
    test_ds = ImageFolder(test_path, transform=val_transform)

    n_val = int(len(full_train) * cfg.VAL_SPLIT)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.SEED)
    )
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE,
                            shuffle=False, num_workers=cfg.NUM_WORKERS,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE,
                             shuffle=False, num_workers=cfg.NUM_WORKERS,
                             pin_memory=True)

    print(f"Датасет: {cfg.DATA_PATH}")
    print(f"Размер изображений: {cfg.IMAGE_SIZE}×{cfg.IMAGE_SIZE}")
    print(f"Train: {n_train} | Val: {n_val} | Test: {len(test_ds)}")
    print(f"Классы: {full_train.classes}")
    return train_loader, val_loader, test_loader