"""
Таблица 3. Зависимость F1-macro от доли обучающей выборки.

Обучает CNN и ViT на подмножествах 10 / 25 / 50 / 75 / 100 %
обучающей выборки для каждого датасета и записывает результаты.

Запуск:
    py subset_experiment.py

Результаты:
    subset_results.csv  — числа для таблицы
    subset_plot.png     — график для статьи
"""
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.metrics import f1_score

from config import (ConfigFER, ConfigExpW,
                    get_train_transform, get_val_transform,
                    get_train_transform_full)
from models import GhostNetV2_FER, ViT_FER
from utils import set_seed, get_scheduler, run_epoch


# ── Настройки эксперимента ────────────────────────────────────────────────────
SUBSETS    = [0.10, 0.25, 0.50, 0.75]
SEEDS      = [42]           
OUTPUT_CSV = 'subset_results.csv'
OUTPUT_PNG = 'subset_plot.png'

DATASETS = [
    ('FER-2013', ConfigFER),
    ('ExpW',     ConfigExpW),
]


# ── Вспомогательные функции ───────────────────────────────────────────────────

def compute_weights(subset_idx, dataset, num_classes, device):
    targets = [dataset.targets[i] for i in subset_idx]
    counts  = np.bincount(targets, minlength=num_classes).astype(np.float32)
    w       = np.sqrt(counts.sum() / (num_classes * counts))
    return torch.tensor(w, dtype=torch.float32).to(device)


def build_loaders(cfg, subset_ratio, seed):
    """
    Строит три DataLoader-а.
    Val и test — полные; train — subset_ratio от обучающей части.
    """
    train_path = os.path.join(cfg.DATA_PATH, 'train')
    test_path  = os.path.join(cfg.DATA_PATH, 'test')

    train_tf = get_train_transform(cfg.IMAGE_SIZE, cfg.IN_CHANNELS)
    val_tf   = get_val_transform(cfg.IMAGE_SIZE,   cfg.IN_CHANNELS)

    # Индексы train/val (фиксированы через cfg.SEED)
    full     = ImageFolder(train_path)
    n_total  = len(full)
    n_val    = int(n_total * cfg.VAL_SPLIT)
    n_train  = n_total - n_val
    g        = torch.Generator().manual_seed(cfg.SEED)
    all_idx  = torch.randperm(n_total, generator=g).tolist()
    train_idx = all_idx[:n_train]
    val_idx   = all_idx[n_train:]

    # Подмножество train
    n_sub     = max(7, int(len(train_idx) * subset_ratio))  # >= 7 (по одному на класс)
    rng       = np.random.default_rng(seed)
    sub_idx   = rng.choice(train_idx, size=n_sub, replace=False).tolist()

    train_ds  = Subset(ImageFolder(train_path, transform=train_tf), sub_idx)
    val_ds    = Subset(ImageFolder(train_path, transform=val_tf),   val_idx)
    test_ds   = ImageFolder(test_path, transform=val_tf)

    kw = dict(num_workers=cfg.NUM_WORKERS, pin_memory=True,
              persistent_workers=cfg.NUM_WORKERS > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.BATCH_SIZE,
                              shuffle=False, **kw)

    weights = compute_weights(sub_idx, train_ds.dataset,
                              cfg.NUM_CLASSES, cfg.DEVICE)
    print(f"    {subset_ratio*100:.0f}%: {n_sub} train / "
          f"{len(val_idx)} val / {len(test_ds)} test")
    return train_loader, val_loader, test_loader, weights


def get_f1(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            preds_all.extend(model(imgs).argmax(1).cpu().numpy())
            labels_all.extend(lbls.numpy())
    return f1_score(labels_all, preds_all, average='macro') * 100


def train_one(model, train_loader, val_loader, criterion, cfg, seed):
    """Обучает модель, возвращает веса лучшей эпохи по val_f1."""
    set_seed(seed)
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.LEARNING_RATE,
                            weight_decay=cfg.WEIGHT_DECAY)
    warmup    = max(2, cfg.NUM_EPOCHS // 10)
    scheduler = get_scheduler(optimizer, warmup_epochs=warmup,
                              total_epochs=cfg.NUM_EPOCHS)
    use_amp   = cfg.DEVICE.type == 'cuda'
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_f1, best_state = 0.0, None

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        if epoch == 7:
            train_loader.dataset.dataset.transform = \
                get_train_transform_full(cfg.IMAGE_SIZE, cfg.IN_CHANNELS)

        # run_epoch теперь возвращает 3 значения: loss, accuracy, f1
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion,
            optimizer, scaler, cfg.DEVICE, use_amp, 
            training=True, num_classes=cfg.NUM_CLASSES
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion,
            optimizer, scaler, cfg.DEVICE, use_amp,
            training=False, num_classes=cfg.NUM_CLASSES
        )
        scheduler.step()

        # Сохраняем по F1, а не по accuracy
        if val_f1 > best_f1:
            best_f1    = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def find_crossover(cnn_vals, vit_vals, subsets):
    """
    Находит 'точку равенства' — долю выборки, при которой ViT
    впервые догоняет CNN по F1-macro. Возвращает строку или 'не достигнута'.
    """
    for i in range(len(subsets)):
        if vit_vals[i] >= cnn_vals[i]:
            return f"{subsets[i]*100:.0f}%"
    return "не достигнута"


# ── Основной эксперимент ──────────────────────────────────────────────────────

def run_experiment():
    all_results = []

    for ds_name, cfg in DATASETS:
        print(f"\n{'='*60}")
        print(f"  Датасет: {ds_name}  |  {cfg.IMAGE_SIZE}x{cfg.IMAGE_SIZE}"
              f"  |  ch={cfg.IN_CHANNELS}")
        print(f"{'='*60}")

        cnn_means, vit_means = [], []
        cnn_stds,  vit_stds  = [], []

        for ratio in SUBSETS:
            cnn_f1s, vit_f1s = [], []

            for seed in SEEDS:
                print(f"\n  ratio={ratio*100:.0f}%  seed={seed}")
                train_loader, val_loader, test_loader, weights = \
                    build_loaders(cfg, ratio, seed)
                criterion = nn.CrossEntropyLoss(
                    weight=weights, label_smoothing=0.1
                )

                # CNN
                set_seed(seed)
                cnn = GhostNetV2_FER(
                    num_classes=cfg.NUM_CLASSES, dropout=cfg.DROPOUT,
                    img_size=cfg.IMAGE_SIZE, in_channels=cfg.IN_CHANNELS,
                ).to(cfg.DEVICE)
                cnn = train_one(cnn, train_loader, val_loader,
                                criterion, cfg, seed)
                f1c = get_f1(cnn, test_loader, cfg.DEVICE)
                cnn_f1s.append(f1c)
                print(f"    CNN  F1-macro = {f1c:.2f}%")
                del cnn; torch.cuda.empty_cache()

                # ViT
                set_seed(seed)
                vit = ViT_FER(
                    img_size=cfg.IMAGE_SIZE, patch_size=cfg.PATCH_SIZE,
                    in_channels=cfg.IN_CHANNELS, num_classes=cfg.NUM_CLASSES,
                    embed_dim=cfg.EMBED_DIM, depth=cfg.DEPTH,
                    num_heads=cfg.NUM_HEADS, dropout=cfg.VIT_DROPOUT,
                ).to(cfg.DEVICE)
                vit = train_one(vit, train_loader, val_loader,
                                criterion, cfg, seed)
                f1v = get_f1(vit, test_loader, cfg.DEVICE)
                vit_f1s.append(f1v)
                print(f"    ViT  F1-macro = {f1v:.2f}%")
                del vit; torch.cuda.empty_cache()

            cnn_means.append(np.mean(cnn_f1s))
            cnn_stds.append(np.std(cnn_f1s))
            vit_means.append(np.mean(vit_f1s))
            vit_stds.append(np.std(vit_f1s))

            all_results.append({
                'dataset':  ds_name,
                'subset':   f"{ratio*100:.0f}%",
                'cnn_mean': np.mean(cnn_f1s),
                'cnn_std':  np.std(cnn_f1s),
                'vit_mean': np.mean(vit_f1s),
                'vit_std':  np.std(vit_f1s),
            })

        crossover = find_crossover(cnn_means, vit_means, SUBSETS)
        print(f"\n  Точка равенства ({ds_name}): {crossover}")
        for r in all_results[-len(SUBSETS):]:
            r['crossover'] = crossover if r['subset'] == SUBSETS[-1] else ''

    # ── CSV ───────────────────────────────────────────────────────────────────
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV сохранён: {OUTPUT_CSV}")

    # ── Итоговая таблица в консоли ────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  ТАБЛИЦА 3. Зависимость F1-macro (%) от доли обучающей выборки")
    print(f"{'='*72}")
    header = f"{'Модель / датасет':<22}" + "".join(
        f"{int(s*100):>8}%" for s in SUBSETS
    ) + f"  {'Точка равенства':>16}"
    print(header)
    print("-" * 72)

    for ds_name, _ in DATASETS:
        rows = [r for r in all_results if r['dataset'] == ds_name]
        crossover = next(
            (r['crossover'] for r in rows if r.get('crossover')), '—'
        )
        cnn_row = f"{'CNN — ' + ds_name:<22}" + "".join(
            f"{r['cnn_mean']:>9.1f}" for r in rows
        ) + f"  {crossover:>16}"
        vit_row = f"{'ViT — ' + ds_name:<22}" + "".join(
            f"{r['vit_mean']:>9.1f}" for r in rows
        )
        print(cnn_row)
        print(vit_row)
        print()

    # ── График ────────────────────────────────────────────────────────────────
    pct = [int(s * 100) for s in SUBSETS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, (ds_name, _) in zip(axes, DATASETS):
        rows     = [r for r in all_results if r['dataset'] == ds_name]
        cnn_m    = [r['cnn_mean'] for r in rows]
        vit_m    = [r['vit_mean'] for r in rows]
        cnn_s    = [r['cnn_std']  for r in rows]
        vit_s    = [r['vit_std']  for r in rows]

        ax.plot(pct, cnn_m, 'o-', label='CNN (GhostNetV2)', linewidth=2)
        ax.fill_between(pct,
                        [m - s for m, s in zip(cnn_m, cnn_s)],
                        [m + s for m, s in zip(cnn_m, cnn_s)],
                        alpha=0.15)
        ax.plot(pct, vit_m, 's--', label='ViT', linewidth=2)
        ax.fill_between(pct,
                        [m - s for m, s in zip(vit_m, vit_s)],
                        [m + s for m, s in zip(vit_m, vit_s)],
                        alpha=0.15)

        ax.set_title(ds_name, fontsize=13)
        ax.set_xlabel('Доля обучающей выборки (%)')
        ax.set_ylabel('F1-macro (%)')
        ax.set_xticks(pct)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle('Зависимость F1-macro от объёма обучающих данных',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"График сохранён: {OUTPUT_PNG}")


if __name__ == '__main__':
    run_experiment()