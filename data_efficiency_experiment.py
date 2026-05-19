# data_efficiency_experiment.py
"""
Эксперимент: зависимость F1-macro от доли обучающей выборки
Сравнение CNN (GhostNetV2) и ViT на FER2013 и ExpW
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
import json
from datetime import datetime

# Твои модули
from config import ConfigFER, ConfigExpW
from data import build_dataloaders
from models import GhostNetV2_FER, ViT_FER
from utils import set_seed, get_scheduler, run_epoch


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def get_labels(dataset):
    """
    Извлекает метки из датасета.
    Работает с ImageFolder, Subset(ImageFolder), Subset(Subset(...))
    """
    # Идём вглубь Subset-ов пока не найдём ImageFolder
    current = dataset
    indices_list = []
    while hasattr(current, 'indices'):
        indices_list.append(current.indices)
        current = current.dataset
    
    # current теперь ImageFolder с .targets
    targets = np.array(current.targets)
    
    # Применяем цепочку индексов в обратном порядке
    for ind in reversed(indices_list):
        targets = targets[ind]
    
    return targets


def create_stratified_subset(dataset, fraction, seed=42):
    """Стратифицированная подвыборка с сохранением баланса классов"""
    if fraction >= 1.0:
        return dataset
    
    from sklearn.model_selection import train_test_split
    
    labels = get_labels(dataset)
    n_total = len(labels)
    n_samples = max(int(n_total * fraction), 7)  # минимум по 1 на класс
    
    indices = np.arange(n_total)
    selected, _ = train_test_split(
        indices,
        train_size=n_samples,
        stratify=labels,
        random_state=seed
    )
    return Subset(dataset, selected)


def compute_f1(model, loader, device):
    """F1-macro на всём лоадере"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return f1_score(all_labels, all_preds, average='macro', zero_division=0)


# ---------------------------------------------------------------------------
# Обучение на одной доле
# ---------------------------------------------------------------------------

def train_on_fraction(cfg, dataset_name, model_type, fraction,
                      train_dataset, val_loader):
    """
    Обучает модель на fraction*100% данных.
    Возвращает dict с результатами.
    """
    
    # Создаём подвыборку
    train_subset = create_stratified_subset(train_dataset, fraction, cfg.SEED)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    
    n_train = len(train_subset)
    print(f"\n{'='*55}")
    print(f"  {model_type.upper()} | {dataset_name} | {fraction*100:.0f}% | n={n_train}")
    print(f"{'='*55}")
    
    # Модель
    dropout_val = 0.3 if cfg.IMAGE_SIZE <= 48 else 0.5
    
    if model_type == 'cnn':
        model = GhostNetV2_FER(
            num_classes=cfg.NUM_CLASSES,
            dropout=dropout_val,
            img_size=cfg.IMAGE_SIZE,
            in_channels=cfg.IN_CHANNELS
        )
    else:
        model = ViT_FER(
            img_size=cfg.IMAGE_SIZE,
            patch_size=cfg.PATCH_SIZE,
            in_channels=cfg.IN_CHANNELS,
            num_classes=cfg.NUM_CLASSES,
            embed_dim=cfg.EMBED_DIM,
            depth=cfg.DEPTH,
            num_heads=cfg.NUM_HEADS,
            dropout=0.1
        )
    
    model = model.to(cfg.DEVICE)
    
    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Оптимизатор
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )
    
    warmup = max(2, cfg.NUM_EPOCHS // 10)
    scheduler = get_scheduler(optimizer, warmup_epochs=warmup, total_epochs=cfg.NUM_EPOCHS)
    
    use_amp = cfg.DEVICE.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # Обучение
    best_f1 = 0.0
    best_state = None
    patience = 12
    patience_counter = 0
    
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            cfg.DEVICE, use_amp, training=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, scaler,
            cfg.DEVICE, use_amp, training=False
        )
        
        val_f1 = compute_f1(model, val_loader, cfg.DEVICE)
        scheduler.step()
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | Loss {train_loss:.3f} | "
                  f"Acc {train_acc:.1f}% | Val {val_acc:.1f}% | F1 {val_f1:.4f}")
        
        if patience_counter >= patience:
            print(f"  Early stop @ epoch {epoch}")
            break
    
    # Финальная оценка
    model.load_state_dict(best_state)
    final_f1 = compute_f1(model, val_loader, cfg.DEVICE)
    
    print(f"  ✅ F1={final_f1:.4f}")
    
    return {
        'fraction': fraction,
        'n_train': n_train,
        'f1_macro': float(final_f1),
    }


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def main():
    set_seed(42)
    
    # Конфиги для экспериментов
    experiments = [
        (ConfigFER,  'FER-2013'),
        (ConfigExpW, 'ExpW'),
    ]
    
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    all_results = {}
    
    os.makedirs('experiments', exist_ok=True)
    
    for cfg, ds_name in experiments:
        print(f"\n{'#'*60}")
        print(f"# ЗАГРУЗКА: {ds_name}")
        print(f"{'#'*60}")
        
        # Твоя функция загрузки
        train_loader, val_loader, test_loader = build_dataloaders(cfg)
        
        train_dataset = train_loader.dataset  # Subset(ImageFolder)
        
        print(f"  Train: {len(train_dataset)} | Val: {len(val_loader.dataset)}")
        
        for model_type in ['cnn', 'vit']:
            print(f"\n  {'─'*45}")
            print(f"  >>> {model_type.upper()} на {ds_name}")
            print(f"  {'─'*45}")
            
            results = {}
            for frac in fractions:
                res = train_on_fraction(
                    cfg, ds_name, model_type,
                    frac, train_dataset, val_loader
                )
                results[frac] = res
            
            all_results[f'{model_type}_{ds_name}'] = results
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # --- Итоговая таблица ---
    print(f"\n\n{'='*80}")
    print("  Таблица 2. Зависимость F1-macro от доли обучающей выборки")
    print(f"{'='*80}")
    header = f"{'Модель / датасет':<20}"
    for f in fractions:
        header += f" {f*100:>6.0f}%"
    header += f"  {'Точка равенства':<15}"
    print(header)
    print(f"{'─'*80}")
    
    for ds_name in ['FER-2013', 'ExpW']:
        cnn_key = f'cnn_{ds_name}'
        vit_key = f'vit_{ds_name}'
        
        # CNN строка
        if cnn_key in all_results:
            row = f"CNN — {ds_name:<13}"
            for f in fractions:
                f1 = all_results[cnn_key][f]['f1_macro']
                row += f" {f1*100:>6.1f}"
            row += f"  {'—':<15}"
            print(row)
        
        # ViT строка
        if vit_key in all_results:
            row = f"ViT — {ds_name:<13}"
            for f in fractions:
                f1 = all_results[vit_key][f]['f1_macro']
                row += f" {f1*100:>6.1f}"
            
            # Точка равенства
            cnn_f1 = [all_results[cnn_key][f]['f1_macro'] for f in fractions]
            vit_f1 = [all_results[vit_key][f]['f1_macro'] for f in fractions]
            
            eq = None
            for i, f in enumerate(fractions):
                if vit_f1[i] >= cnn_f1[i]:
                    eq = f
                    break
            
            row += f"  {eq*100:.0f}%" if eq else f"  {'не догоняет'}"
            print(row)
        print()
    
    print(f"{'─'*80}")
    print("* «Точка равенства» — доля выборки, при которой ViT догоняет CNN по F1-macro.")
    
    # Сохраняем JSON
    output = {}
    for k, v in all_results.items():
        output[k] = {str(f): {'f1_macro': r['f1_macro'], 'n_train': r['n_train']}
                     for f, r in v.items()}
    
    with open('experiments/fraction_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Сохранено: experiments/fraction_results.json")


if __name__ == '__main__':
    main()