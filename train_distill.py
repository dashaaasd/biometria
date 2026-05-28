import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder

from config import Config
from data import build_dataloaders, compute_class_weights
from models.vit_distill import ViT_Distill
from utils import set_seed, get_scheduler, run_epoch


# ─── НАСТРОЙКИ ДЛЯ ЭКСПЕРИМЕНТА ───────────────────────────────────────────────
SUBSET_RATIO = 1.0   # ← МЕНЯЙ ЗДЕСЬ: 0.10, 0.25, 0.50, 0.75, 1.00 (1.00 = полная выборка)
# ──────────────────────────────────────────────────────────────────────────────


def load_teacher(cfg, device):
    """Загрузка предобученного GhostNet (учителя)"""
    try:
        from models import GhostNetV2_FER
    except ImportError:
        print("❌ Не удалось импортировать GhostNetV2_FER")
        return None
    
    teacher_path = os.path.join(cfg.MODEL_SAVE_PATH, cfg.MODEL_CNN_NAME)
    
    if not os.path.exists(teacher_path):
        print(f"❌ Модель учителя не найдена: {teacher_path}")
        return None
    
    teacher = GhostNetV2_FER(
        num_classes=cfg.NUM_CLASSES,
        dropout=cfg.DROPOUT,
        img_size=cfg.IMAGE_SIZE,      
        in_channels=cfg.IN_CHANNELS
    ).to(device)
    
    checkpoint = torch.load(teacher_path, map_location=device)
    if 'model_state' in checkpoint:
        teacher.load_state_dict(checkpoint['model_state'])
    elif 'model_state_dict' in checkpoint:
        teacher.load_state_dict(checkpoint['model_state_dict'])
    else:
        teacher.load_state_dict(checkpoint)
    
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    print(f"✅ Учитель загружен: {teacher_path}")
    return teacher


def build_subset_loaders(cfg, subset_ratio, seed):
    """
    Строит DataLoader для подвыборки train (subset_ratio от полного train).
    Val и test — полные.
    """
    from config import get_train_transform, get_val_transform
    
    train_path = os.path.join(cfg.DATA_PATH, 'train')
    
    # Полный train dataset (только для получения индексов)
    full_train_dataset = ImageFolder(train_path)
    n_total = len(full_train_dataset)
    
    # Фиксируем порядок индексов через seed
    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_total)
    rng.shuffle(all_indices)
    
    # Берём subset
    n_sub = max(7, int(n_total * subset_ratio))
    sub_indices = all_indices[:n_sub].tolist()
    
    # Создаём подвыборку с правильными transforms
    train_tf = get_train_transform(cfg.IMAGE_SIZE, cfg.IN_CHANNELS)
    train_ds = Subset(
        ImageFolder(train_path, transform=train_tf),
        sub_indices
    )
    
    # Val и test через стандартную функцию
    _, val_loader, test_loader = build_dataloaders(cfg)
    
    # DataLoader для подвыборки
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True
    )
    
    print(f"  Подвыборка: {subset_ratio*100:.0f}% = {n_sub} изображений")
    print(f"  Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    return train_loader, val_loader, test_loader


def train_one_distill(model, train_loader, val_loader, teacher, cfg, seed):
    """
    Обучает ViT_Distill с учителем.
    Использует ту же логику, что и train_one в train_vit.py
    """
    set_seed(seed)
    
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.LEARNING_RATE,
                            weight_decay=cfg.WEIGHT_DECAY)
    warmup = max(2, cfg.NUM_EPOCHS // 10)
    scheduler = get_scheduler(optimizer, warmup_epochs=warmup,
                              total_epochs=cfg.NUM_EPOCHS)
    use_amp = cfg.DEVICE.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    best_f1, best_state = 0.0, None
    
    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        # RandomErasing на 7 эпохе
        if epoch == 7:
            from config import get_train_transform_full
            if hasattr(train_loader.dataset, 'dataset'):
                train_loader.dataset.dataset.transform = \
                    get_train_transform_full(cfg.IMAGE_SIZE, cfg.IN_CHANNELS)
            print(f"  RandomErasing включён с эпохи {epoch}")
        
        # Training with distillation
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []
        
        pbar = tqdm(train_loader, desc=f'Train E{epoch}', leave=False)
        for images, labels in pbar:
            images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            with torch.no_grad():
                teacher_logits = teacher(images)
            
            with torch.autocast(device_type=cfg.DEVICE.type, enabled=use_amp):
                loss = model(images, teacher_logits=teacher_logits)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # Для подсчёта train_f1
            with torch.no_grad():
                logits = model(images)
                preds = logits.argmax(1).detach().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
            
            from sklearn.metrics import f1_score
            current_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            pbar.set_postfix(loss=f'{loss.item():.4f}', f1=f'{current_f1:.4f}')
        
        train_loss = total_loss / len(train_loader)
        
        # Валидация через run_epoch
        criterion = nn.CrossEntropyLoss()
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion,
            optimizer, scaler, cfg.DEVICE, use_amp,
            training=False, num_classes=cfg.NUM_CLASSES
        )
        
        scheduler.step()
        
        print(f'  Train: loss={train_loss:.4f}  f1_macro={current_f1*100:.2f}%')
        print(f'  Val:   loss={val_loss:.4f}  acc={val_acc:.2f}%  f1_macro={val_f1:.2f}%')
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    return model, best_f1


def main():
    cfg = Config
    set_seed(cfg.SEED)
    
    print("=" * 60)
    print(f"ОБУЧЕНИЕ ViT С ДИСТИЛЛЯЦИЕЙ")
    print(f"Датасет: {cfg.DATA_PATH}")
    print(f"Размер: {cfg.IMAGE_SIZE}x{cfg.IMAGE_SIZE}")
    print(f"Подвыборка: {SUBSET_RATIO*100:.0f}% от train")
    print(f"Устройство: {cfg.DEVICE}")
    print("=" * 60)
    
    # 1. Загрузка данных (полная или подвыборка)
    if SUBSET_RATIO >= 1.0:
        # Полная выборка
        train_loader, val_loader, test_loader = build_dataloaders(cfg)
    else:
        # Подвыборка
        train_loader, val_loader, test_loader = build_subset_loaders(cfg, SUBSET_RATIO, cfg.SEED)
    
    # 2. Загрузка учителя
    teacher = load_teacher(cfg, cfg.DEVICE)
    if teacher is None:
        print("❌ Нет учителя — выход")
        return
    
    # 3. Создание модели студента
    model = ViT_Distill(
        img_size=cfg.IMAGE_SIZE,
        patch_size=cfg.PATCH_SIZE,
        in_channels=cfg.IN_CHANNELS,
        num_classes=cfg.NUM_CLASSES,
        embed_dim=cfg.EMBED_DIM,
        depth=cfg.DEPTH,
        num_heads=cfg.NUM_HEADS,
        dropout=cfg.VIT_DROPOUT
    ).to(cfg.DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Студент ViT_Distill | Параметры: {num_params:,}")
    
    # 4. Обучение
    model, best_val_f1 = train_one_distill(model, train_loader, val_loader, teacher, cfg, cfg.SEED)
    
    # 5. Финальное тестирование
    print("\n" + "=" * 60)
    print("ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ")
    
    from sklearn.metrics import f1_score, accuracy_score
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Test'):
            images = images.to(cfg.DEVICE)
            logits = model(images)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    test_acc = accuracy_score(all_labels, all_preds) * 100
    test_f1_macro = f1_score(all_labels, all_preds, average='macro') * 100
    test_f1_weighted = f1_score(all_labels, all_preds, average='weighted') * 100
    
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test F1-macro: {test_f1_macro:.2f}%")
    print(f"Test F1-weighted: {test_f1_weighted:.2f}%")
    
    # Сохраняем модель
    subset_str = f"subset_{int(SUBSET_RATIO*100)}"
    save_path = os.path.join(cfg.MODEL_SAVE_PATH, f"{cfg.MODEL_VIT_NAME_DIS}_{subset_str}.pth")
    os.makedirs(cfg.MODEL_SAVE_PATH, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_f1': best_val_f1,
        'test_acc': test_acc,
        'test_f1_macro': test_f1_macro,
        'test_f1_weighted': test_f1_weighted,
        'subset_ratio': SUBSET_RATIO,
    }, save_path)
    print(f"\n✅ Модель сохранена: {save_path}")
    print(f"Лучшая val F1-macro: {best_val_f1:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main()